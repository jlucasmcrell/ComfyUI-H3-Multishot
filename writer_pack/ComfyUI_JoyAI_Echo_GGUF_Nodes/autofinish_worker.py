"""Detached finishing worker for JoyEcho multishot renders.

Spawned by JoyEcho_AutoFinish (see joyecho_autofinish.py). Runs OUTSIDE the
ComfyUI process so the render's queue item completes immediately.

Pipeline (all file-based - no giant tensors, no SaveVideo re-encode):
  1. SNAPSHOT the per-shot masters (shot_XXX.mp4 + .wav) into a run-scoped
     folder. Critical for queued back-to-back renders: our upscale jobs run
     AFTER any later queued render (FIFO), by which time the live shot_XXX
     files belong to THAT run. The snapshot pins this run's frames.
  2. Submit one RTXBatchVideoUpscale job per shot via the ComfyUI API
     (streams 80-frame chunks; jobs queue behind whatever is running).
  3. Pad each wav to its upscaled video's exact duration, lossless-concat
     both streams, mux with a single AAC encode.
  4. Write <name>_<stamp>_MASTER.mp4 next to the shot masters.

Log: <shots_dir>/_autofinish_<stamp>.log  (this file is the only place
errors surface - the node fires and forgets by design).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

# The pack's own bin/ is checked FIRST so machines without a system
# ffmpeg (the 3090 box shipped none - every master attempt there died
# with "FATAL: ffmpeg/ffprobe not found") work out of the box. Drop
# ffmpeg.exe + ffprobe.exe into <pack>/bin/ and the worker finds them.
_PACK_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
FFMPEG_CANDIDATES = [
    os.path.join(_PACK_BIN, "ffmpeg.exe"),
    r"C:\ffmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]
FFPROBE_CANDIDATES = [
    os.path.join(_PACK_BIN, "ffprobe.exe"),
    r"C:\ffmpeg\bin\ffprobe.exe",
    "ffprobe",
]


def _find(cands):
    for c in cands:
        if os.path.isfile(c) or shutil.which(c):
            return c
    return None


def api(base, path, data=None, timeout=30):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _probe_dur(fp, path):
    out = subprocess.run([fp, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def _glitch_master_video(ff, fp, allv, snap, boundaries_f, n, amt_base, say):
    """Port of the in-graph VHS glitch (nodes.py vhs_glitch transition) so the
    MASTER carries it too. The graph applies it to the concatenated tensor
    that SaveVideo writes; this worker rebuilds the master from the per-shot
    files, which are PRE-glitch, so without this pass the master is the one
    file that never glitches (found 2026-07-22). Same recipe: snow mix,
    horizontal tearing bands, dropout scanlines, triangular envelope over n
    frames centered on each boundary. Deterministic per boundary."""
    import numpy as np
    pr = subprocess.run([fp, "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height,r_frame_rate",
                         "-of", "csv=p=0", allv],
                        capture_output=True, text=True).stdout.strip().split(",")
    W, H = int(pr[0]), int(pr[1])
    num, den = pr[2].split("/")
    fps = float(num) / float(den or 1)
    out = os.path.join(snap, "allv_glitch.mp4")
    dec = subprocess.Popen([ff, "-v", "error", "-i", allv, "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([ff, "-y", "-v", "error", "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                            "-i", "-", "-c:v", "libx264", "-crf", "16",
                            # bf 0 + tune grain: same recipe as _save_shot_video.
                            # This pass re-encodes the WHOLE master; with default
                            # B-frames the b-pyramid bit-starves alternating
                            # frames and fine detail pulses every other frame
                            # (the 07-19 pumping artifact, reintroduced here by
                            # the 07-22 glitch port, caught 2026-07-23).
                            "-preset", "medium", "-tune", "grain", "-bf", "0",
                            "-pix_fmt", "yuv420p", "-an", out],
                           stdin=subprocess.PIPE)
    plan = {}
    for bi, b in enumerate(boundaries_f):
        start = max(0, b - n // 2)
        end = start + n
        span = max(1, end - start - 1)
        for k, fidx in enumerate(range(start, end)):
            plan[fidx] = (bi, k, span)
    fsz = W * H * 3
    idx = 0
    while True:
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz:
            break
        if idx in plan:
            bi, k, span = plan[idx]
            rng = np.random.RandomState(1009 * (bi + 1) + k)
            env = 1.0 - abs((k - span / 2.0) / (span / 2.0 or 1.0))
            amt = amt_base * (0.35 + 0.65 * max(0.0, env))
            f = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
            snow = rng.rand(H, W, 1).astype(np.float32)
            f = f * (1.0 - amt * 0.8) + snow * (amt * 0.8)
            for _ in range(int(1 + amt * 6)):
                y0 = rng.randint(0, max(1, H - 8))
                bh = rng.randint(2, max(3, H // 20))
                dx = rng.randint(-W // 6, W // 6 + 1)
                f[y0:y0 + bh] = np.roll(f[y0:y0 + bh], dx, axis=1)
            for _ in range(int(amt * 4)):
                y = rng.randint(0, H)
                f[y:y + 1] = rng.rand()
            buf = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
        enc.stdin.write(buf)
        idx += 1
    dec.stdout.close()
    enc.stdin.close()
    dec.wait()
    enc.wait()
    if enc.returncode != 0 or not os.path.isfile(out):
        say("WARNING: glitch video encode failed; master left clean")
        return None
    return out


def _glitch_master_audio(ff, fp, wav_in, snap, boundaries_t, n, fps, amt_base, say):
    """Tape-static bed at each boundary, ported from the graph: window is the
    WIDER of the glitch burst or 1.2s (JoyEcho room tone fades at shot edges;
    the static must span that dead seam), raised-cosine envelope."""
    import numpy as np
    pr = subprocess.run([fp, "-v", "error", "-select_streams", "a:0",
                         "-show_entries", "stream=sample_rate,channels",
                         "-of", "csv=p=0", wav_in],
                        capture_output=True, text=True).stdout.strip().split(",")
    sr, ch = int(pr[0]), int(pr[1])
    raw = subprocess.run([ff, "-v", "error", "-i", wav_in, "-f", "f32le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).copy()
    for bi, bt in enumerate(boundaries_t):
        rng = np.random.RandomState(2027 * (bi + 1))
        c = int(round(bt * sr))
        n_s = max(int(round(n / fps * sr)), int(round(1.2 * sr)))
        s0 = max(0, c - n_s // 2)
        s1 = min(len(x), s0 + n_s)
        if s1 <= s0:
            continue
        ln = s1 - s0
        t = np.linspace(0.0, 1.0, ln, dtype=np.float32)
        env = (0.5 - 0.5 * np.cos(t * 2.0 * np.pi)).astype(np.float32)[:, None]
        noise = rng.rand(ln, ch).astype(np.float32) * 2.0 - 1.0
        x[s0:s1] = np.clip(x[s0:s1] * (1.0 - 0.35 * amt_base * env)
                           + noise * (0.10 * amt_base) * env, -1.0, 1.0)
    out = os.path.join(snap, "alla_glitch.wav")
    p = subprocess.Popen([ff, "-y", "-v", "error", "-f", "f32le", "-ar", str(sr),
                          "-ac", str(ch), "-i", "-", "-c:a", "pcm_f32le", out],
                         stdin=subprocess.PIPE)
    p.communicate(x.tobytes())
    if p.returncode != 0 or not os.path.isfile(out):
        say("WARNING: glitch audio encode failed; audio left clean")
        return None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots-dir", required=True)
    ap.add_argument("--upscaled-dir", required=True)
    ap.add_argument("--name", default="JOYECHO")
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--quality", default="ULTRA")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    # Mirror the Generate node's transition defaults. The AutoFinish node does
    # not see those widgets, so these are worker-side settings; pass
    # --transition cut to keep the master clean.
    ap.add_argument("--transition", default="vhs_glitch")
    ap.add_argument("--glitch-frames", type=int, default=8)
    ap.add_argument("--glitch-intensity", type=float, default=0.7)
    # "cas" = deterministic ffmpeg bicubic + contrast-adaptive sharpen (zero
    # temporal churn - the 2026-07-23 upscale-tournament winner by eye).
    # "rtx" = the legacy RTXBatchVideoUpscale path (sharper synthesis, but
    # reshuffles fine detail per frame). Both are ignored when the run made
    # shot_hires masters - those are always used directly.
    ap.add_argument("--upscale-mode", default="cas", choices=["cas", "rtx"])
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(a.shots_dir, f"_autofinish_{stamp}.log")
    log = open(log_path, "a", encoding="utf-8")

    def say(msg):
        log.write(f"{datetime.now().strftime('%H:%M:%S')}  {msg}\n")
        log.flush()

    try:
        ff = _find(FFMPEG_CANDIDATES)
        fp = _find(FFPROBE_CANDIDATES)
        if not ff or not fp:
            say("FATAL: ffmpeg/ffprobe not found")
            return 1

        # Select ONLY the current run's shots. The shot counter restarts at
        # 000 every run and leftovers from longer previous runs linger at
        # higher indexes (found live 2026-07-19: shot_010..019 from the
        # prior day sat beside a fresh 10-shot run - a bare glob would have
        # upscaled all 20 and welded yesterday's shots onto the master).
        # Rule: walk contiguously from shot_000; every shot of the current
        # run is written AFTER shot_000, so stop at the first missing index
        # or the first file OLDER than shot_000 (60s grace).
        first = os.path.join(a.shots_dir, "shot_000.mp4")
        if not os.path.isfile(first):
            say(f"FATAL: {first} missing")
            return 1
        t_anchor = os.path.getmtime(first) - 60
        shots, i = [], 0
        while True:
            p = os.path.join(a.shots_dir, f"shot_{i:03d}.mp4")
            if not os.path.isfile(p) or os.path.getmtime(p) < t_anchor:
                break
            shots.append(p)
            i += 1
        stale = len(glob.glob(os.path.join(a.shots_dir, "shot_[0-9][0-9][0-9].mp4"))) - len(shots)
        say(f"current run = {len(shots)} shots (shot_000..shot_{len(shots)-1:03d}); "
            f"{stale} stale leftover(s) excluded")

        # Transition sidecar written by the Generate node: overrides the CLI
        # defaults so the master always matches the graph's actual widgets.
        # Same-run guard: ignore a sidecar older than this run's shot_000.
        sidecar = os.path.join(a.shots_dir, "_transition.json")
        if os.path.isfile(sidecar) and os.path.getmtime(sidecar) >= t_anchor:
            try:
                with open(sidecar, "r", encoding="utf-8") as fh:
                    tj = json.load(fh)
                a.transition = str(tj.get("transition", a.transition))
                a.glitch_frames = int(tj.get("frames", a.glitch_frames))
                a.glitch_intensity = float(tj.get("intensity", a.glitch_intensity))
                say(f"transition sidecar: {a.transition}, {a.glitch_frames} frames, "
                    f"intensity {a.glitch_intensity}")
            except Exception as e:  # noqa: BLE001
                say(f"WARNING: transition sidecar unreadable ({e}); using defaults")
        else:
            say(f"no same-run transition sidecar; defaults: {a.transition}, "
                f"{a.glitch_frames} frames, intensity {a.glitch_intensity}")
        if not shots:
            say(f"FATAL: no current-run shot masters in {a.shots_dir}")
            return 1

        # HIRES-AWARE ROUTE (2026-07-23): if this run's in-process hires
        # refine produced a shot_hires_XXX master for EVERY shot, those ARE
        # the finished shots - snapshot and assemble from them directly and
        # skip the RTX stage (RTX from the base shots would throw away the
        # refine's synthesized detail and upscale the wrong files). Same-run
        # guard: a hires file older than this run's shot_000 belongs to a
        # previous run and does not count. A partial set (refine crashed
        # mid-pass) falls back to the RTX path. The RTX path itself is
        # unchanged and remains the route for hires_factor=1.0 runs.
        hires = []
        for i in range(len(shots)):
            hp = os.path.join(a.shots_dir, f"shot_hires_{i:03d}.mp4")
            if os.path.isfile(hp) and os.path.getmtime(hp) >= t_anchor:
                hires.append(hp)
            else:
                break
        use_hires = len(hires) == len(shots)
        if hires and not use_hires:
            say(f"hires masters for only {len(hires)}/{len(shots)} shots "
                f"(incomplete refine) - falling back to RTX from base shots")
        if use_hires:
            say(f"hires masters detected for all {len(shots)} shots - "
                f"finishing from them; RTX upscale stage skipped")
            shots = hires

        # 1. snapshot (protects against a queued next render overwriting)
        snap = os.path.join(a.shots_dir, f"_finish_{stamp}")
        os.makedirs(snap, exist_ok=True)
        pairs = []
        for s in shots:
            base = os.path.splitext(os.path.basename(s))[0]
            wav = os.path.join(a.shots_dir, base + ".wav")
            sv = os.path.join(snap, os.path.basename(s))
            shutil.copy2(s, sv)
            wv = None
            if os.path.isfile(wav):
                wv = os.path.join(snap, os.path.basename(wav))
                shutil.copy2(wav, wv)
            pairs.append((sv, wv))
        say(f"snapshotted {len(pairs)} shots -> {snap}")

        # 2. upscale each snapshot sequentially via the API - unless the
        # snapshots already ARE the hires masters (see hires-aware route).
        if use_hires:
            ups = [sv for sv, _ in pairs]
        elif a.upscale_mode == "cas":
            # deterministic finish: bicubic to scale, contrast-adaptive
            # sharpen, clean bf0/tune-grain encode. Seconds per shot, CPU
            # only, cannot reshuffle detail (fixed kernels every frame).
            ups = []
            for i, (sv, _) in enumerate(pairs):
                out = os.path.join(snap, f"cas_{i:03d}.mp4")
                vf = (f"scale=trunc(iw*{a.scale}/2)*2:trunc(ih*{a.scale}/2)*2:"
                      f"flags=bicubic,cas=0.55")
                t0 = time.time()
                r = subprocess.run([ff, "-y", "-v", "error", "-i", sv, "-vf", vf,
                                    "-c:v", "libx264", "-crf", "16",
                                    "-preset", "medium", "-tune", "grain",
                                    "-bf", "0", "-an", out],
                                   capture_output=True, text=True)
                if r.returncode != 0 or not os.path.isfile(out):
                    say(f"FATAL: shot {i} cas upscale failed: {r.stderr[:300]}")
                    return 1
                ups.append(out)
                say(f"shot {i}: bicubic+cas x{a.scale} in {time.time()-t0:.0f}s")
        else:
            ups = []
            for i, (sv, _) in enumerate(pairs):
                before = set(glob.glob(os.path.join(a.upscaled_dir, "upscaled_*.mp4"))) \
                    if os.path.isdir(a.upscaled_dir) else set()
                g = {"1": {"class_type": "RTXBatchVideoUpscale",
                           "inputs": {"video_path": sv, "scale_factor": a.scale,
                                      "quality": a.quality,
                                      "batch_size": a.batch_size,
                                      "keep_audio": False}}}
                pid = api(a.comfy, "/prompt", {"prompt": g})["prompt_id"]
                say(f"shot {i}: submitted {pid}")
                t0 = time.time()
                while True:
                    if time.time() - t0 > 3600:
                        say(f"FATAL: shot {i} upscale timed out")
                        return 1
                    h = api(a.comfy, f"/history/{pid}")
                    if pid in h:
                        st = h[pid].get("status", {})
                        if st.get("completed") or st.get("status_str") == "success":
                            break
                        if st.get("status_str") == "error":
                            say(f"FATAL: shot {i} upscale error: {json.dumps(st)[:400]}")
                            return 1
                    time.sleep(5)
                new = sorted(set(glob.glob(os.path.join(a.upscaled_dir, "upscaled_*.mp4"))) - before,
                             key=os.path.getmtime)
                if not new:
                    say(f"FATAL: shot {i} produced no output")
                    return 1
                ups.append(new[-1])
                say(f"shot {i}: done in {time.time()-t0:.0f}s -> {os.path.basename(new[-1])}")

        # 3. concat video, concat audio CONTINUOUSLY, single mux.
        #
        # Each JoyEcho wav runs ~30ms SHORTER than its video (audio-latent vs
        # frame-count rounding): measured 2026-07-20, video 12.520s vs audio
        # 12.490s per 313-frame shot.
        #
        # Butt-splicing the raw wavs makes that shortfall COMPOUND: shot N's
        # audio starts N*30ms EARLY relative to its picture (120ms by shot 5,
        # 300ms by shot 10). Audio leading video is the more perceptible
        # direction, so sync degrades visibly through the piece.
        #
        # The fix is to pad each wav to its OWN video's ABSOLUTE duration
        # before concatenating. This cannot accumulate - shot i's audio then
        # begins at exactly sum(video durations before i), so every shot is
        # aligned by construction, and the ~30ms of silence lands at the END
        # of each shot (in the pause before the cut) where it is inaudible.
        #
        # NOTE: an earlier version of this file padded each shot by a FIXED
        # amount on top of its existing length, which DOES accumulate. That
        # is what the previous comment here was describing. Padding TO a
        # target duration and padding BY an amount are not the same thing.
        vlist, alist = [], []
        for i, ((_, wv), up) in enumerate(zip(pairs, ups)):
            vlist.append("file '" + up.replace(os.sep, "/") + "'")
            if not wv:
                continue
            vdur_i = subprocess.run(
                [fp, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", up],
                capture_output=True, text=True).stdout.strip()
            try:
                float(vdur_i)
            except (TypeError, ValueError):
                # Could not read this shot's duration - fall back to the raw
                # wav rather than guessing, and say so.
                say(f"shot {i}: WARNING no video duration, audio left unpadded")
                alist.append("file '" + wv.replace(os.sep, "/") + "'")
                continue
            wpad = os.path.join(snap, f"a_{i:03d}_pad.wav")
            # apad extends; -t trims. Together they force the exact duration
            # whether the wav is short (normal) or long.
            subprocess.run([ff, "-y", "-v", "error", "-i", wv, "-af", "apad",
                            "-t", vdur_i, "-c:a", "pcm_f32le", wpad], check=True)
            alist.append("file '" + wpad.replace(os.sep, "/") + "'")
        vtxt = os.path.join(snap, "v.txt")
        open(vtxt, "w").write("\n".join(vlist))
        allv = os.path.join(snap, "allv.mp4")
        subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", vtxt, "-c", "copy", "-an", allv], check=True)

        # 3b. VHS glitch at shot boundaries. The graph glitches the stream
        # SaveVideo writes; this master is rebuilt from pre-glitch shot files
        # and needs its own pass or it comes out clean (found 2026-07-22).
        boundaries_t = []
        if a.transition == "vhs_glitch" and len(ups) > 1:
            try:
                durs = [_probe_dur(fp, up) for up in ups]
                acc = 0.0
                for d in durs[:-1]:
                    acc += d
                    boundaries_t.append(acc)
                prf = subprocess.run([fp, "-v", "error", "-select_streams", "v:0",
                                      "-show_entries", "stream=r_frame_rate",
                                      "-of", "csv=p=0", allv],
                                     capture_output=True, text=True).stdout.strip()
                num, den = prf.split("/")
                fps_v = float(num) / float(den or 1)
                boundaries_f = [int(round(t * fps_v)) for t in boundaries_t]
                gv = _glitch_master_video(ff, fp, allv, snap, boundaries_f,
                                          max(1, a.glitch_frames),
                                          max(0.1, min(1.0, a.glitch_intensity)), say)
                if gv:
                    allv = gv
                    say(f"VHS glitch applied to master video at {len(boundaries_f)} "
                        f"boundaries ({a.glitch_frames} frames, intensity {a.glitch_intensity})")
            except Exception as e:  # noqa: BLE001
                say(f"WARNING: glitch pass skipped ({type(e).__name__}: {e}); master left clean")
                boundaries_t = []

        final = os.path.join(a.shots_dir, f"{a.name}_{stamp}_MASTER.mp4")
        if alist:
            atxt = os.path.join(snap, "a.txt")
            open(atxt, "w").write("\n".join(alist))
            alla = os.path.join(snap, "alla.wav")
            subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                            "-i", atxt, "-c", "copy", alla], check=True)
            # Pad the TAIL only, to total video length (apad+-t is a no-op
            # when audio already matches or exceeds it).
            vdur = subprocess.run([fp, "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=duration",
                                   "-of", "default=noprint_wrappers=1:nokey=1", allv],
                                  capture_output=True, text=True).stdout.strip()
            alla_p = os.path.join(snap, "alla_pad.wav")
            subprocess.run([ff, "-y", "-v", "error", "-i", alla, "-af", "apad",
                            "-t", vdur, "-c:a", "pcm_f32le", alla_p], check=True)
            if boundaries_t:
                try:
                    ga = _glitch_master_audio(ff, fp, alla_p, snap, boundaries_t,
                                              max(1, a.glitch_frames), 25.0,
                                              max(0.1, min(1.0, a.glitch_intensity)), say)
                    if ga:
                        alla_p = ga
                        say(f"tape-static bed applied to master audio at "
                            f"{len(boundaries_t)} boundaries")
                except Exception as e:  # noqa: BLE001
                    say(f"WARNING: audio glitch skipped ({type(e).__name__}: {e})")
            subprocess.run([ff, "-y", "-v", "error", "-i", allv, "-i", alla_p,
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", final],
                           check=True)
        else:
            shutil.copy2(allv, final)
        info = subprocess.run([fp, "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "stream=width,height,duration",
                               "-of", "csv=s=x:p=0", final],
                              capture_output=True, text=True).stdout.strip()
        say(f"DONE: {final} ({info})")
        return 0
    except Exception as e:  # noqa: BLE001
        say(f"FATAL: {type(e).__name__}: {e}")
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
