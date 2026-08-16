"""Streaming master assembly: peak host RAM = one shot, not the whole chain.

The RAM path holds every decoded shot until the end, then levels and joins -
up to ~21 GB for a six-shot upscaled chain even after the in-place fix, and
historically the step that killed 64 GB machines (issue #13). This module
writes each shot to a LOSSLESS temp file the moment it is decoded, keeps only
1-D per-frame statistics, and at the end re-streams the temps one at a time
through the exact _mn_normalize gain math into one ffmpeg encoder.

Parity contract (v1):
  * master_normalize off / luma / luma+contrast - EXACT: the gains are computed
    from the same per-frame luma/std series the RAM path uses; frames pass
    through torch with the same clamped gains.
  * color_level=scene, join_blend, join_fx - NOT streamed in v1: the caller
    falls back to the RAM path with a printed reason. Those passes read whole
    neighbouring shots; streaming them is a v2 problem.
  * Temps are lossless (libx264 -qp 0), so the master sees exactly one lossy
    encode, the same count as the RAM path's SaveVideo.

The master file is written by this module (x264 CRF 10, yuv420p, AAC audio) -
in streaming mode the sampler returns the PATH, and its frames output carries
a single black placeholder frame.
"""
import json
import os
import subprocess


def _ffmpeg():
    import shutil
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError(
            "low_ram_master needs ffmpeg on PATH (lossless shot temps + the "
            "final master encode). Install ffmpeg or turn low_ram_master off.")
    return p


class ShotStreamWriter:
    """Accumulates shots on disk instead of in RAM."""

    def __init__(self, out_dir, fps=24):
        # mkdtemp, not a pid-named dir: two runs in one comfy process share a
        # pid, and the second overwrote the first's temps (measured).
        import tempfile
        os.makedirs(out_dir, exist_ok=True)
        out_dir = tempfile.mkdtemp(prefix="stage_", dir=out_dir)
        self.dir = out_dir
        self.fps = float(fps)
        self.shots = []          # [{path, frames, w, h}]
        self.luma = []           # 1-D tensors, per-frame mean luma
        self.std = []            # 1-D tensors, per-frame std (contrast)
        # One-shot deferral: the newest shot stays in RAM until the NEXT one
        # arrives, because join_blend (seamless_tail/latent_handoff) mutates
        # the previous shot's tail frames in place. Peak RAM = ~two shots.
        self.pending = None

    def add(self, imgs):
        """Deferred staging: stash this shot, stage the previous one."""
        prev, self.pending = self.pending, imgs
        if prev is not None:
            self._stage(prev)

    def _stage(self, imgs):
        """imgs: [T,H,W,C] float 0..1 on CPU. Writes lossless, keeps stats."""
        import torch
        t, h, w, _ = imgs.shape
        path = os.path.join(self.dir, "shot_%03d.mkv" % len(self.shots))
        cmd = [_ffmpeg(), "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "%dx%d" % (w, h), "-r", "%.6f" % self.fps,
               "-i", "-",
               "-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
               "-pix_fmt", "yuv444p", path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        # chunked so the uint8 staging never doubles the shot
        for i in range(0, t, 32):
            chunk = (imgs[i:i + 32].clamp(0, 1) * 255).round().to(torch.uint8)
            proc.stdin.write(chunk.numpy().tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("lossless temp write failed for %s" % path)
        self.luma.append(imgs.mean(dim=(1, 2, 3)).float())
        self.std.append(imgs.std(dim=(1, 2, 3)).float())
        self.shots.append({"path": path, "frames": int(t), "w": w, "h": h})
        print("[H3StreamMaster] shot %d staged lossless (%d frames, %.1f MB) - "
              "host RAM holds statistics only."
              % (len(self.shots), t, os.path.getsize(path) / 2 ** 20),
              flush=True)

    # ------------------------------------------------------------------
    def _gains(self, mode, med=9):
        """The same math as _mn_normalize, computed from stored stats.

        Returns per-frame (luma_gain, contrast_gain) 1-D tensors, or None.
        """
        import torch
        import torch.nn.functional as F
        if mode == "off" or not self.shots:
            return None
        luma = torch.cat(self.luma)
        n = luma.shape[0]
        if n < 8:
            return None
        k = max(3, min(int(med) | 1, (n // 2) * 2 - 1))
        pad = k // 2

        def _med(v):
            x = F.pad(v[None, None], (pad, pad), mode="replicate")
            return x.unfold(-1, k, 1).median(dim=-1).values[0, 0]

        # EXACT mirror of _mn_normalize (parity audit 2026-08-16: the first
        # version diverged four ways - target, clamps, contrast anchor and the
        # application formula - and cost ~6 dB of parity).
        med_l = _med(luma)
        target = luma.median()                    # raw median, not med-of-med
        lg = (target / med_l.clamp_min(1e-4)).clamp(0.70, 1.43)
        cg = None
        if "contrast" in mode:
            sd = torch.cat(self.std)
            med_s = _med(sd)
            # anchor to SHOT 1: contrast only ratchets up, so shot 1 is the
            # clean one - a timeline median would pull shot 1 UP instead.
            s_target = sd[:self.shots[0]["frames"]].median()
            cg = (s_target / med_s.clamp_min(1e-4)).clamp(0.70, 1.43)
        return lg, cg, luma

    def finalize(self, master_path, mode, waveform, sr,
                 prompt=None, extra_pnginfo=None):
        """Re-stream temps through the gains into one encoder."""
        import torch
        if self.pending is not None:
            self._stage(self.pending)
            self.pending = None
        gains = self._gains(mode)
        if gains is not None:
            lg_all, cg_all, luma_all = gains
            print("[H3StreamMaster] normalize (%s): global target from %d "
                  "frames of stats." % (mode, sum(s["frames"] for s in self.shots)),
                  flush=True)
        w, h = self.shots[0]["w"], self.shots[0]["h"]

        wav_path = os.path.join(self.dir, "master_audio.wav")
        # stdlib wave, NOT torchaudio: torchaudio 2.13's save() delegates to
        # torchcodec, which portable installs do not carry (measured: both
        # first streaming runs died exactly here).
        import wave
        wf = waveform if waveform.ndim == 2 else waveform[0]
        pcm = (wf.cpu().float().clamp(-1, 1).t() * 32767.0).round().to(
            torch.int16).numpy().tobytes()
        with wave.open(wav_path, "wb") as wv:
            wv.setnchannels(int(wf.shape[0]))
            wv.setsampwidth(2)
            wv.setframerate(int(sr))
            wv.writeframes(pcm)

        cmd = [_ffmpeg(), "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "%dx%d" % (w, h), "-r", "%.6f" % self.fps, "-i", "-",
               "-i", wav_path,
               "-c:v", "libx264", "-crf", "10", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k"]
        # same container tags SaveVideo writes, so the master itself can be
        # dropped onto a ComfyUI canvas to recover the graph
        import json as _json
        _tagged = False
        if prompt is not None:
            cmd += ["-metadata", "prompt=" + _json.dumps(prompt)]
            _tagged = True
        if extra_pnginfo is not None:
            for _k, _v in extra_pnginfo.items():
                cmd += ["-metadata", "%s=%s" % (_k, _json.dumps(_v))]
            _tagged = True
        if _tagged:
            # mp4 silently drops custom tag keys without this movflag
            cmd += ["-movflags", "use_metadata_tags"]
        cmd += ["-shortest", master_path]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        off = 0
        for s in self.shots:
            dec = subprocess.Popen(
                [_ffmpeg(), "-v", "error", "-i", s["path"],
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE)
            frame_bytes = s["w"] * s["h"] * 3
            fi = 0
            while True:
                buf = dec.stdout.read(frame_bytes * 16)
                if not buf:
                    break
                nfr = len(buf) // frame_bytes
                x = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
                x = x.reshape(nfr, s["h"], s["w"], 3).float() / 255.0
                if gains is not None:
                    s_ = slice(off + fi, off + fi + nfr)
                    g = lg_all[s_].view(-1, 1, 1, 1)
                    if cg_all is not None:
                        # the ORIGINAL formula: contrast about the frame's own
                        # pre-normalize mean, luma gain applied to the mean
                        # only - never to the deviations.
                        c = cg_all[s_].view(-1, 1, 1, 1)
                        m = luma_all[s_].view(-1, 1, 1, 1)
                        x = (x - m) * c + m * g
                    else:
                        x = x * g
                enc.stdin.write((x.clamp(0, 1) * 255).round()
                                .to(torch.uint8).numpy().tobytes())
                fi += nfr
            dec.wait()
            off += s["frames"]
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("master encode failed - shot temps kept in %s"
                               % self.dir)
        for s in self.shots:
            try:
                os.remove(s["path"])
            except OSError:
                pass
        try:
            os.remove(wav_path)
        except OSError:
            pass
        # the stage dir is per-run and now empty; its tmp_<pid> parent is
        # shared by runs in this process, so only remove it once it drains
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(self.dir))
        except OSError:
            pass
        print("[H3StreamMaster] master written: %s (%.1f MB)"
              % (master_path, os.path.getsize(master_path) / 2 ** 20),
              flush=True)
        return master_path
