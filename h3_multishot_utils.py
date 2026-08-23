# -*- coding: utf-8 -*-
"""H3 multishot utilities - JoyEcho-style single-script prompting for the
MiniMax H3 chained workflow. One node, no dependencies.

Accepts the same script formats the JoyEcho stack uses:
  - JSON: {"prompts": ["shot 1 ...", "shot 2 ...", "shot 3 ..."]}
  - plain text with --- separators between shots
Feeds up to 4 shot prompts as separate STRING outputs. Missing shots fall
back to the previous shot's prompt so a 2-shot script still runs a 3-shot
graph without erroring.
"""
import json
import re



def _repair_json(text):
    """Parse JSON, auto-closing unterminated brackets/quotes.

    Long multi-prompt scripts get truncated or lose their final brace all the
    time (a 4,500-char script with the closing '}' missing is not a typo the
    author can see). Returns (data, note): data is None on real failure and
    note carries the error; note is a description when a repair was applied,
    or "" when the text parsed clean.
    """
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as e:
        first_err = str(e)   # bind now; Python clears the except-name on exit

    # walk the text tracking string state, then close what is still open
    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or
                          (ch == "]" and stack[-1] == "[")):
                stack.pop()

    candidate = text.rstrip()
    fixes = []
    if in_str:
        candidate += '"'
        fixes.append("closed an open string")
    if candidate.endswith(","):
        candidate = candidate[:-1]
        fixes.append("dropped a trailing comma")
    # trailing comma before a closer, e.g.  ["a","b",]  or  {"k":1,}
    cleaned = re.sub(r",(\s*[\]}])", r"\1", candidate)
    if cleaned != candidate:
        candidate = cleaned
        fixes.append("removed comma(s) before a closing bracket")
    for opener in reversed(stack):
        candidate += "}" if opener == "{" else "]"
    if stack:
        fixes.append("added " + "".join("}" if o == "{" else "]"
                                        for o in reversed(stack)))
    if not fixes:
        return None, first_err
    try:
        return json.loads(candidate), ", ".join(fixes)
    except json.JSONDecodeError as e:
        return None, str(e)



_AT_NBANDS = 8


def _up_model_list():
    try:
        import folder_paths
        return folder_paths.get_filename_list("upscale_models")
    except Exception:
        return []


def _load_up_model(name):
    """Load an upscale model by NAME, through ComfyUI's own loader node.

    Do not reimplement this. The first version here copied the loader's body
    from an older ComfyUI - state dict, spandrel, eval() - and missed that the
    current one also attaches a CoreModelPatcher, which ImageUpscaleWithModel
    then reads as upscale_model.patcher.load_device. It crashed AFTER shot 1
    had rendered. Calling the real node means this cannot drift out of sync
    with whatever ComfyUI does next.
    """
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader
    out = UpscaleModelLoader().load_model(name)
    # the V3 node API returns a NodeOutput; older returns a plain tuple
    for attr in ("result", "results"):
        if hasattr(out, attr):
            out = getattr(out, attr)
            break
    while isinstance(out, (tuple, list)):
        out = out[0]
    if not hasattr(out, "patcher"):
        raise RuntimeError(
            "upscale_model_name=%r loaded, but the result has no .patcher - "
            "ComfyUI's upscale loader has changed shape again. Wire a Load "
            "Upscale Model node instead, or set upscale_model_name to "
            "(none)." % name)
    return out


def _mn_normalize(parts, mode, med=9):
    """Level luma across the WHOLE assembled chain, to ONE global target.

    Deflicker over the finished timeline. Two design points, both learned the
    hard way:

    * ONE GLOBAL target, not a rolling or smoothed-local one. The per-shot
      colour mode drives each shot to a rolling house and leaves a hard step
      at every join (measured 29% warmth). A smoothed-local target is no
      better for chained video: a shot boundary is a STEP, a wide smoothing
      window turns it into a ramp, and the gain then ramps with it instead of
      cancelling it (unit-tested: a 15% step came out 13.5%). Driving every
      frame to a single number removes steps by construction, because every
      frame lands on the same number.
    * LUMA ONLY. Texture drift cannot be fixed here. Blur is the only lever
      available after the fact, and blur removes real detail along with the
      model's invented detail - it satisfies a metric by destroying
      information. A ratcheting chain wants a shorter chain, an anchor-only
      conditioning diet, or a model-side fix; it does not want a filter.

    A short median tracks the actual level rather than reacting to single-frame
    noise. Gain is clamped so nothing is invented and black stays black.
    """
    import torch
    if mode == "off" or not parts:
        return parts, ""
    n = sum(int(p.shape[0]) for p in parts)
    if n < 8:
        return parts, ""
    # Per-frame statistics only - never the whole timeline as one tensor.
    # Concatenating it costs frames x H x W x 3 x 4 bytes, which is 31 GB for a
    # 12-shot 1088x1920 chain, and it used to be built three times over. Every
    # number below this point is one-dimensional.
    luma = torch.cat([p.mean(dim=(1, 2, 3)) for p in parts])
    k = max(3, min(int(med) | 1, (n // 2) * 2 - 1)); pad = k // 2

    def _med(v):
        return torch.nn.functional.pad(
            v[None, None], (pad, pad), mode="replicate"
        )[0, 0].unfold(0, k, 1).median(-1).values

    med_l = _med(luma)
    target = luma.median()
    gain = (target / med_l.clamp_min(1e-4)).clamp(0.70, 1.43)

    if mode == "luma+contrast":
        # Second drift, measured on master_00014_: the mean holds flat while
        # the SPREAD grows every hop (p25 6->2, p95 85->96 over three shots).
        # Matching the mean re-centres that and hands the next shot a
        # higher-contrast start, so the luma fix was masking it.
        #
        # An affine remap about each frame's own mean rescales amplitude and
        # leaves every edge where it was - not the blur the note below rules
        # out. Spatial accretion is still not fixable here.
        sd = torch.cat([p.std(dim=(1, 2, 3)) for p in parts])
        med_s = _med(sd)
        # Anchor to SHOT 1, not the timeline median. Contrast here only ever
        # ratchets up, so shot 1 is the one frame-set with nothing accreted
        # onto it - the median target pulls shot 1 UP to meet the drift
        # (measured on master_00014_: +11.7% texture on shot 1 for no benefit,
        # 1.069 vs 1.064 per hop). Anchoring to shot 1 leaves it untouched
        # (+0.1%) and only ever pulls later shots down.
        s_target = sd[:parts[0].shape[0]].median()
        cgain = (s_target / med_s.clamp_min(1e-4)).clamp(0.70, 1.43)
    else:
        cgain = None
    # Apply per part, releasing each input as it is consumed. The caller does
    # `parts, msg = _mn_normalize(parts, ...)` and rebinds immediately, so
    # dropping the reference here lets each input shot be freed while the rest
    # of the chain is still being processed. Peak becomes one finished timeline
    # plus one shot, instead of two whole timelines.
    res, i = [], 0
    for idx in range(len(parts)):
        p = parts[idx]
        j = i + int(p.shape[0])
        g = gain[i:j, None, None, None]
        # Frames may be stored fp16 to halve the host timeline. fp16's step near
        # 1.0 is ~9.8e-4 against an 8-bit output step of 3.9e-3 - enough, but
        # only 4x. Do the affine in fp32 and store back in the input dtype, so
        # the memory saving costs no precision.
        _dt = p.dtype
        _pf = p.float() if _dt != torch.float32 else p
        if cgain is not None:
            m = luma[i:j, None, None, None]
            _out = ((_pf - m) * cgain[i:j, None, None, None] + m * g).clamp(0, 1)
        else:
            _out = (_pf * g).clamp(0, 1)
        res.append(_out.to(_dt) if _dt != torch.float32 else _out)
        if _pf is not p:
            del _pf
        del _out
        parts[idx] = None
        del p
        i = j
    after = torch.cat([q.mean(dim=(1, 2, 3)) for q in res])
    msg = ("luma %.3f-%.3f -> %.3f-%.3f (target %.3f, gain %.3f-%.3f)"
           % (float(luma.min()), float(luma.max()), float(after.min()),
              float(after.max()), float(target), float(gain.min()), float(gain.max())))
    if mode == "luma+contrast":
        a_sd = torch.cat([q.std(dim=(1, 2, 3)) for q in res])
        sd0 = sd
        msg += ("; contrast %.4f-%.4f -> %.4f-%.4f"
                % (float(sd0.min()), float(sd0.max()),
                   float(a_sd.min()), float(a_sd.max())))
    return res, msg


def _at_ltas(wav, sr=32000, nfft=2048):
    """Long-term average spectrum of a [C, L] / [1, C, L] waveform in
    _AT_NBANDS log-spaced bands (100 Hz .. ~12 kHz). Long-term, so per-shot
    speech content averages out and only the spectral TILT is measured."""
    import torch
    x = wav.reshape(-1, wav.shape[-1]).float().mean(0)
    n = (x.shape[-1] // nfft) * nfft
    if n < nfft:
        return None
    S = torch.stft(x[:n], nfft, hop_length=nfft // 2,
                   window=torch.hann_window(nfft), return_complex=True)
    P = (S.abs() ** 2).mean(-1)
    f = torch.linspace(0, sr / 2, P.shape[0])
    edges = torch.logspace(2, 4.08, _AT_NBANDS + 1)
    out = []
    for i in range(_AT_NBANDS):
        m = (f >= edges[i]) & (f < edges[i + 1])
        out.append(P[m].mean() if m.any() else P.new_tensor(0.0))
    return torch.stack(out).clamp_min(1e-12)


def _at_flatten(wav, house, sr=32000, nfft=2048, max_db=9.0):
    """EQ-match a shot's long-term spectral envelope to the house envelope.

    The audio twin of _cg_flatten, aimed the other way: chained audio drifts
    DULLER per hop where chained video drifts sharper. Measured 2026-08-11 on
    8-shot chains: 4-10 kHz energy fell 84-92% on pure-recency conditioning
    (bank_pinned=0) and 8-13% even with the pinned slot. Constant per-shot
    gains applied via STFT - a linear filter, so no pumping. Clamped to
    +/-max_db, half-strength in the top band so it cannot manufacture hiss
    where the model genuinely rendered none.
    """
    import torch
    cur = _at_ltas(wav, sr, nfft)
    if cur is None or house is None:
        return wav, 0.0
    gain_db = (10.0 * torch.log10(house / cur)).clamp(-max_db, max_db)
    gain_db[-1] = gain_db[-1] * 0.5
    if float(gain_db.abs().max()) < 0.75:
        return wav, 0.0
    f = torch.linspace(0, sr / 2, nfft // 2 + 1)
    edges = torch.logspace(2, 4.08, _AT_NBANDS + 1)
    centres = (edges[:-1] * edges[1:]).sqrt()
    logf = torch.log10(f.clamp_min(1.0))
    logc = torch.log10(centres)
    idx = torch.bucketize(logf, logc).clamp(1, _AT_NBANDS - 1)
    x0, x1 = logc[idx - 1], logc[idx]
    w = ((logf - x0) / (x1 - x0)).clamp(0, 1)
    curve = 10.0 ** ((gain_db[idx - 1] * (1 - w) + gain_db[idx] * w) / 20.0)
    shape = wav.shape
    x = wav.reshape(-1, shape[-1]).float()
    win = torch.hann_window(nfft)
    S = torch.stft(x, nfft, hop_length=nfft // 4, window=win,
                   return_complex=True)
    S = S * curve.unsqueeze(0).unsqueeze(-1)
    y = torch.istft(S, nfft, hop_length=nfft // 4, window=win,
                    length=shape[-1])
    return y.reshape(shape).to(wav.dtype), float(gain_db.abs().max())


def _mmh3_encode_ref_audio(audio_vae, audio):
    """ComfyUI's ref-audio encoder, wherever this ComfyUI keeps it.

    Up to 0.33 it was a staticmethod on MiniMaxH3ReferenceToVideo; master
    (2026-08) moved it to a module-level _encode_ref_audio. Resolve at call
    time so neither layout raises AttributeError (GitHub issue #15)."""
    from comfy_extras import nodes_minimax_h3 as mmh3
    fn = getattr(mmh3, "_encode_ref_audio", None)
    if fn is None:
        fn = getattr(getattr(mmh3, "MiniMaxH3ReferenceToVideo", None),
                     "_encode_ref_audio", None)
    if fn is None:
        raise RuntimeError(
            "This ComfyUI's nodes_minimax_h3 has no _encode_ref_audio "
            "(neither module-level nor on MiniMaxH3ReferenceToVideo). "
            "Update ComfyUI-H3-Multishot, or report the ComfyUI version.")
    return fn(audio_vae, audio)


def _wav_for_vae(audio_vae, audio, what):
    """AUDIO dict -> [1, C, L] waveform at the VAE's own sample rate, stereo.

    Mirrors the native node's _encode_ref_audio (nodes_minimax_h3.py): resample
    to the VAE's rate before encoding. The spine path used to skip this, so a
    44.1/48 kHz voice file - i.e. nearly every real-world file - was encoded as
    if it were 32 kHz. The latent was garbage, and because the spine LOCKS the
    audio stream to that latent at every sampling step, the render came out as
    static (user-reported on ref2va; the same file worked through the native
    node, which resamples).

    Mono is upmixed to stereo by duplication - the encoder wants two channels,
    and refusing a mono file helps nobody.
    """
    w = audio["waveform"]
    sr = int(audio["sample_rate"])
    w3 = w if w.ndim == 3 else w.unsqueeze(0)
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        import torchaudio
        w3 = torchaudio.functional.resample(w3, sr, vae_sr)
        print(f"[H3] {what}: resampled {sr} -> {vae_sr} Hz", flush=True)
    if w3.shape[1] == 1:
        w3 = w3.repeat(1, 2, 1)
        print(f"[H3] {what}: mono upmixed to stereo", flush=True)
    return w3[:1], vae_sr


def _write_shot_mp4(imgs, wav, sr, prefix, label, tag):
    """Write one decoded shot to disk immediately, and never let that kill
    the render.

    A long chain represents hours of GPU time that only becomes a file at
    the very end, when the master is muxed. Anything that fails after the
    last shot - a mux OOM, a full disk, a cancelled tab - has historically
    destroyed every shot at once (issue #13). Each shot written as it
    decodes turns that from lost work into a joining job.

    Returns the path written, or None (already reported) on failure.
    """
    try:
        import os
        from fractions import Fraction
        import folder_paths
        from comfy_api.latest import InputImpl, Types
        w = wav if wav.ndim == 3 else wav.unsqueeze(0)
        folder, fname, counter, _sub, _pfx = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(),
            imgs.shape[2], imgs.shape[1])
        path = os.path.join(folder, f"{fname}_{counter:05}_.mp4")
        InputImpl.VideoFromComponents(Types.VideoComponents(
            images=imgs.detach().cpu(),
            audio={"waveform": w.detach().cpu(), "sample_rate": sr},
            frame_rate=Fraction(24))).save_to(path)
        print(f"[{tag}] {label} -> {path}", flush=True)
        return path
    except Exception as e:
        print(f"[{tag}] {label} FAILED to save (render continues): {e}",
              flush=True)
        return None



def _up_model_factor(model, default=4.0):
    """The fixed enlargement factor of a loaded upscale model.

    Needed to predict output size before anything is upscaled. ESRGAN-family
    models expose it as `scale`; fall back to 4x, which is what almost every
    model in circulation is, rather than refusing to predict.
    """
    for attr in ("scale", "scale_factor", "upscale_factor"):
        v = getattr(model, attr, None)
        if v is None:
            v = getattr(getattr(model, "model", None), attr, None)
        try:
            if v and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            pass
    return default

def _upscale_frames(imgs, scale, model, tag):
    """Enlarge decoded frames, AFTER sampling. Pixels, never latents.

    The old two-pass path interpolated the raw latent between passes; H3's
    latent is not spatially smooth, so that landed off-manifold and produced
    colour noise no matter how the schedule was split. Anything done here is
    downstream of the VAE, so it cannot leave the manifold - the worst case is
    a soft picture, not a broken one.

    Per shot, so peak memory is one shot's frames rather than the whole chain.
    """
    if model is None and (not scale or abs(scale - 1.0) < 1e-6):
        return imgs
    import comfy.utils
    h, w = int(imgs.shape[1]), int(imgs.shape[2])
    if model is not None:
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        import comfy.model_management as _mm
        import os as _os
        import torch as _t
        # Chunked. One call with every frame allocates the whole upscaled batch
        # in fp32 on the CPU on top of the input - 11 GB for 243 frames at
        # 1472x2560, which is where a 12-shot 736x1280 run died after two hours.
        # Writing into a preallocated output holds (output + one chunk) instead.
        _n = int(imgs.shape[0])
        _ch = max(1, int(_os.environ.get("H3_UPSCALE_CHUNK", "16")))
        if _n > _ch:
            _out = None
            for _s in range(0, _n, _ch):
                _p = ImageUpscaleWithModel().upscale(model, imgs[_s:_s + _ch])[0]
                if _out is None:
                    _out = _t.empty((_n,) + tuple(_p.shape[1:]), dtype=_p.dtype)
                _out[_s:_s + int(_p.shape[0])] = _p
                del _p
            imgs = _out
        else:
            imgs = ImageUpscaleWithModel().upscale(model, imgs)[0]
        # Free it AT ONCE. Left resident it survives into the next shot, and
        # the DiT can then only load partially - measured 18.5 s/it on shot 1
        # (full load) against 349 s/it on shot 2 (431 MB offloaded), a 19x
        # collapse from ~65 MB of upscaler weights holding the door open.
        try:
            _mm.free_memory(_mm.get_total_memory(_mm.get_torch_device()),
                            _mm.get_torch_device(), [getattr(model, "patcher", None)])
            if hasattr(model, "patcher"):
                model.patcher.model.to(_mm.unet_offload_device())
            _mm.soft_empty_cache()
        except Exception as _e:
            print("[%s] upscale: could not free the upscaler (%s) - the next "
                  "shot may load the DiT partially and run far slower"
                  % (tag, _e), flush=True)
        print("[%s] upscale: model %dx%d -> %dx%d" %
              (tag, w, h, int(imgs.shape[2]), int(imgs.shape[1])), flush=True)
        if scale and abs(scale - 1.0) > 1e-6:
            # the model has a fixed factor; land on the size actually asked for
            th, tw = int(round(h * scale)), int(round(w * scale))
            x = imgs.movedim(-1, 1)
            x = comfy.utils.common_upscale(x, tw, th, "lanczos", "disabled")
            imgs = x.movedim(1, -1)
            print("[%s] upscale: resized to %dx%d" % (tag, tw, th), flush=True)
        return imgs
    th, tw = int(round(h * scale)), int(round(w * scale))
    x = imgs.movedim(-1, 1)
    x = comfy.utils.common_upscale(x, tw, th, "lanczos", "disabled")
    print("[%s] upscale: lanczos %dx%d -> %dx%d" % (tag, w, h, tw, th), flush=True)
    return x.movedim(1, -1)


def _smart_head_trim(wav, sr, trim, search_s=0.75):
    """Remove `trim` samples from a chained shot's audio HEAD, cutting at
    the QUIETEST spot in the first `search_s` seconds instead of blindly
    at sample 0.

    The blind head cut clips the attack of any word the model placed at
    the very start of the shot (the 'blip' - ear-verified on a render
    2026-08-10: a syllable burst 0.4s after the weld with its front
    shaved). Cutting the same number of samples out of the quietest
    window preserves every onset; the audio before the cut plays at most
    trim/sr (~42ms) late against the video, in a region that is quiet by
    construction, and sync is exact after the cut. Quietest-window-at-0
    reproduces the old behaviour bit for bit."""
    import torch
    n = wav.shape[-1]
    if n <= trim:
        return wav[..., :0]
    limit = min(n - trim, int(sr * search_s))
    if limit <= 1:
        return wav[..., trim:]
    mono = wav.float().abs()
    while mono.ndim > 1:
        mono = mono.mean(0)
    sq = mono[:limit + trim] ** 2
    cs = torch.cumsum(torch.cat([torch.zeros(1, device=sq.device), sq]), 0)
    win_energy = cs[trim:limit + trim] - cs[:limit]
    i = int(win_energy.argmin())
    if i > 0:
        print(f"[H3Multishot] smart weld: seam cut moved {i / sr * 1000:.0f}ms "
              f"into the head (quietest gap), word onsets preserved",
              flush=True)
    return torch.cat([wav[..., :i], wav[..., i + trim:]], dim=-1)


def _xfade_audio(parts, sr, ms=40):
    """Concatenate shot audio with a short equal-power crossfade at each seam.

    Each shot is sampled independently, so its waveform starts and ends at a
    hard boundary. Butt-joining them puts a step discontinuity in the signal at
    every seam, which reads as a click and as "spliced clips" to a listener.
    A ~40ms equal-power fade removes the step without audibly shortening
    anything.
    """
    import torch
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    n = max(1, int(sr * ms / 1000.0))
    out = parts[0]
    for nxt in parts[1:]:
        k = min(n, out.shape[-1], nxt.shape[-1])
        if k < 8:                      # too short to fade; butt-join
            out = torch.cat([out, nxt], dim=-1)
            continue
        t = torch.linspace(0, 1, k, dtype=out.dtype, device=out.device)
        fade_out = torch.cos(t * 3.14159265 / 2)   # equal power
        fade_in = torch.sin(t * 3.14159265 / 2)
        head, tail = out[..., :-k], out[..., -k:]
        seam = tail * fade_out + nxt[..., :k] * fade_in
        out = torch.cat([head, seam, nxt[..., k:]], dim=-1)
    return out


def _parse_script(text):
    """JoyEcho script -> list of shot prompts. JSON {"prompts": [...]} or
    plain text with --- separators. Malformed JSON fails LOUD."""
    text = (text or "").strip()
    shots = []
    if text.startswith("{") or text.startswith("["):
        data, repaired = _repair_json(text)
        if data is None:
            raise ValueError(
                f"H3 script looks like JSON but does not parse ({repaired}). "
                f"Auto-repair of unclosed brackets/quotes was attempted and "
                f"failed. Common cause: a doubled {{ on the first lines, or a "
                f"missing comma between prompts. Fix the script or use plain "
                f"prompts separated by --- lines.")
        if repaired:
            print(f"[H3Multishot] script JSON was incomplete; auto-repaired "
                  f"({repaired}). Consider fixing the source.", flush=True)
        if isinstance(data, dict):
            shots = [str(p) for p in data.get("prompts", [])]
        elif isinstance(data, list):
            shots = [str(p) for p in data]
    if not shots:
        shots = [b.strip().replace('\\"', '"')
                 for b in re.split(r"(?m)^---\s*$", text) if b.strip()]
    if not shots:
        shots = [text]
    return shots


class H3ScriptSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"script": ("STRING", {
            "multiline": True, "dynamicPrompts": False,
            "default": "Shot 1 prompt goes here.\n---\n"
                       "Shot 2 prompt goes here.\n---\n"
                       "Shot 3 prompt goes here.",
            "tooltip": "One prompt per shot, separated by --- on its own "
                       "line. (JSON {\"prompts\": [...]} also accepted, for "
                       "generated scripts.)"}),
            "shot_count": ("INT", {
                "default": 0, "min": 0, "max": 3,
                "tooltip": "This workflow ALWAYS renders 3 segments and "
                           "joins them (~30s master). 0 = count from the "
                           "script. 3 = three scenes. 2 = the third segment "
                           "continues scene 2. 1 = one scene sustained for "
                           "the full 30s. Scripts with >3 prompts: extras "
                           "are dropped (see console)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("shot_1", "shot_2", "shot_3", "shot_4", "shot_count")
    FUNCTION = "split"
    CATEGORY = "conditioning/minimax"

    def split(self, script, shot_count=0):
        shots = _parse_script(script)
        if shot_count and shot_count > 0:
            if len(shots) > shot_count:
                print(f"[H3ScriptSplit] shot_count={shot_count}: dropping "
                      f"{len(shots) - shot_count} extra script shot(s).",
                      flush=True)
                shots = shots[:shot_count]
            while len(shots) < shot_count:
                shots.append(shots[-1])
        n = len(shots)
        if n < 3:
            print(f"[H3ScriptSplit] script has {n} shot(s); a 3-shot graph "
                  f"will render the last prompt {3 - n} extra time(s) as a "
                  f"continuation.", flush=True)
        elif n > 3:
            print(f"[H3ScriptSplit] script has {n} shots; a 3-shot graph "
                  f"DROPS shot(s) 4+. Trim the script or wait for the "
                  f"dynamic-count workflow.", flush=True)
        while len(shots) < 4:
            shots.append(shots[-1])
        return (shots[0], shots[1], shots[2], shots[3], n)


# ---------------------------------------------------------------------------
# AUTO ACTIVATION RESERVE
#
# VRAM has two tenants with opposite tolerance for being remote. WEIGHTS
# stream well: sequential, known order, prefetched behind ~50s of compute per
# step, so 20GB offloaded costs well under a second. The ALLOCATOR POOL
# (activations) does not stream: it is random-access and re-touched all step,
# and when it does not fit, the driver evicts blind mid-step. Measured on a
# 3090: starving the pool by ~3GB = 533 s/it vs 99 s/it. Measured on a 5090:
# 168W at "99% utilisation" - a card waiting, not computing.
#
# So the correct split is reserve >= pool, and stream whatever weights do not
# fit - overshooting is nearly free, undershooting is 5-10x. The pool scales
# with the render shape, which is why any hand-set number (a GB figure, a
# memory_usage_factor) is correct for exactly one resolution and a trap at
# every other: LOWER the resolution with a fixed factor and the reserve
# shrinks below the pool - the render gets SLOWER, the opposite of what any
# person expects.
#
# This engine removes the knob:
#   - memory_required(input_shape) is overridden with a function of the
#     ACTUAL shape comfy passes at load time - never a constant.
#   - Unmeasured shapes reserve 60% of currently-free VRAM: generous enough
#     to be cliff-proof at any resolution, and only slightly slower than
#     optimal (a few more GB of weights stream).
#   - Our samplers measure the true allocator peak of every run and cache it
#     per (GPU, model, shape-cells) in the user dir. From the second run at
#     a shape, the reserve is measured * 1.25 - per machine, no telemetry to
#     read, no number to know.
# ---------------------------------------------------------------------------

_AUTO_FLOOR = 8 * 1024**3          # never reserve less: workspaces + margin
# First-run fraction of free VRAM. Deliberately HIGH: over-reserving merely
# streams more weights (<1s/step behind 50-100s of compute), while
# under-reserving is the 5-10x cliff. 0.60 was calibrated on a 32GB card and
# proved WRONG on a 24GB one: 60% of the 3090's 21.9GB free = 13.2GB against
# a ~17.5GB pool -> max_reserved 25.06GB on a 24GB card -> 492 s/it. At 0.88
# the same card reserves 19.3GB and streams the difference. Measurement then
# tightens DOWN from the safe side.
_AUTO_FRACTION = 0.88              # unmeasured shapes: fraction of free VRAM
_AUTO_WEIGHT_NUCLEUS = 2 * 1024**3  # always leave a little room for weights
_AUTO_MARGIN = 1.25                # measured pool -> reserve headroom
_AUTO_KEEPOUT = 1024 * 1024**2     # left free beyond weights+pool. 384 MB was
                                   # too small: ComfyUI reserves its own ~117 MB
                                   # buffer and counts 'usable' differently from
                                   # get_free_memory, so a clamp computed to keep
                                   # 20.3 GB of weights resident still offloaded
                                   # 401 MB and then aborted in a CUDA kernel.
_AUTO_MIN_POOL = 1536 * 1024**2    # below this, prefer a loud OOM to a silent crawl
_auto_cache = None                 # lazy {key: pool_bytes}
_auto_last = {"key": None, "model": None}   # what the next sampling run is
_auto_session = {}                 # key -> reserve pinned for this session


def _auto_cache_path():
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:
        import os
        base = os.path.dirname(os.path.abspath(__file__))
    import os
    return os.path.join(base, "h3_auto_reserve.json")


_AUTO_SCHEMA = 2  # bumped when the pool measurement changed meaning


def _auto_cache_load():
    global _auto_cache
    if _auto_cache is None:
        import io as _io, os
        _auto_cache = {}
        p = _auto_cache_path()
        if os.path.isfile(p):
            try:
                _auto_cache = json.load(_io.open(p, encoding="utf-8"))
            except Exception:
                _auto_cache = {}
        # Entries written before the full-weights fix can be too large by
        # however much the DiT had been offloaded that shot, and the store
        # keeps the largest value forever - so one bad shot poisons a shape
        # permanently. Drop pre-schema caches once instead of carrying them.
        if _auto_cache.pop("_schema", None) != _AUTO_SCHEMA:
            _stale = len(_auto_cache)
            if _stale:
                print("[H3AutoReserve] discarded %d cached reserve(s) written "
                      "before the partial-load measurement fix. Those numbers "
                      "could be inflated by the offloaded weight bytes and "
                      "never shrank. They re-measure on the next run."
                      % _stale, flush=True)
            _auto_cache = {}
    return _auto_cache


def _auto_cache_store(key, pool_bytes):
    """Record a pool measurement. Returns True if it raised the stored value."""
    cache = _auto_cache_load()
    prev = cache.get(key, 0)
    # keep the largest pool ever seen for the shape; shrinking on a lucky
    # run risks the cliff on the next unlucky one
    if pool_bytes <= prev:
        return False
    cache[key] = int(pool_bytes)
    try:
        import io as _io
        _out = dict(cache)
        _out["_schema"] = _AUTO_SCHEMA
        _io.open(_auto_cache_path(), "w", encoding="utf-8").write(
            json.dumps(_out, indent=1))
    except Exception as e:
        print(f"[H3AutoReserve] cache write failed ({e}) - measurements "
              f"will not persist across restarts", flush=True)
    return True


def _auto_cache_floor(key, pool_bytes):
    """Record a LOWER BOUND from a run whose peak was truncated by the card.

    Same ratchet as _auto_cache_store - it only ever raises - but named
    separately at the call site so it is obvious that the number is known to
    understate the real need. Without this a ceiling-bound shot contributes
    nothing at all, which is how the chained shots deadlocked: their payload key
    stayed empty, the fallback reserve was too small for them, and every attempt
    to measure was thrown away for being too small.
    """
    return _auto_cache_store(key, pool_bytes)


_auto_payload = {"sig": ""}


def _auto_set_payload(sig):
    """Samplers call this before each shot with a conditioning-payload
    signature (keyframes / reference blocks / audio refs). Two shots with
    the same latent shape but different payloads need DIFFERENT pools -
    render-verified 2026-08-10: shot 1 (bare) measured a 4.9 GB pool,
    shot 2 (chain keyframe + 10s self-anchor audio ref) then spilled to
    system RAM at that reserve and ran 5x slower per step."""
    _auto_payload["sig"] = str(sig or "")


def _payload_scheme(sig):
    """Which sampler wrote a payload signature: H3MultishotSampler writes
    "kf%da%d...", H3MultishotMemorySampler writes "%s%d_k%dr%d...". The two
    namespaces never compare equal, so a raw string mismatch is NOT evidence
    of a heavier payload (24 GB test-lab finding F008, 2026-08-16)."""
    return "kf" if str(sig or "").startswith("kf") else "cont"


def _payload_mult(src_sig, dst_sig):
    """Bare->payload scale when borrowing across signatures. Same string or
    different sampler scheme -> 1.0 (equal-cells evidence taken as-is);
    within one scheme the measured x1.6 bare->payload jump stands. The old
    flat "1.6 unless identical" fired on every CORE run against a
    Memory-sampler cache: 12.4 GB became a 19.8 GB request, 129 MB of DiT
    stayed resident, 66 s/it (24 GB lab, S1)."""
    if src_sig == dst_sig:
        return 1.0
    if _payload_scheme(src_sig) != _payload_scheme(dst_sig):
        return 1.0
    return 1.6


_PAYLOAD_ADD_BYTES = 4 * 1024**3    # a keyframe + an audio ref, in GB, roughly fixed


def _payload_need(bare, src_sig, dst_sig):
    """Pool NEED for a payload signature, from a bare (or other) measurement.

    24 GB test-lab finding F010 (2026-08-16/17): the flat x1.6 was above the
    93rd percentile of 16 measured bare->payload pairs (median 1.40x, max
    1.73x) and the cost is ADDITIVE, not multiplicative - a largely fixed
    number of GB, so a big relative jump on a 6 GB pool and negligible on a
    15 GB one. Measured: 6 GB bare -> ~1.5x; 11 GB -> +3.1..+4.0 GB;
    15 GB -> ~1.1x. A flat multiplier therefore over-reserved worst at large
    geometry - exactly where a 24 GB card evicts weights instead (S3 run 3
    reserved 20.6 GB for a 15.4 GB need). Model it as +4 GB capped at the old
    x1.6, so it can only ask for less than before, never more.
    """
    m = _payload_mult(src_sig, dst_sig)
    if m <= 1.0:
        return int(bare)
    return int(min(bare * 1.6, bare + _PAYLOAD_ADD_BYTES))


def _model_family(stem):
    """Coarse quant family for the reserve borrow. The pool tracks geometry
    WITHIN a family, not across: comfy-native quant paths (w4a8/int8/nvfp4)
    carry dequant scratch that GGUF does not - the same card's cache rates
    read 3.0-5.7 GB/Mcell for w4a8 against 2.2-2.5 for GGUF, and a GGUF-
    based borrow under-reserved a w4a8 first run into WDDM paging (the 24 GB lab box lab
    F009, 2026-08-16). Under-reserving is the fatal direction."""
    s = str(stem or "").lower()
    if re.search(r"q\d|gguf|curve|k_[msl]|_q[0-9]", s):
        return "gguf"
    return "native"


_CROSS_FAMILY_MULT = 1.8   # w4a8/gguf per-cell ratio measured on the 3090 cache


def _auto_key(model_name, cells, sig=None):
    try:
        import torch
        dev = torch.cuda.get_device_name(0)
    except Exception:
        dev = "cpu"
    import os
    stem = os.path.splitext(os.path.basename(model_name))[0]
    s = _auto_payload["sig"] if sig is None else sig
    return f"{dev}|{stem}|{cells}|{s}"


def _install_auto_reserve(patcher, model_name):
    """Shape-aware memory_required on the BaseModel (clone-safe)."""

    def memory_required(input_shape, *args, **kwargs):
        cells = 1
        try:
            for d in list(input_shape)[1:]:
                cells *= int(d)
        except Exception:
            cells = 0
        key = _auto_key(model_name, cells)
        _auto_last["key"] = key
        # comfy (and DynamicVRAM) call memory_required repeatedly - per load
        # AND per sampling step. The answer must be STABLE for a shape:
        # recomputing "60% of free" as free shrinks is a feedback loop
        # (reserving memory reduces free, which reduces the next answer).
        # Pin the first computation per (model, shape) for the session;
        # a fresh measurement invalidates the pin.
        pinned = _auto_session.get(key)
        if pinned is not None:
            return pinned
        cache = _auto_cache_load()
        measured = cache.get(key) or 0
        _known_need = int(measured)   # best evidence of true pool need
        if measured:
            # a real measurement carries its own x1.25 margin - the old
            # 8 GB floor here overrode good small measurements and forced
            # weight offload on 24 GB cards for nothing
            reserve = max(int(measured * _AUTO_MARGIN), 2 * 1024**3)
            how = f"measured pool {measured/2**30:.1f} GB x {_AUTO_MARGIN}"
        else:
            # unmeasured payload variant of a measured shape: estimate from
            # the sibling instead of guessing from free VRAM. Reference and
            # keyframe tokens ride every step; x1.6 covered the measured
            # bare->payload jump with margin.
            sib = sib_sig = None
            prefix = key.rsplit("|", 1)[0] + "|"
            mysig = key.rsplit("|", 1)[1]
            for k2, v2 in cache.items():
                if k2.startswith(prefix) and v2 and (sib is None or v2 > sib):
                    sib, sib_sig = v2, k2.rsplit("|", 1)[1]
            if sib:
                _known_need = _payload_need(sib, sib_sig, mysig)
                reserve = max(int(_known_need * _AUTO_MARGIN), _AUTO_FLOOR)
                how = (f"payload variant of a measured shape: sibling pool "
                       f"{sib/2**30:.1f} GB -> need {_known_need/2**30:.1f} GB "
                       f"(payload +4 GB capped x1.6) x {_AUTO_MARGIN}")
            else:
                # FIRST RUN at an unseen (model, shape). The pool tracks the
                # GEOMETRY, not the checkpoint - measured 2026-08-16: a Q8
                # GGUF and a mixed-precision file wanted the same ~11-13 GB
                # at the same cells. So borrow the nearest measurement on
                # this card from ANY model and shape, scaled by cell count,
                # before falling back to the free-VRAM placeholder - which
                # planned 3.3 GB for an 11 GB pool twice in one afternoon
                # and produced a crawl on a 5090 and a dead CUDA context on
                # a 3090. Max over candidates: over-reserving streams a few
                # GB of weights (cheap); under-reserving pages (fatal).
                borrowed = bfrom = None
                try:
                    mydev, _, _, mysig = key.split("|", 3)
                    # same-payload siblings first; only if none exist, other
                    # payloads with the bare->payload x1.6 on top. Otherwise a
                    # bare shot 1 borrows a pinned shot's pool times 1.6 and
                    # over-reserves by double.
                    # nearest shape wins, NOT the largest estimate: pools are
                    # not purely linear in cells (fixed overhead fattens the
                    # per-cell rate of small shapes), so extrapolating far
                    # overshoots - a real cache here scaled small w4a8 shapes
                    # to 26 GB while two same-cells entries said 14-15.
                    # rank: shape distance first, then same payload, then the
                    # larger estimate. A same-cells measurement beats a
                    # same-payload one from a distant shape - a real cache
                    # extrapolated distant small shapes to 26-31 GB while
                    # same-cells entries said 15.
                    best = None
                    myfam = _model_family(key.split("|", 3)[1])
                    for k2, v2 in cache.items():
                        try:
                            d2, m2, c2, s2 = k2.split("|", 3)
                            c2 = int(c2)
                        except ValueError:
                            continue
                        if d2 != mydev or not v2 or not c2 or not cells:
                            continue
                        ratio = cells / c2
                        if not (0.4 <= ratio <= 2.5):
                            continue   # no wild-scale extrapolation
                        fam_miss = 0 if _model_family(m2) == myfam else 1
                        est = int(_payload_need(v2 * ratio, s2, mysig)
                                  * (_CROSS_FAMILY_MULT if fam_miss else 1.0))
                        # same quant family first (F009), then nearest shape,
                        # then same payload, then the fatter estimate
                        rank = (fam_miss, abs(ratio - 1.0),
                                0 if s2 == mysig else 1, -est)
                        if best is None or rank < best:
                            best, borrowed, bfrom = rank, est, (c2, v2)
                except Exception:
                    borrowed = None
                if borrowed:
                    _known_need = borrowed
                    reserve = max(int(borrowed * _AUTO_MARGIN), _AUTO_FLOOR)
                    how = ("borrowed pool: %.1f GB measured at cells=%d "
                           "scaled to %.1f GB (pool tracks geometry, not "
                           "the checkpoint)"
                           % (bfrom[1] / 2**30, bfrom[0], borrowed / 2**30))
                    print("[H3AutoReserve] first run with this model at this "
                          "shape - borrowing a measured pool from another "
                          "model/shape on this card: %.1f GB at cells=%d "
                          "scales to %.1f GB here. This shot still records "
                          "its own measurement."
                          % (bfrom[1] / 2**30, bfrom[0], borrowed / 2**30),
                          flush=True)
                else:
                    try:
                        import comfy.model_management as mm
                        free = mm.get_free_memory(mm.get_torch_device())
                    except Exception:
                        free = 24 * 1024**3
                    reserve = max(int(min(free * _AUTO_FRACTION,
                                          free - _AUTO_WEIGHT_NUCLEUS)),
                                  _AUTO_FLOOR)
                    how = (f"first run at this shape: "
                           f"{_AUTO_FRACTION:.0%} of free")
        # CLAMP AGAINST THE CARD. Every GB reserved here comes out of the
        # weights budget, and a DiT that misses a FULL load streams the
        # remainder over PCIe every step. Measured 2026-08-12 at 960x544:
        # shot 1 reserved 7.8 GB and loaded completely at 18.8 s/it; shot 2's
        # larger payload reserved 9.4 GB, left the DiT 399 MB short, and ran
        # at 283 s/it - a 15x collapse bought by 1.6 GB of headroom the
        # measurement said was not needed. The failure modes are asymmetric:
        # too small OOMs loudly and you fix it, too large silently costs 15x.
        # So the pool yields to the weights - but ONLY while the cut stays at
        # or above the measured need. Cutting below it does not keep the
        # weights resident (the allocator evicts them for real allocations
        # regardless - measured on a 3090: clamped to 1.5 GB "to keep weights
        # resident" and the shot's own measurement then read resident 0.0),
        # so past that line the weights yield instead.
        try:
            import comfy.model_management as _cm
            _dev = _cm.get_torch_device()
            _free = _cm.get_free_memory(_dev)
            _w = int(patcher.model_size())
            _cap = int(_free - _w - _AUTO_KEEPOUT)
            if _cap <= 0:
                # Before accepting the tight regime, sweep leftovers: an item
                # that died without a clean boundary leaves the previous
                # model resident, and only the multishot remote-TE lane had a
                # shot-1 sweep - the classic/local-TE lane planned against
                # the stale number and paged at 141 W (twice, 2026-08-22).
                # Everything resident here has finished its work (the TE
                # encodes before the DiT plans), so unloading costs one
                # reload at worst; ComfyUI reloads on demand.
                try:
                    _before_sw = _free
                    _cm.unload_all_models()
                    _cm.free_memory(_cm.get_total_memory(_dev) * 0.9, _dev)
                    _free = _cm.get_free_memory(_dev)
                    _cap = int(_free - _w - _AUTO_KEEPOUT)
                    if _free - _before_sw > 2**30:
                        print("[H3AutoReserve] cleared %.1f GB of leftovers "
                              "before reserve planning (reported free "
                              "%.1f -> %.1f GB)."
                              % ((_free - _before_sw) / 2**30,
                                 _before_sw / 2**30, _free / 2**30),
                              flush=True)
                except Exception:
                    pass
            if _cap <= 0:
                # The weights alone exceed free VRAM. Reserving the measured
                # pool here is the WORST possible move: every byte of reserve
                # pushes another byte of weights out, and the old `_cap > 0`
                # guard skipped the clamp entirely in exactly this case. A
                # 3090 with 10.3 GB free and 20.3 GB of weights reserved
                # 22.1 GB and loaded "0.00 MB usable, 0.00 MB loaded,
                # 20796.43 MB offloaded" - every layer streamed off disk on
                # every step, 128 s/it against ~29 s/it resident, and after
                # four hours the file-reader gave out with
                # hostbuf_file_reader_read failed.
                _was = reserve
                # NOT the floor. Reserving the minimum guarantees the weights
                # load but starves the activation pool, and the allocator then
                # spills ACTIVATIONS instead - measured worse than the problem
                # it replaced: 36 -> 55 -> 331 s/it across three shots while
                # every shot still reported 'loaded completely, full load:
                # True'. Use the measured pool when a sibling shot has given
                # us one: shot 2 measured 8.1 GB and peaked at 22.6 with 14.5
                # GB of weights, which fits a 24 GB card exactly. The inflated
                # payload estimate (x1.6 x1.25 compounding off an already
                # bumped sibling) is what asks for 16 GB and does not fit.
                reserve = (max(int(_known_need), _AUTO_MIN_POOL)
                           if _known_need else _AUTO_MIN_POOL)
                reserve = max(min(reserve, int(_free - _AUTO_KEEPOUT)),
                              _AUTO_MIN_POOL)
                how += (" | TIGHT %.1f -> %.1f GB: weights %.1f GB vs "
                        "%.1f GB reported free"
                        % (_was / 2**30, reserve / 2**30, _w / 2**30,
                           _free / 2**30))
                print("[H3AutoReserve] TIGHT: %.1f GB of weights against "
                      "%.1f GB reported free, so this shot has no headroom. "
                      "Reserving %.1f GB (the measured pool) rather than the "
                      "payload estimate, which does not fit. Note the reported "
                      "figure understates what ComfyUI ends up with, so the "
                      "weights may still load completely - watch for a spill "
                      "instead: high GPU utilisation at low wattage. If the "
                      "render crawls, lower frames_per_shot or resolution, free "
                      "the other ComfyUI instance, or load a smaller DiT."
                      % (_w / 2**30, _free / 2**30, reserve / 2**30), flush=True)
                # Driver headroom applies HERE too. Reported free is misleading
                # in this branch (resident weights count as used but get
                # reused), so the peak still lands wherever weights+pool put
                # it - measured 2026-08-16 on a 5090: the tight requeue of a
                # shape whose first run peaked at a healthy 29.4/32.6 sat at
                # 31.5/32.6 and crawled at 142 W. The main headroom block below
                # is gated on _cap > 0 and its clamp uses reported free, both
                # wrong for this regime, so bump against TOTAL directly:
                # streaming ~2 GB more weights is cheap, the last few percent
                # of VRAM are not.
                try:
                    _total_t = _cm.get_total_memory(_dev)
                except Exception:
                    _total_t = _free
                # No keepout credit: comfy fills weights to free-reserve no
                # matter what, so crediting keepout under-raised the reserve
                # by ~1 GB and left the peak in the WDDM zone (767 s/it,
                # Zara run 2026-08-23). Full bump; 1 GB extra streaming is
                # the measured-cheap side.
                _bump = int(_total_t * 0.09)
                if _bump:
                    reserve += _bump
                    how += (" | +driver headroom (tight) +%.1f GB"
                            % (_bump / 2**30))
                    print("[H3AutoReserve] driver headroom (tight path): "
                          "raising the reserve by %.1f GB so extra weights "
                          "stream instead of the peak riding the last few "
                          "percent of VRAM (that zone measured 2-12x slower)."
                          % (_bump / 2**30), flush=True)
            elif reserve > _cap:
                _was = reserve
                _card_max = max(int(_free - _AUTO_KEEPOUT), _AUTO_MIN_POOL)
                if _known_need and _cap < _known_need:
                    # The cut would land below the measured need. Weight
                    # residency is not achievable in this regime - the
                    # allocator evicts weights to satisfy the sampler's real
                    # allocations no matter what is reserved (3090: clamped to
                    # 1.5 GB "to keep weights resident", measurement then read
                    # resident 0.0, 24.3 GB driver spill, ~2x step time). So
                    # give the pool its need, bounded by the card, and let the
                    # weights stream: that is the cheaper side here.
                    # Bare need, not need*margin. The x1.25 margin guards a
                    # pool overrun against PINNED weights; here the weights
                    # stream regardless, so an overrun just evicts a little
                    # more of them - graceful. Every GB of margin trimmed is a
                    # GB of weights that stays resident instead of streaming
                    # every step (measured: 18.4 reserve left 3.3 GB resident,
                    # 14.7 leaves 7.0).
                    reserve = min(max(int(_known_need), _AUTO_MIN_POOL),
                                  _card_max)
                    _resid = max(0, int(_free - _AUTO_KEEPOUT) - reserve)
                    how += (" | NEED %.1f -> %.1f GB (bare need, margin "
                            "yielded to weights): ~%.1f GB of weights can stay "
                            "resident"
                            % (_was / 2**30, reserve / 2**30, _resid / 2**30))
                    print("[H3AutoReserve] pool need %.1f GB cannot fit "
                          "beside %.1f GB of weights in %.1f GB free. "
                          "Reserving %.1f GB for the pool and letting the "
                          "weights stream - clamping the pool here does not "
                          "keep the weights resident, it only adds a driver "
                          "spill on top of the offload."
                          % (_known_need / 2**30, _w / 2**30,
                             _free / 2**30, reserve / 2**30), flush=True)
                    if _card_max < _known_need:
                        print("[H3AutoReserve] WARNING: even with zero weights "
                              "resident the card has %.1f GB for a %.1f GB "
                              "pool. This can die inside a CUDA kernel. Lower "
                              "frames_per_shot or resolution."
                              % (_card_max / 2**30, _known_need / 2**30),
                              flush=True)
                else:
                    # No measurement says the cut goes below need, so this is
                    # margin-trimming: keep the weights resident. Measured
                    # 2026-08-12: a 399 MB weight shortfall cost 15x streaming.
                    # BUT the opposite cliff is real too (2026-08-21, 5090):
                    # trimming a BORROWED pool's margin to squeeze 20.4 GB of
                    # weights fully resident left ~0.6 GB of slack where the
                    # streaming path keeps ~1.9, and the peak spilled into
                    # driver memory - 590 s/it against the streaming path's 27
                    # on the same total budget. The trim gamble is only worth
                    # taking when it is SMALL: if keeping the weights resident
                    # means eating more than 0.75 GB of the pool's margin,
                    # stream a sliver instead - that side of the trade is
                    # measured shallow (1-2 GB streaming ~ 27 s/it all day).
                    # The discriminator is HEADROOM, not trim size: every
                    # clean run keeps ~9% of the card as driver slack, and
                    # both measured crawls ran with less (65 s/it at reduced
                    # slack, 590 s/it at ~0.6 GB). Keep the weights resident
                    # ONLY if the clamped pool still holds the bare need PLUS
                    # the full driver-headroom bump; otherwise stream - the
                    # multi-GB streaming regime is measured shallow (~27 s/it
                    # all day on this card).
                    try:
                        import comfy.model_management as _mmt
                        _tt = _mmt.get_total_memory(_mmt.get_torch_device())
                    except Exception:
                        _tt = _free
                    _full_bump = int(_tt * 0.09)  # no keepout credit (2026-08-23)
                    _clamp_res = max(_cap, _AUTO_MIN_POOL)
                    _bare = int(_known_need) if _known_need else int(reserve / 1.25)
                    if _clamp_res - _bare < _full_bump:
                        reserve = min(_bare + _full_bump, _card_max)
                        how += (" | resident-clamp REFUSED (slack %.1f GB < "
                                "headroom %.1f GB): streaming"
                                % ((_clamp_res - _bare) / 2**30,
                                   _full_bump / 2**30))
                        print("[H3AutoReserve] keeping all %.1f GB of weights "
                              "resident would leave only %.1f GB of slack "
                              "over the pool's bare need - under the %.1f GB "
                              "driver headroom every clean run keeps "
                              "(2026-08-21: that squeeze ran 22x slower than "
                              "streaming). Reserving %.1f GB and letting "
                              "~%.1f GB of weights stream."
                              % (_w / 2**30, (_clamp_res - _bare) / 2**30,
                                 _full_bump / 2**30, reserve / 2**30,
                                 max(0.0, (_w - (_free - reserve
                                 - _AUTO_KEEPOUT))) / 2**30), flush=True)
                        _clamped_resident = False
                    else:
                        reserve = _clamp_res
                        _clamped_resident = True
                    if _clamped_resident:
                        how += (" | CLAMPED %.1f -> %.1f GB to keep the "
                                "weights (%.1f GB) resident out of %.1f GB "
                                "free" % (_was / 2**30, reserve / 2**30,
                                          _w / 2**30, _free / 2**30))
                    # Only shout when the pre-clamp figure came from EVIDENCE.
                    # On a first run at an unseen shape there is no
                    # measurement, and `_was` is the placeholder from the
                    # "%.0f%% of free" branch - a fraction of whatever happens
                    # to be free, not an estimate of this shot's need. It is
                    # therefore always far above the clamp, so this warning
                    # fired on every new shape and predicted a server-killing
                    # crash for renders that were fine.
                    #
                    # Field log 2026-08-15, 5090, two different shapes:
                    #   cells=6384960  "asked for 26.5 GB"
                    #   cells=6993216  "asked for 26.5 GB"   <- 10% more pixels
                    # Identical, because both are 88% of the same 30.1 GB free.
                    # Both clamped, both rendered clean, and both then MEASURED
                    # 11.4 GB - less than half the figure being warned about.
                    # A warning that cries wolf on every new resolution teaches
                    # people to cancel jobs that would have worked.
                    if _was > reserve * 1.35 and _known_need:
                        print("[H3AutoReserve] WARNING: this shape previously "
                              "MEASURED a %.1f GB activation pool and only "
                              "%.1f GB is available after the %.1f GB of "
                              "weights. That is not a slow render - it usually "
                              "dies inside a CUDA kernel and takes the server "
                              "with it. Lower frames_per_shot or resolution, or "
                              "load a smaller quantisation of the DiT."
                              % (_known_need / 2**30, reserve / 2**30,
                                 _w / 2**30), flush=True)
                    elif _was > reserve * 1.35:
                        print("[H3AutoReserve] first run at this shape - the "
                              "%.1f GB figure is a placeholder (%.0f%% of "
                              "free), not a measurement, and has been clamped "
                              "to %.1f GB so the %.1f GB of weights stay "
                              "resident. The real pool gets measured during "
                              "this shot and used from the next one on."
                              % (_was / 2**30, _AUTO_FRACTION * 100,
                                 reserve / 2**30, _w / 2**30), flush=True)
                if measured and reserve < measured:
                    how += (" [tight: below the measured pool %.1f GB]"
                            % (measured / 2**30))
            # DRIVER HEADROOM (measured 2026-08-16). Five identical runs
            # at ~96% VRAM took 27-175 minutes - same shape, same seed, a
            # lottery. The same run with the reserve raised so ~3.6 GB of
            # weights streamed instead peaked at 29.5/32.6 and took 15
            # minutes, faster than every resident run. WDDM demotes
            # unpredictably in the last few percent of VRAM, and streamed
            # weights are cheap (core prefetch overlaps them) - so when
            # weights + pool would land in that zone, RAISE the reserve:
            # it is the one lever that pushes weights off and peak down.
            if _cap > 0:
                try:
                    _total = _cm.get_total_memory(_dev)
                except Exception:
                    _total = _free
                _wddm = int(_total * 0.09)
                _pool_real = int(_known_need) if _known_need else reserve
                # Full wddm on top of the pool - no keepout credit (see the
                # tight-path note; the 2026-08-23 767 s/it crawl planned
                # reserve = pool + wddm - keepout and peaked 1.1 GB into
                # the zone this exists to keep clear).
                _target = _pool_real + _wddm
                if (_w + _pool_real + _AUTO_KEEPOUT + _wddm > _free
                        and reserve < _target):
                    _target = min(_target,
                                  max(int(_free - _AUTO_KEEPOUT),
                                      _AUTO_MIN_POOL))
                    how += (" | +driver headroom %.1f -> %.1f GB"
                            % (reserve / 2**30, _target / 2**30))
                    print("[H3AutoReserve] driver headroom: raising the "
                          "reserve %.1f -> %.1f GB so ~%.1f GB of weights "
                          "stream instead of riding the last few percent "
                          "of VRAM (that zone measured 2-12x slower)."
                          % (reserve / 2**30, _target / 2**30,
                             max(0.0, (_w + _pool_real + _AUTO_KEEPOUT
                                       + _wddm - _free)) / 2**30),
                          flush=True)
                    reserve = _target
        except Exception:
            pass
        _auto_session[key] = reserve
        print(f"[H3AutoReserve] shape cells={cells}: reserving "
              f"{reserve/2**30:.1f} GB ({how})", flush=True)
        return reserve

    patcher.model.memory_required = memory_required
    patcher.memory_required = memory_required
    _auto_last["model"] = model_name


def _auto_measure_begin():
    """Call right before sampling: snapshot the allocator + clock."""
    import time as _t
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        return {"res": torch.cuda.memory_reserved(), "t0": _t.time()}
    except Exception:
        return {"res": None, "t0": _t.time()}


def _auto_measure_end(before, patcher=None, steps=None):
    """Call right after sampling: cache the real pool, and DETECT the two
    silent failure modes by name - a system-RAM spill (peak at the card's
    ceiling + step time collapsed) used to present as an unexplained 5-10x
    slowdown with nothing in the log."""
    import time as _t
    if not isinstance(before, dict):
        before = {"res": before, "t0": None}
    key = _auto_last["key"]
    if key is None:
        return
    # ---- step-time tracking (works even when CUDA stats are unavailable)
    sit = None
    if before.get("t0") and steps:
        sit = (_t.time() - before["t0"]) / max(1, int(steps))
        base_key = "sit|" + key.rsplit("|", 1)[0]   # same shape, any payload
        best = _auto_session.get(base_key)
        if best is None or sit < best:
            _auto_session[base_key] = sit
        elif best > 0 and sit > 2.5 * best:
            print(f"[H3AutoReserve] SLOWDOWN: {sit:.0f}s/step vs "
                  f"{best:.0f}s/step earlier this session ({sit/best:.1f}x). "
                  f"This is the VRAM-spill signature: the driver is paging "
                  f"to system RAM instead of erroring. Fix: raise the "
                  f"activation reserve, drop resolution/frames, or remove "
                  f"reference payload (audio refs / keyframes ride every "
                  f"step).", flush=True)
    if before.get("res") is None:
        return
    try:
        import torch
        peak = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        if peak >= total * 0.97:
            print(f"[H3AutoReserve] WARNING: peak reserved "
                  f"{peak/2**30:.1f} GB of {total/2**30:.1f} GB - the "
                  f"allocator hit the card's ceiling; any overflow was "
                  f"paged to system RAM by the driver (silent, slow).",
                  flush=True)
        loaded = 0
        try:
            loaded = int(patcher.loaded_size()) if patcher is not None else 0
        except Exception:
            try:
                loaded = int(getattr(patcher.model,
                                     "model_loaded_weight_memory", 0))
            except Exception:
                loaded = 0
        full = 0
        try:
            full = int(patcher.model_size()) if patcher is not None else 0
        except Exception:
            full = 0
        # `peak` is max_memory_reserved() - VRAM. Weights that were offloaded
        # to the host were never in VRAM and so were never in peak. Subtract
        # the RESIDENT bytes, not the full model size.
        #
        # Whether the sample is usable depends on residency, in three cases:
        #
        #   fully offloaded  peak IS the pool, nothing to subtract. This is the
        #                    cleanest sample available - record it. On a 24 GB
        #                    card at production shapes this is the NORMAL mode,
        #                    not a fault, so discarding it stops all learning.
        #   partial          peak is bounded by what fit on the card rather than
        #                    by what the shot wanted. Measures the ceiling, not
        #                    the need. Discard.
        #   fully resident   subtract the weights and record, as always.
        #
        # An earlier version of this collapsed the first two cases and discarded
        # both, which wiped the cache and then left every shot reporting "first
        # run at this shape" - 1.5 GB reserves, 24.3 GB of driver spill and ~2x
        # step time on the 3090. Diagnosed there with before/after logs.
        pool = peak - before["res"] - loaded
        _frac = (loaded / float(full)) if full else 1.0
        _partial = bool(full) and 0.02 < _frac < 0.98
        # 2.2.5: a blanket discard on every partial load DEADLOCKS the chained
        # shots. Field log 2026-08-14, 5090, Q8_0 at 736x1280x243: shot 1 loads
        # completely and measures fine, shots 2-4 carry the context pin plus the
        # self-anchor audio ref, cannot fit the 20.8 GB DiT beside them, load
        # partially - and are then discarded. Their payload key therefore never
        # receives a single measurement, so it keeps the fallback reserve, so it
        # keeps loading partially. Learning requires a full load; a full load
        # requires the knowledge the rule refuses to record. Every shot after the
        # first ran 57 s/it against shot 1's 39.
        #
        # The discard is only actually justified when the peak was CEILING-BOUND
        # - i.e. the run pressed against the card and the pool got truncated. If
        # peak sits well below total VRAM the activations completed normally and
        # the pool figure is honest, even though some weights were streaming. So
        # gate the discard on proximity to the ceiling instead of on residency
        # alone, and when we do discard, still keep the observation as a FLOOR so
        # the estimate can only ratchet toward the truth rather than never move.
        _total = 0
        try:
            import torch as _t
            _total = int(_t.cuda.get_device_properties(0).total_memory)
        except Exception:
            _total = 0
        _ceiling_bound = bool(_total) and peak > _total * 0.94
        if _partial and _ceiling_bound:
            _floor = _auto_cache_floor(key, pool)
            print(f"[H3AutoReserve] measurement capped: only "
                  f"{loaded/2**30:.1f} of {full/2**30:.1f} GB of weights were "
                  f"resident AND peak {peak/2**30:.1f} GB pressed the "
                  f"{_total/2**30:.1f} GB card, so the pool was truncated. "
                  f"Keeping {pool/2**30:.1f} GB as a floor"
                  f"{' (raised)' if _floor else ''}; not treating it as the "
                  f"full need.", flush=True)
        elif _partial:
            # streaming weights, but the pool itself was never squeezed
            if pool > 512 * 1024**2:
                _auto_cache_store(key, pool)
                _auto_session.pop(key, None)
                print(f"[H3AutoReserve] measured pool {pool/2**30:.1f} GB with "
                      f"weights streaming ({loaded/2**30:.1f} of "
                      f"{full/2**30:.1f} GB resident). Peak {peak/2**30:.1f} GB "
                      f"stayed clear of the {_total/2**30:.1f} GB ceiling, so "
                      f"the activation figure is sound - recording it. This is "
                      f"the sample chained shots could never contribute before.",
                      flush=True)
        elif bool(full) and _frac <= 0.02 and pool > 512 * 1024**2:
            # ~0% of the weights were resident: the run streamed everything,
            # and with the whole card to itself the pool inflates to fill
            # whatever reserve it was handed (allocator caches). If the
            # figure hugs the reserve it is an artifact OF the reserve, and
            # storing it locks the shape into all-streaming forever -
            # measured 2026-08-16 on the 3090: a cancelled 21.8 GB-reserve
            # run recorded a "21.7 GB pool" at a shape two healthy runs had
            # measured at 14-15.
            _pinned = _auto_session.get(key) or 0
            if _pinned and pool >= _pinned * 0.85:
                print(f"[H3AutoReserve] discarding this shot's pool figure "
                      f"({pool/2**30:.1f} GB): no weights were resident and "
                      f"it hugs the {_pinned/2**30:.1f} GB reserve, so it "
                      f"measures the reserve, not the need.", flush=True)
            else:
                _auto_cache_store(key, pool)
                _auto_session.pop(key, None)
                print(f"[H3AutoReserve] measured pool {pool/2**30:.1f} GB "
                      f"with all weights streaming - it sits well under the "
                      f"reserve, so the figure is real need.", flush=True)
        elif pool > 512 * 1024**2:          # ignore no-op runs
            _auto_cache_store(key, pool)
            _auto_session.pop(key, None)     # re-pin from measured
            print(f"[H3AutoReserve] measured pool {pool/2**30:.1f} GB for "
                  f"this shape+payload (peak {peak/2**30:.1f} - resident "
                  f"weights {loaded/2**30:.1f}) - next run reserves "
                  f"{max(pool*_AUTO_MARGIN, 2*1024**3)/2**30:.1f} GB",
                  flush=True)
    except Exception:
        pass


class H3ModelLoaderAny:
    """One dropdown, both formats: .safetensors loads through comfy core,
    .gguf routes through ComfyUI-GGUF (patched for minimax_h3). Keeps the
    published workflow at exactly one loader node."""

    @staticmethod
    def _list_names(folder="diffusion_models"):
        """Every model file the dropdown offers: core's list plus a RECURSIVE
        walk for .gguf, which is not in supported_pt_extensions so
        get_filename_list never returns it (and a flat listdir misses
        anything filed under diffusion_models/gguf/). Shared by INPUT_TYPES
        and _resolve_name - the resolver used to consult get_filename_list
        alone, so its "moved into a gguf/ subfolder" fallback could never
        find a gguf (2.6.0, found on the 24 GB box)."""
        import folder_paths
        import os
        try:
            files = list(folder_paths.get_filename_list(folder))
        except Exception:
            files = []
        gguf = []
        try:
            dirs = folder_paths.get_folder_paths(folder)
        except Exception:
            dirs = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for root, _dirs, fs in os.walk(d):
                for f in fs:
                    if f.lower().endswith(".gguf"):
                        gguf.append(os.path.relpath(os.path.join(root, f), d))
        return sorted(set(files) | set(gguf))

    @classmethod
    def INPUT_TYPES(cls):
        names = cls._list_names("diffusion_models")
        return {"required": {"model_name": (names, {
            "tooltip": "safetensors or GGUF - loader routes automatically."})},
            "optional": {"activation_reserve_gb": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                "tooltip": "0 = AUTO (recommended). The pack sizes the "
                "activation reserve for the actual render shape, measures the "
                "real peak each run, and tightens itself per machine - lower "
                "resolutions get faster automatically. Set a number only to "
                "pin the reserve by hand; that number is for ONE resolution "
                "and the wrong number is 5-10x slower, not a little slower."})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders/minimax"

    @classmethod
    def VALIDATE_INPUTS(cls, model_name=None, **kwargs):
        # ComfyUI validates every combo value BEFORE anything runs, so a
        # workflow saved on a box that keeps the file at the root fails on a
        # box that filed it under gguf/ (or vice versa) with value_not_in_list
        # and nothing renders (24 GB lab F007: all three shipped workflows
        # unqueueable there). Accept the value here; load() resolves a moved
        # file by unique basename and raises a message that names the file
        # when it truly is not there. Same pattern RiftPromptSource uses.
        return True

    @staticmethod
    def _resolve_name(model_name):
        """F007 (24 GB lab): a saved combo value stops matching the moment the
        model tree is reorganised (a file moved into a gguf/ subfolder, or a
        workflow saved on a box that keeps it at the root). Rather than fail
        the whole queue with value_not_in_list, fall back to a UNIQUE basename
        match across the folders this loader reads, and say what was resolved.
        Ambiguous (two files, same basename) stays an error - guessing wrong
        is worse than stopping."""
        import folder_paths
        import os
        want = os.path.basename(str(model_name)).lower()
        cands = []
        for folder in ("diffusion_models", "unet", "checkpoints"):
            try:
                for f in H3ModelLoaderAny._list_names(folder):
                    if os.path.basename(f).lower() == want:
                        cands.append((folder, f))
            except Exception:
                pass
        # exact match wins silently
        for folder, f in cands:
            if f == model_name:
                return model_name
        uniq = sorted(set(f for _, f in cands))
        if len(uniq) == 1 and uniq[0] != model_name:
            print("[H3ModelLoader] %r is not at the saved path; resolved by "
                  "basename to %r (the model tree moved since this workflow "
                  "was saved)." % (model_name, uniq[0]), flush=True)
            return uniq[0]
        return model_name

    @staticmethod
    def _warn_quant_backend(model_name):
        """24 GB lab finding, 2026-08-17: on torch < cu130 comfy-kitchen's cuda
        backend is disabled and triton is off unless --enable-triton-backend, so
        every comfy-native quantised op (w4a8/int8/nvfp4/fp8) runs the EAGER
        fallback - weights dequantised to bf16 before each matmul. Measured
        A/B, one flag, same seed: 36.7 -> 17.6 s/it (-52%), and ~1.6x the
        activation pool. GGUF bypasses comfy-kitchen entirely and is unaffected.
        Say it once at load so nobody pays 2x for a checkpoint they chose for
        speed."""
        n = str(model_name).lower()
        if n.endswith(".gguf"):
            return
        try:
            import comfy_kitchen as ck
            b = ck.list_backends()
        except Exception:
            return
        def live(name):
            info = b.get(name) if isinstance(b, dict) else None
            return bool(info) and info.get("available") and not info.get("disabled")
        if live("cuda") or live("triton"):
            return
        print("[H3ModelLoader] NOTE: only comfy-kitchen's 'eager' backend is "
              "live on this box (cuda needs torch cu130+; triton needs "
              "--enable-triton-backend). Comfy-native quantised checkpoints "
              "(w4a8/int8/nvfp4/fp8) dequantise to bf16 every step here - "
              "measured ~2x slower and ~1.6x the activation pool. Either add "
              "--enable-triton-backend to your launch line, or use the GGUF "
              "build of this model, which is unaffected.", flush=True)

    def load(self, model_name, activation_reserve_gb=0.0):
        model_name = self._resolve_name(model_name)
        self._warn_quant_backend(model_name)
        out = self._load_inner(model_name)
        patcher = out[0]
        # stash the checkpoint name so samplers can check task compatibility
        # (fl2va = first/last-frame hand-off, ref2va = reference rows). The
        # wrong pairing does not error - it silently underperforms.
        try:
            patcher.model.h3_checkpoint_name = str(model_name)
        except Exception:
            pass
        if activation_reserve_gb and activation_reserve_gb > 0:
            _cap = int(activation_reserve_gb * (1024 ** 3))
            # Must live on the inner BaseModel, not the ModelPatcher: LoRA
            # stacks and guiders clone() the patcher before sampling and an
            # instance attribute does not survive the clone, silently
            # restoring comfy's estimate. Clones share this BaseModel.
            patcher.model.memory_required = lambda *a, _c=_cap, **k: _c
            patcher.memory_required = lambda *a, _c=_cap, **k: _c
            print(f"[H3ModelLoader] activation reserve PINNED at "
                  f"{activation_reserve_gb:.1f} GB (manual - correct for one "
                  f"resolution only; 0 = auto adapts to any)", flush=True)
        else:
            _install_auto_reserve(patcher, model_name)
        return out

    def _load_inner(self, model_name):
        import folder_paths
        # With VALIDATE_INPUTS accepting any name, the not-found case lands
        # here instead of at queue time - so say it clearly, with the folder
        # ComfyUI actually searched, before either loader path gets a chance
        # to fail obscurely.
        try:
            _found = folder_paths.get_full_path("diffusion_models", model_name)
        except Exception:
            _found = None
        if not _found:
            import os as _os
            _roots = [d for d in folder_paths.get_folder_paths("diffusion_models")]
            _hit = None
            for _d in _roots:
                if _os.path.isfile(_os.path.join(_d, model_name)):
                    _hit = _os.path.join(_d, model_name)
                    break
            if not _hit:
                raise RuntimeError(
                    "[H3ModelLoader] model file not found: %r. Searched: %s. "
                    "Pick your model file in this node's dropdown (a workflow "
                    "saved on another machine remembers a name your models "
                    "folder does not have)." % (model_name, ", ".join(_roots)))
        if model_name.lower().endswith(".gguf"):
            # resolve the live UnetLoaderGGUF from the global registry -
            # custom node packages load under mangled module names, so the
            # registry is the only stable handle.
            import nodes as core_nodes
            cls = core_nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
            if cls is None:
                raise RuntimeError(
                    "ComfyUI-GGUF not loaded - install/enable it and restart.")
            # ComfyUI-GGUF rejects unknown architectures before reading any
            # tensor, and upstream does not know minimax_h3. Import-time
            # patching covers the packaged install; re-assert here in case
            # ComfyUI-GGUF loaded after us. The relative import only exists
            # in the packaged install - LOOSE-FILE installs (this file
            # dropped straight into custom_nodes/) have no parent package,
            # so fall back to doing the patch inline.
            try:
                from .h3_gguf_arch import ensure_minimax_arch
                ensure_minimax_arch()
            except ImportError:
                import sys as _sys
                for _m in list(_sys.modules.values()):
                    try:
                        if (_m is not None
                                and isinstance(getattr(_m, "IMG_ARCH_LIST",
                                                       None), set)
                                and hasattr(_m, "TXT_ARCH_LIST")):
                            if "minimax_h3" not in _m.IMG_ARCH_LIST:
                                _m.IMG_ARCH_LIST.add("minimax_h3")
                                print("[H3ModelLoader] taught ComfyUI-GGUF "
                                      "the 'minimax_h3' architecture (in "
                                      "memory, loose-file fallback)",
                                      flush=True)
                            break
                    except Exception:
                        continue
            return cls().load_unet(model_name)
        import comfy.sd
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return (comfy.sd.load_diffusion_model(path),)


_UPSCALER_UTILS = None


def _load_upscaler_utils():
    """Load ComfyUI-MiniMaxH3_LatentUpscaler's utils.py by path (cached).

    Two-pass upscale reuses that pack's NestedTensor upscale + CONST re-noise
    math rather than duplicating it. Works whether this file is installed
    loose in custom_nodes or inside ComfyUI-H3-Multishot/.
    """
    global _UPSCALER_UTILS
    if _UPSCALER_UTILS is not None:
        return _UPSCALER_UTILS
    import importlib.util
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "ComfyUI-MiniMaxH3_LatentUpscaler", "utils.py"),
        os.path.join(os.path.dirname(here),
                     "ComfyUI-MiniMaxH3_LatentUpscaler", "utils.py"),
    ]
    try:
        import folder_paths
        for d in folder_paths.get_folder_paths("custom_nodes"):
            candidates.append(
                os.path.join(d, "ComfyUI-MiniMaxH3_LatentUpscaler", "utils.py"))
    except Exception:
        pass
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(
                "h3_latent_upscaler_utils", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _UPSCALER_UTILS = mod
            return mod
    raise RuntimeError(
        "two_pass_upscale needs the ComfyUI-MiniMaxH3_LatentUpscaler pack "
        "(github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler) installed in "
        "custom_nodes - it provides the NestedTensor upscale/re-noise math.")


def _upscale_av_exact(tr, latent_dict, target_h, target_w):
    """Spatially upscale the VIDEO member of an AV latent to EXACT latent dims.

    The pack's own upscaler works from a single scale_by, which can round to a
    grid one cell off the requested resolution; sampling to a fixed widget
    resolution needs exact dims so every shot decodes to width x height.
    """
    import torch.nn.functional as F
    members, was_nested = tr.extract_tensor(latent_dict["samples"])
    v = members[0]
    orig = tuple(v.shape)
    x = v
    if len(orig) > 4:  # [B,C,T,H,W] -> [B*T,C,H,W]
        x = x.reshape(orig[0], orig[1], -1, orig[-2], orig[-1])
        x = x.movedim(2, 1).reshape(-1, orig[1], orig[-2], orig[-1])
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear",
                      align_corners=False)
    if len(orig) > 4:
        x = x.reshape(orig[0], -1, orig[1], target_h, target_w).movedim(2, 1)
    out = latent_dict.copy()
    out["samples"] = tr.wrap_tensor([x, *members[1:]], was_nested=was_nested)
    return out



# ---------------------------------------------------------------------------
# Chain gain control (seam sharpening)
#
# MEASURED 2026-08-08 on 6 real multishot renders, both rigs, no LoRA: every
# content-continuous seam steps UP in texture energy by +25..47% (5-7x the
# non-seam control), and because each shot tail becomes the next shot keyframe
# anchor, the level COMPOUNDS - a 6-shot chain measured 3.3x sharper by its
# last shot than its first. The loop: anchor at level L -> the model generates
# its continuation at ~1.25-1.47 x L -> that output tail is the next anchor.
# Gain > 1 in an autoregressive loop ratchets.
#
# These helpers measure texture energy (variance of a 3x3 Laplacian on luma)
# and apply a separable gaussian, so the chain can be level-controlled.
# ---------------------------------------------------------------------------

def _cg_lap_var(img):
    """CONTRAST-NORMALISED texture energy of an IMAGE batch [B,H,W,C] in 0..1.

    var(laplacian) / var(luma). The raw laplacian scales with local contrast,
    so on its own it cannot tell "the shot got brighter" from "the shot got
    sharper" - and a leveller driven by it chases exposure instead of detail
    (measured: it held the answering-machine scene flat, whose contrast was
    steady, and missed a 2.7x drift in the dim coma scene, whose contrast
    swung 0.13-0.19). Dividing by the frame's own variance measures detail
    SCALE, which is what must stay constant across a chain.

    Letterbox bars are excluded - pure black borders would drag both terms.
    """
    import torch
    import torch.nn.functional as F
    x = img if img.ndim == 4 else img.unsqueeze(0)
    g = (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114).unsqueeze(1)
    if g.shape[-1] > 8 and g.shape[-2] > 8:
        mx_r = g.amax(dim=(0, 1, 3)) > 0.02      # rows with any signal
        mx_c = g.amax(dim=(0, 1, 2)) > 0.02      # cols with any signal
        if bool(mx_r.any()) and bool(mx_c.any()):
            g = g[:, :, mx_r][:, :, :, mx_c]
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     dtype=g.dtype, device=g.device).view(1, 1, 3, 3)
    lap = float(F.conv2d(g, k, padding=1).var())
    con = float(g.var())
    return lap / max(con, 1e-9)


def _cg_gauss(img, sigma):
    """Separable gaussian blur on an IMAGE batch [B,H,W,C]."""
    import torch
    import torch.nn.functional as F
    if sigma <= 0:
        return img
    r = max(1, int(round(sigma * 3)))
    xs = torch.arange(-r, r + 1, dtype=img.dtype, device=img.device)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    v = img.permute(0, 3, 1, 2)                      # [B,C,H,W]
    c = v.shape[1]
    kh = k.view(1, 1, 1, -1).expand(c, 1, 1, k.numel())
    v = F.conv2d(F.pad(v, (r, r, 0, 0), mode="reflect"), kh, groups=c)
    kv = k.view(1, 1, -1, 1).expand(c, 1, k.numel(), 1)
    v = F.conv2d(F.pad(v, (0, 0, r, r), mode="reflect"), kv, groups=c)
    return v.permute(0, 2, 3, 1).clamp(0, 1)


def _cg_sigma_for(img, target_lap, max_sigma=1.6):
    """Smallest gaussian sigma bringing img texture energy down to target."""
    if target_lap <= 0:
        return 0.0
    # fine steps: the lap response to sigma is steep below ~0.7, so a coarse
    # grid overshoots badly (measured on synthetic texture: 0.3 -> 0.97x but
    # 0.6 -> 0.31x). 0.05 keeps the landing within a few percent.
    best_s, best_d = 0.0, abs(_cg_lap_var(img) - target_lap)
    s = 0.05
    while s <= max_sigma + 1e-9:
        d = abs(_cg_lap_var(_cg_gauss(img, s)) - target_lap)
        if d < best_d:
            best_d, best_s = d, s
        s += 0.05
    return best_s



def _cg_flatten(imgs, target, block=8, max_sigma=1.6):
    """Level a shot to a constant texture energy: blur only, per block.

    Sigma is searched on one representative frame per block (cheap) and then
    smoothed across blocks so the correction cannot pump frame to frame.
    Frames already at or below target are left untouched - this never
    sharpens, so it can only remove drift, never invent detail.
    """
    import torch
    n = imgs.shape[0]
    if n == 0 or target <= 0:
        return imgs, 0.0
    idx = list(range(0, n, block))
    sig = []
    for i in idx:
        f = imgs[i:i + 1]
        sig.append(_cg_sigma_for(f, target, max_sigma)
                   if _cg_lap_var(f) > target * 1.02 else 0.0)
    # smooth (moving average of 3) so sigma cannot jump between blocks
    sm = []
    for j in range(len(sig)):
        lo, hi = max(0, j - 1), min(len(sig), j + 2)
        sm.append(sum(sig[lo:hi]) / (hi - lo))
    if max(sm) <= 0.0:
        return imgs, 0.0
    out = imgs.clone()
    for j, i in enumerate(idx):
        s = sm[j]
        if s > 0.02:
            out[i:i + block] = _cg_gauss(imgs[i:i + block], s)
    return out, (sum(sm) / len(sm))


def _sampler_names():
    """From core, so the list cannot rot out of step with ComfyUI."""
    try:
        import comfy.samplers
        return list(comfy.samplers.KSampler.SAMPLERS)
    except Exception:
        return ["res_multistep", "euler", "dpmpp_2m"]


def _scheduler_names():
    try:
        import comfy.samplers
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:
        return ["simple", "normal", "beta"]


def _mmproj_postprocess(gg_loader, vsd, label):
    """Everything gguf_mmproj_loader does AFTER gguf_sd_loader.

    The explicit-file path skipped all of it, so the vision tower loaded under
    raw llama.cpp names (v.blk.*, mm.*) that nothing downstream consumes -
    user-reported as a matmul shape error mid-render, cured by switching back
    to (auto). Same file, different loader.

    Uses upstream's own map and helpers rather than reimplementing the rename,
    so if their key map changes this follows it. If their internals move, fail
    loudly here with an instruction rather than silently returning half a
    vision tower.
    """
    import torch
    try:
        sd_map_replace = gg_loader.sd_map_replace
        CLIP_VISION_SD_MAP = gg_loader.CLIP_VISION_SD_MAP
        dequantize_tensor = gg_loader.dequantize_tensor
        is_quantized = gg_loader.is_quantized
    except AttributeError as e:
        raise RuntimeError(
            f"This ComfyUI-GGUF build does not expose the vision key map "
            f"({e}), so an explicitly chosen mmproj cannot be renamed to the "
            f"layout the encoder expects. Set mmproj_name back to (auto) and "
            f"keep the mmproj beside the encoder with a matching name.")

    # 1. 4D patch_embd pair -> 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"),
                               dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"),
                               dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    # 2. the rename that makes the tower addressable at all
    vsd = sd_map_replace(vsd, CLIP_VISION_SD_MAP)

    # 3. split q/k/v -> fused qkv
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        for k, v in vsd.items():
            if any(x in k for x in ("attn_q", "attn_k", "attn_v")):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                attns.setdefault(k_attn, {})[k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v)
                              else torch.float16))
        for k, v in attns.items():
            sfx = k.split(".")[-1]
            vsd[k] = torch.cat([v[f"q.{sfx}"], v[f"k.{sfx}"], v[f"v.{sfx}"]],
                               dim=0)

    if not any(k.startswith("visual.") for k in vsd):
        raise RuntimeError(
            f"The mmproj '{label}' loaded but produced no visual.* tensors, so "
            f"it is not a vision sidecar for this encoder. Pick the -mmproj "
            f"file that belongs to the encoder you selected, or set "
            f"mmproj_name to (auto).")
    return vsd


class H3ClipLoaderAny:
    """One dropdown for text encoders, both formats: .safetensors through
    comfy core CLIPLoader, .gguf through ComfyUI-GGUF's CLIPLoaderGGUF
    (which auto-pairs a matching -mmproj sidecar for vision)."""

    @classmethod
    def INPUT_TYPES(cls):
        import os
        import folder_paths
        files = set(folder_paths.get_filename_list("text_encoders"))
        for d in folder_paths.get_folder_paths("text_encoders"):
            if not os.path.isdir(d):
                continue
            # RECURSIVE, same reason as the model loader above.
            for root, _dirs, fs in os.walk(d):
                for f in fs:
                    if f.lower().endswith(".gguf") and "mmproj" not in f.lower():
                        files.add(os.path.relpath(os.path.join(root, f), d))
        mm = ["(auto)"]
        for d in folder_paths.get_folder_paths("text_encoders"):
            if not os.path.isdir(d):
                continue
            for root, _dirs, fs in os.walk(d):
                for f in fs:
                    if f.lower().endswith(".gguf") and "mmproj" in f.lower():
                        mm.append(os.path.relpath(os.path.join(root, f), d))
        import nodes as core_nodes
        types = core_nodes.CLIPLoader.INPUT_TYPES()["required"]["type"]
        return {"required": {
            "clip_name": (sorted(files), {
                "tooltip": "safetensors or GGUF - routed automatically. GGUF "
                           "encoders auto-pair their -mmproj vision sidecar."}),
            "type": types,
        }, "optional": {
            # NEW WIDGETS APPEND LAST - inserting above shifts every saved
            # workflow's values by one slot.
            "mmproj_name": (mm, {
                "default": "(auto)",
                "tooltip": "Vision sidecar for a GGUF encoder. '(auto)' uses "
                           "ComfyUI-GGUF's pairing, which matches on FILENAME "
                           "inside the encoder's own folder - rename either "
                           "file, or split them across folders, and the match "
                           "fails. If auto finds nothing and exactly one "
                           "mmproj sits beside the encoder, that one is used "
                           "anyway. Pick a file here to override entirely; "
                           "then names and folders do not matter."}),
        }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "loaders/minimax"

    # llama.cpp/qwen2vl-era names -> the H3 encoder's exact visual.* layout
    # (established 2026-08-04 against the official int8 file). Ordered, and
    # chosen so no rule can re-hit another rule's output.
    _VISION_FIXES = [
        ("visual.merger.ln_q.", "visual.merger.norm."),
        ("attn_qkv.", "attn.qkv."),
        ("mlp.up_proj.", "mlp.linear_fc1."),
        ("mlp.down_proj.", "mlp.linear_fc2."),
        (".fc1.", ".linear_fc1."),
        (".fc2.", ".linear_fc2."),
        ("v.position_embd.weight", "visual.pos_embed.weight"),
    ]

    def load(self, clip_name, type, mmproj_name="(auto)"):
        import os
        import re
        import sys
        import nodes as core_nodes
        if not clip_name.lower().endswith(".gguf"):
            return core_nodes.CLIPLoader().load_clip(clip_name, type=type)

        gg_cls = core_nodes.NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")
        if gg_cls is None:
            raise RuntimeError(
                "ComfyUI-GGUF not loaded - install/enable it and restart.")
        gg = sys.modules[gg_cls.__module__]
        # gguf_mmproj_loader lives in their loader module, which nodes.py
        # does not re-export - resolve it from where gguf_clip_loader is
        # actually defined.
        gg_loader = sys.modules[gg.gguf_clip_loader.__module__]

        import folder_paths
        import comfy.sd
        import comfy.model_management
        clip_path = folder_paths.get_full_path("clip", clip_name)

        # --- text side: their mapper, then truncate to the official H3
        # shape (Qwen3-VL-32B cut to 50 layers; no final norm, no lm_head).
        sd = gg.gguf_clip_loader(clip_path)
        drop = re.compile(r"model\.layers\.(5[0-9]|6[0-9])\.")
        sd = {k: v for k, v in sd.items()
              if not drop.match(k) and k not in ("model.norm.weight",
                                                 "lm_head.weight")}

        # --- vision side: their sidecar loader, then correct the names to
        # H3's layout (their map is qwen2vl-era: wrong merger keys, missing
        # deepstack and qkv rules).
        # Upstream pairs the sidecar by FILENAME inside the encoder's own
        # folder (gguf_mmproj_loader: strip quant suffix, substring match),
        # so a rename on either file - or splitting them across folders -
        # breaks the pair. Upstream then logs an error and returns {}, which
        # renders as "the model ignores my reference image". Three ways out,
        # in order, and a hard failure rather than a silent one.
        vsd = None
        if mmproj_name and mmproj_name != "(auto)":
            mm_path = folder_paths.get_full_path("text_encoders", mmproj_name)
            if not mm_path:
                raise RuntimeError(
                    f"mmproj_name '{mmproj_name}' is not in the "
                    f"text_encoders folder any more.")
            vsd, _ = gg_loader.gguf_sd_loader(mm_path, is_text_model=True)
            vsd = _mmproj_postprocess(gg_loader, vsd, mmproj_name)
            print(f"[H3ClipLoader] vision sidecar (explicit): {mmproj_name}",
                  flush=True)
        else:
            vsd = gg_loader.gguf_mmproj_loader(clip_path)
            if not vsd:
                # upstream's name match failed; if exactly one mmproj sits
                # beside the encoder the intent is unambiguous, so use it
                _dir = os.path.dirname(clip_path)
                cands = [f for f in os.listdir(_dir)
                         if f.lower().endswith(".gguf")
                         and "mmproj" in f.lower()]
                if len(cands) == 1:
                    vsd, _ = gg_loader.gguf_sd_loader(
                        os.path.join(_dir, cands[0]), is_text_model=True)
                    print(f"[H3ClipLoader] filename pairing failed, but "
                          f"'{cands[0]}' is the only mmproj beside the "
                          f"encoder - using it. Set mmproj_name to silence "
                          f"this.", flush=True)
        if not vsd:
            raise RuntimeError(
                f"No vision sidecar (-mmproj) resolved for '{clip_name}'. "
                f"The H3 encoder NEEDS its vision tower for image "
                f"references and shot chaining. Either keep the mmproj file "
                f"beside the encoder with a matching name, or just pick it "
                f"explicitly in this node's mmproj_name widget - with that "
                f"set, names and folders do not matter.")
        # merger mlp indices -> linear_fc1/2 by ascending index
        idxs = sorted({m.group(1) for k in vsd
                       for m in [re.match(r"visual\.merger\.mlp\.(\d+)\.", k)]
                       if m})
        # deepstack mergers: llama.cpp indexes them by the vision layer they
        # tap (8/16/24), the H3 encoder by list position (0/1/2) - remap
        # ascending, and sort NUMERICALLY (lexically 16 < 8).
        ds = sorted({int(m.group(1)) for k in vsd
                     for m in [re.match(r"v\.deepstack\.(\d+)\.", k)]
                     if m})
        fixed = {}
        for k, v in vsd.items():
            for i, name in zip(idxs, ("linear_fc1", "linear_fc2")):
                k = k.replace(f"visual.merger.mlp.{i}.",
                              f"visual.merger.{name}.")
            for pos, layer in enumerate(ds):
                k = k.replace(f"v.deepstack.{layer}.",
                              f"visual.deepstack_merger_list.{pos}.")
            for a, b in self._VISION_FIXES:
                k = k.replace(a, b)
            fixed[k] = v
        sd.update(fixed)

        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=getattr(comfy.sd.CLIPType, type.upper(),
                              comfy.sd.CLIPType.STABLE_DIFFUSION),
            state_dicts=[sd],
            model_options={
                "custom_operations": gg.GGMLOps,
                "initial_device":
                    comfy.model_management.text_encoder_offload_device(),
            },
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = gg.GGUFModelPatcher.clone(clip.patcher)
        return (clip,)


class H3AudioTrimStart:
    """Trim N seconds off the FRONT of an audio clip. Exists so the multishot
    master can drop each chained shot's duplicated first frame (1/24s) from
    video AND audio together, keeping lip sync exact across seams."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "seconds": ("FLOAT", {"default": 0.04167, "min": 0.0, "max": 10.0,
                                  "step": 0.00001}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "trim"
    CATEGORY = "audio"

    def trim(self, audio, seconds):
        sr = audio["sample_rate"]
        wav = audio["waveform"]
        n = int(round(seconds * sr))
        return ({"sample_rate": sr, "waveform": wav[..., n:]},)


class H3MultishotSampler:
    """The whole multishot pipeline in one node: parse script, loop shots,
    chain each shot's last frame into the next shot's first_frame, seam-trim,
    and return master frames + master audio. shot_count is REAL here: N shots
    means N sampled shots, no wasted execution.

    JoyEcho architecture applied to H3: multishot complexity lives inside the
    node so the canvas stays legible."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "script": ("STRING", {
                "multiline": True, "dynamicPrompts": False,
                "default": "Shot 1 prompt goes here.\n---\n"
                           "Shot 2 prompt goes here.\n---\n"
                           "Shot 3 prompt goes here.",
                "tooltip": "One prompt per shot, separated by --- on its own "
                           "line. JSON {\"prompts\": [...]} also accepted."}),
            "shot_count": ("INT", {
                "default": 0, "min": 0, "max": 8,
                "tooltip": "The TOTAL number of shots - not shots per prompt. "
                           "Leave it at 0 and the script decides: one shot per "
                           "--- block, which is what you want for a written scene. "
                           "1-8 forces the total instead: extra blocks are dropped, "
                           "and if the script is short the last block repeats as a "
                           "continuation. Every shot renders."}),
            "width": ("INT", {"default": 768, "min": 32, "max": 4096,
                              "step": 32}),
            "height": ("INT", {"default": 1344, "min": 32, "max": 4096,
                               "step": 32}),
            "frames_per_shot": ("INT", {
                "default": 243, "min": 5, "max": 481, "step": 17,
                "tooltip": "Frames at 24fps on H3's 17k+5 grid (243 = ~10.1s;"
                           " 362 = trained max ~15.1s; beyond is untested)."}),
            "seed": ("INT", {"default": 0, "min": 0,
                             "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
            "seed_per_shot": ("BOOLEAN", {
                "default": True, "label_on": "vary per shot",
                "label_off": "same seed every shot",
                "tooltip": "Leave ON. Measured: varying the seed per shot holds the "
                           "face across the chain; using one seed for every shot "
                           "made BOTH the face and the voice drift. Identity "
                           "lives in the conditioning, not the seed."}),
        },
        "optional": {
            "start_image": ("IMAGE", {
                "tooltip": "Optional first frame (I2V). Shot 1 starts from this "
                           "image; later shots continue chaining from the "
                           "previous shot's last frame as usual. Leave "
                           "unconnected for pure text-to-video."}),
            "reference_images": ("IMAGE", {
                "tooltip": "Optional SUBJECT/CHARACTER reference images (batch "
                           "= multiple refs, e.g. via Batch Images), carried "
                           "into EVERY shot as <Picture 1>, <Picture 2>, ... "
                           "- distinct from start_image (which only seeds the "
                           "I2V chain frame). Bind them in each shot's prompt: "
                           "'She looks like the woman in <Picture 1>.' "
                           "Verified on the ref2va checkpoint."}),
            "voice_ref": ("AUDIO", {
                "tooltip": "Optional VOICE ANCHOR carried into EVERY shot as a "
                           "reference audio (<Audio 1>). Feed a clean solo "
                           "line of the character - e.g. a slice of stage A's "
                           "output - and the voice is PINNED across the chain "
                           "instead of re-performed from text (verified: "
                           "control drifted, voice-ref held). Bind it in each "
                           "shot's prompt: 'Her voice is the voice in "
                           "<Audio 1>.' Works with keyframe chaining via the "
                           "refs+keyframes merge patch. NOTE: verified on the "
                           "ref2va checkpoint; fl2va was not trained with "
                           "reference rows, so wire the ref2va model when "
                           "using this."}),
            "sampler_name": (_sampler_names(), {
                "default": "res_multistep",
                "tooltip": "Sampling algorithm. res_multistep is the default "
                           "and what every measurement in the docs used."}),
            "scheduler": (_scheduler_names(), {
                "default": "simple",
                "tooltip": "Sigma schedule. simple is the default and what "
                           "the docs measured."}),
            # NEW WIDGETS GO LAST, ALWAYS: saved canvases map widgets_values
            # by index, and a widget inserted mid-list silently shifts every
            # value after it on the next load (the v1.4 lesson).
            "sampler_override": ("STRING", {
                "forceInput": True,
                "tooltip": "Link a sampler NAME here (e.g. from H3 Studio "
                           "Controls) to drive this widget from one master "
                           "source. Overrides sampler_name when connected."}),
            "scheduler_override": ("STRING", {
                "forceInput": True,
                "tooltip": "Link a scheduler NAME here to single-source it. "
                           "Overrides scheduler when connected."}),
            "self_anchor_voice": ("BOOLEAN", {
                "default": False, "label_on": "anchor to shot 1's voice",
                "label_off": "off",
                "tooltip": "AUTOMATIC voice identity: after shot 1 renders, "
                           "its own audio becomes the reference (<Audio 1>) "
                           "for every later shot - the voice the model "
                           "actually performed is pinned, no file needed. "
                           "Write shot 1 so the character speaks a clean "
                           "solo line. An external voice_ref, if connected, "
                           "takes priority. Use with a ref2va checkpoint."}),
            "output_scale": ("FLOAT", {
                "default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05,
                "tooltip": "FINAL size multiplier, applied after decode. "
                           "No upscale model: a lanczos resize, 1.0 is off. "
                           "WITH a model: the model runs at its OWN fixed "
                           "factor (usually 4x) and this brings the result to "
                           "source x this value, so 2.0 on a 4x model gives "
                           "2x, not 8x. "
                           "CAREFUL - 1.0 does NOT mean off once a model is "
                           "wired; it means do-not-correct, so you get the "
                           "full 4x. At 1344x768 that is 5376x3072: 94 MB a "
                           "frame, 22 GB a shot, and every shot stays in "
                           "system RAM until the master is joined. The console "
                           "prints the projected size when the model loads - "
                           "read it. "
                           "Adds resolution, not detail. Works with every "
                           "continuity mode; the bank still stores "
                           "base-resolution clips."}),
            "upscale_model": ("UPSCALE_MODEL", {
                "tooltip": "Optional. Wire ComfyUI's Load Upscale Model here "
                           "(ESRGAN and friends) to synthesise detail instead "
                           "of merely resizing. Applied per shot after decode, "
                           "at the model's own factor; if output_scale is also "
                           "set, the result is resized to land exactly there. "
                           "Slower than output_scale and it invents texture - "
                           "on a chain, judge it on the LAST shot, where any "
                           "texture ratchet is worst."}),
            "sigmas": ("SIGMAS", {
                "tooltip": "Optional custom sigma schedule, replacing "
                           "sampler/scheduler + steps entirely. Some turbo "
                           "LoRAs ship a schedule they need in order to work "
                           "at all. When this is connected the 'steps' and "
                           "'scheduler' widgets are IGNORED - the step count "
                           "becomes len(sigmas)-1 - and the console says so. "
                           "The two-pass upscale split is taken as a fraction "
                           "of the supplied schedule."}),
            "save_every_shot": ("BOOLEAN", {
                "default": False, "label_on": "write each shot as it decodes",
                "label_off": "off",
                "tooltip": "Write EVERY shot to output/video/H3_SHOTS/ the "
                           "moment it decodes, in addition to the master. "
                           "Insurance for long chains: everything that fails "
                           "after the last shot - a mux OOM, a full disk, a "
                           "cancelled tab - otherwise destroys the whole "
                           "render at once. Shots are written BEFORE the seam "
                           "trim, so consecutive files overlap by ~1s; the "
                           "master is still the clean join. Costs one file "
                           "write per shot."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "IMAGE", "AUDIO")
    RETURN_NAMES = ("master_frames", "master_audio", "shots_rendered",
                    "first_shot_frames", "first_shot_audio")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"

    def run(self, model, clip, video_vae, audio_vae, script, shot_count,
            width, height, frames_per_shot, seed, steps,
            seed_per_shot=False, start_image=None, reference_images=None,
            voice_ref=None, sampler_name="res_multistep", scheduler="simple",
            sampler_override=None, scheduler_override=None,
            self_anchor_voice=False, reference_image_size="match",
            preview_first_shot=False, chain_gain_control="off",
            save_every_shot=False, sigmas=None, output_scale=1.0,
            upscale_model=None):
        # two_pass_upscale is REMOVED as of 2.1.3. It spatially interpolated
        # the raw latent between passes, and H3's latent is not a spatially
        # smooth representation - interpolated values land off-manifold and
        # pass 2, running at low sigma, has no room to pull them back. Three
        # A/Bs at the previously documented recipe (14 steps, beta57) came
        # back as colour-noise mush against clean single-pass controls, with
        # shot 1 - which carries no pin at all - destroyed identically, so it
        # was never about the chaining. Upscaling now happens AFTER decode
        # (output_scale / upscale_model), where it cannot leave the manifold.
        # The branches below are kept only so the diff stays readable; they
        # are unreachable.
        two_pass_upscale = False
        upscale_factor, pass1_fraction, upscale_audio_denoise = 1.5, 0.4, 0.35
        if sampler_override and str(sampler_override).strip():
            sampler_name = str(sampler_override).strip()
        if scheduler_override and str(scheduler_override).strip():
            scheduler = str(scheduler_override).strip()
        import torch
        import node_helpers
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3
        from comfy_extras.nodes_audio import vae_decode_audio

        # --- voice anchor: encode ONCE, ride in every shot's conditioning ---
        voice_block = None
        if voice_ref is not None:
            wav = voice_ref["waveform"]
            if wav.ndim == 2:
                wav = wav.unsqueeze(0)
            wav = wav[:1]
            if wav.shape[1] == 1:              # mono crashes the packed layout
                wav = wav.repeat(1, 2, 1)
            elif wav.shape[1] > 2:
                wav = wav[:, :2]
            sr = int(voice_ref["sample_rate"])
            vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
            if sr != vae_sr:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, vae_sr)
                sr = vae_sr
            limit = 15 * sr                    # ref rows cost speed EVERY step
            if wav.shape[-1] > limit:
                wav = wav[..., :limit]
            vz = audio_vae.encode(wav.movedim(1, -1))     # [1, 32, 2, T]
            voice_block = {"kind": "audio", "ref_audio_t": vz.shape[-1],
                           "audio_latent": vz}
            print(f"[H3Multishot] voice anchor: {wav.shape[-1]/sr:.1f}s ref "
                  f"audio rides in every shot as <Audio 1>.", flush=True)

        # --- subject/character reference images (PR #10, @moonwhaler): encode
        # ONCE, ride in every shot as fixed <Picture 1..N> slots - stable
        # across the whole chain, same reasoning as the voice anchor. ---
        import math
        ref_image_items, ref_image_blocks = [], []
        if reference_images is not None:
            for i in range(reference_images.shape[0]):
                img = reference_images[i:i + 1]
                h, w = img.shape[1], img.shape[2]
                if reference_image_size == "match":
                    scale = min(1.0, math.sqrt((width * height) / (w * h)))
                else:
                    scale = min(1.0, mmh3.REF_IMAGE_SHORT_EDGE / min(w, h))
                tw = max(mmh3.CANVAS_MULTIPLE,
                         round(w * scale / mmh3.CANVAS_MULTIPLE) * mmh3.CANVAS_MULTIPLE)
                th = max(mmh3.CANVAS_MULTIPLE,
                         round(h * scale / mmh3.CANVAS_MULTIPLE) * mmh3.CANVAS_MULTIPLE)
                resized = mmh3._resize(img, tw, th, "disabled")
                z = video_vae.encode(resized)
                ref_image_items.append({"type": "image", "data": resized})
                ref_image_blocks.append({"kind": "image", "latent_h": th // 16,
                                         "latent_w": tw // 16, "latent": z})
            print(f"[H3Multishot] {len(ref_image_blocks)} reference image(s) "
                  f"ride in every shot as <Picture 1..{len(ref_image_blocks)}>.",
                  flush=True)

        shots = _parse_script(script)
        n = shot_count if shot_count > 0 else len(shots)
        if len(shots) > n:
            print(f"[H3Multishot] dropping {len(shots) - n} extra script "
                  f"prompt(s) (shot_count={n}).", flush=True)
            shots = shots[:n]
        while len(shots) < n:
            print(f"[H3Multishot] shot {len(shots) + 1} continues the last "
                  f"prompt (script had fewer prompts than shot_count).",
                  flush=True)
            shots.append(shots[-1])

        if sigmas is not None and len(sigmas) > 1:
            # a supplied schedule wins: some turbo LoRAs only converge on the
            # exact curve they shipped with, and silently re-deriving one from
            # steps+scheduler would make them look broken rather than misused
            steps = int(len(sigmas)) - 1
            print("[%s] custom sigmas supplied (%d steps): the steps and "
                  "scheduler widgets are ignored for this run."
                  % ("H3Multishot", steps), flush=True)
        else:
            sigmas = ncs.BasicScheduler().get_sigmas(model, scheduler,
                                                     steps, 1.0)[0]
        sampler = ncs.KSamplerSelect().get_sampler(sampler_name)[0]

        # --- two-pass upscale setup: low-res pass 1, exact full-res pass 2 ---
        tr = w1 = h1 = sig_hi = sig_lo = lat_th = lat_tw = None
        if two_pass_upscale:
            tr = _load_upscaler_utils()
            f = max(1.0, float(upscale_factor))
            w1 = max(32, int(round(width / f / 32)) * 32)
            h1 = max(32, int(round(height / f / 32)) * 32)
            k = int(round(steps * float(pass1_fraction)))
            k = max(1, min(steps - 1, k))
            sig_hi, sig_lo = sigmas[:k + 1], sigmas[k:]
            probe, _ = mmh3._empty_av_latent(width, height, frames_per_shot)
            pv = probe["samples"]
            pv = pv.unbind()[0] if getattr(pv, "is_nested", False) else pv
            lat_th, lat_tw = int(pv.shape[-2]), int(pv.shape[-1])
            del probe, pv
            print(f"[H3Multishot] two-pass: {w1}x{h1} for {k} steps -> "
                  f"latent x{width / w1:.2f} -> {width}x{height} for "
                  f"{steps - k} steps (audio_denoise "
                  f"{upscale_audio_denoise}).", flush=True)

        frames_parts, audio_parts = [], []
        sr = None
        prev_last = None
        # chain gain control state (see _cg_* helpers)
        _cg_ref = None        # shot 1 tail texture level = the house level
        _cg_anchor_lap = None # texture level of the anchor fed to THIS shot
        _cg_gain = None       # measured per-hop gain (output head / anchor)
        _cg_frozen = None     # frozen_anchor mode: shot 1 tail frame
        _CG_WIN = 24          # frames per measurement window
        if chain_gain_control != "off":
            print(f"[H3Multishot] chain gain control: {chain_gain_control}",
                  flush=True)
        if start_image is not None:
            # I2V: seed the chain so shot 1 uses the supplied frame as its
            # keyframe, exactly the way later shots use the previous shot's
            # last frame. No seam trim on shot 1 - that frame is wanted.
            prev_last = start_image[:1]
            print("[H3Multishot] I2V: shot 1 starts from the supplied image.",
                  flush=True)
        for si, prompt in enumerate(shots):
            print(f"[H3Multishot] shot {si + 1}/{n} "
                  f"({frames_per_shot}f @ {width}x{height})...", flush=True)
            if two_pass_upscale:
                latent, frame_count = mmh3._empty_av_latent(
                    w1, h1, frames_per_shot)
            else:
                latent, frame_count = mmh3._empty_av_latent(
                    width, height, frames_per_shot)
            images, keyframes, keyframes_hi = [], [], []
            if prev_last is not None:
                anchor = prev_last[:1]
                if (chain_gain_control == "frozen_anchor"
                        and _cg_frozen is not None):
                    anchor = _cg_frozen
                    print("[H3Multishot] chain: anchoring on shot 1 tail "
                          "(frozen).", flush=True)
                elif chain_gain_control == "anchor_level" and _cg_ref:
                    # the model returns ~gain x the anchor texture energy, so
                    # feed it an anchor at ref/gain and the shot lands at ref
                    # seed value only - replaced by the measured gain after
                    # the first hop. 1.20 is what this stack actually returns
                    # (measured 1.197 three hops running with a frozen anchor,
                    # and 1.18 adaptively); the old 1.35 guess made shot 2
                    # land ~6% under the house level before converging.
                    g = _cg_gain if _cg_gain and _cg_gain > 1.0 else 1.20
                    target = _cg_ref / g
                    cur = _cg_lap_var(anchor)
                    if cur > target * 1.02:
                        sig = _cg_sigma_for(anchor, target)
                        if sig > 0:
                            anchor = _cg_gauss(anchor, sig)
                            print(f"[H3Multishot] chain: anchor pre-compensated "
                                  f"sigma {sig:.2f} ({cur:.5f} -> "
                                  f"{_cg_lap_var(anchor):.5f}, target "
                                  f"{target:.5f}, gain est {g:.2f})", flush=True)
                _cg_anchor_lap = _cg_lap_var(anchor)
                img = mmh3._resize(anchor, width, height, "disabled")
                images.append(img)
                if two_pass_upscale:
                    # pass-1 cond wants the keyframe latent on the LOW grid;
                    # pass-2 cond re-encodes the same frame at full res (true
                    # detail, not an interpolated latent). Vision tokens use
                    # the full-res image either way.
                    img_lo = mmh3._resize(anchor, w1, h1, "disabled")
                    keyframes.append({"resolved_frame_index": 0,
                                      "image": img_lo})
                    keyframes_hi.append({"resolved_frame_index": 0,
                                         "image": img})
                else:
                    keyframes.append({"resolved_frame_index": 0, "image": img})
            if voice_block is not None or ref_image_blocks:
                # ref-items presentation, in fixed order: the subject/character
                # refs keep their <Picture 1..N> slots, the chain frame takes
                # the next <Picture> slot, the voice gets <Audio 1>. Payload-
                # side the keyframe latent and the ref rows COMPOSE via the
                # refs+keyframes merge patch (h3_avbank_probe) - without that
                # patch comfy's refs branch would discard the keyframe latent.
                items = list(ref_image_items)
                items.extend({"type": "image", "data": im} for im in images)
                if voice_block is not None:
                    items.append({"type": "audio"})
                # The tokenizer emits these as bare "<Picture k>: " / "<Audio j>: "
                # labels before the prompt. Until 2026-08-21 this sampler sent no
                # subject_definitions at all, so the model got labelled references
                # and was never told what they were - the exact gap _subject_defs
                # was written for, implemented on the memory sampler only.
                _p = _subject_defs(sum(1 for i in items if i["type"] == "image"),
                                   sum(1 for i in items if i["type"] == "audio"),
                                   0,
                                   # this sampler has no reference_subjects input,
                                   # so every portrait is <Subject 1>
                                   image_subjects=None,
                                   return_parts=True,
                                   n_chain=len(images))
                if isinstance(_p, tuple):
                    prompt = _compose_ref2va(_p[0], _p[1], _p[2], prompt)
                    if si == 1:
                        print("[H3Multishot] Ref2VA sections added for %d reference "
                              "item(s) (%d portrait, %d continuation frame(s))"
                              % (len(items), len(ref_image_items), len(images)),
                              flush=True)
                tokens = clip.tokenize(prompt, minimax_ref_items=items)
            else:
                import os as _os_ka
                if images and not _os_ka.environ.get("H3_NO_KF_ALIGN"):
                    # The documented I2VA first line (base guide 2.1): the
                    # tokenizer labels this frame "<Picture 1>: " but nothing
                    # told the model it is the target video's 0.00-second
                    # frame. Keyframe modes use this alignment line, not the
                    # ref2va sections.
                    prompt = ("For the target video, at 0.00 seconds into "
                              "the target video, <Picture 1> (from [Shot 1]) "
                              "is fully referenced.\n\n") + prompt
                tokens = clip.tokenize(prompt, images=images)
            cond_base = clip.encode_from_tokens_scheduled(tokens)
            cond = cond_base
            cond_hi = cond_base if two_pass_upscale else None
            if keyframes:
                for kf in keyframes:
                    kf["latent"] = video_vae.encode(kf.pop("image"))
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_keyframes": keyframes,
                    "minimax_frame_count": frame_count,
                })
            if keyframes_hi:
                for kf in keyframes_hi:
                    kf["latent"] = video_vae.encode(kf.pop("image"))
                cond_hi = node_helpers.conditioning_set_values(cond_hi, {
                    "minimax_keyframes": keyframes_hi,
                    "minimax_frame_count": frame_count,
                })
            refs = list(ref_image_blocks)
            if voice_block is not None:
                refs.append(voice_block)
            if refs:
                # image blocks before audio blocks, matching the item order
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_refs": refs,
                })
                if cond_hi is not None:
                    cond_hi = node_helpers.conditioning_set_values(cond_hi, {
                        "minimax_refs": refs,
                    })
            # --- free the text encoder before the DiT loads -----------------
            # The 32B VL encoder (~16.5GB even at Q4) and the H3 DiT (~25GB) do
            # not co-fit on a 32GB card: without this the DiT loads PARTIALLY and
            # streams ~19GB from system RAM every step (60min vs ~15min renders).
            # Conditioning is already computed above, so the encoder weights are
            # safe to evict here; they reload next shot (chained prompts need it).
            # MULTI-GPU (issue #8, @VladiCz): when the TE lives on a DIFFERENT
            # device than the DiT there is nothing to reclaim - evicting just
            # forces a full TE reload every shot (measured: a third of the
            # whole render). Skip the eviction entirely in that case.
            import comfy.model_management as _mm
            _te_dev = getattr(clip.patcher, "load_device", None)
            _dit_dev = getattr(model, "load_device", None)
            if (_te_dev is not None and _dit_dev is not None
                    and str(_te_dev) != str(_dit_dev)):
                if si == 0:
                    print(f"[H3Multishot] TE on {_te_dev}, DiT on {_dit_dev} "
                          f"- separate devices, TE stays resident (no "
                          f"per-shot reload).", flush=True)
                    # sweep stale models from earlier runs once - a remote TE
                    # loads nothing locally, so nothing evicts them otherwise
                    # (same spill as the memory sampler, measured 2026-08-21)
                    try:
                        _dev0 = _mm.get_torch_device()
                        _b40 = _mm.get_free_memory(_dev0) / (1024 ** 3)
                        try:
                            _mm.unload_all_models()
                        except Exception:
                            pass
                        _mm.free_memory(_mm.get_total_memory(_dev0) * 0.9, _dev0)
                        _mm.soft_empty_cache()
                        _af0 = _mm.get_free_memory(_dev0) / (1024 ** 3)
                        if _af0 - _b40 > 0.5:
                            print("[H3Multishot] cleared %.1f GB of leftovers "
                                  "from earlier runs before the first DiT "
                                  "load." % (_af0 - _b40), flush=True)
                    except Exception:
                        pass
            else:
                try:
                    clip.patcher.model.to(_mm.text_encoder_offload_device())
                except Exception as _e:
                    print(f"[H3Multishot] TE offload skipped: {_e}", flush=True)
                try:
                    _dev = _mm.get_torch_device()
                    _mm.free_memory(_mm.get_total_memory(_dev) * 0.9, _dev)
                    _mm.soft_empty_cache()
                    _free = _mm.get_free_memory(_dev) / (1024 ** 3)
                    print(f"[H3Multishot] TE evicted; {_free:.1f} GB free "
                          f"for the DiT", flush=True)
                except Exception as _e:
                    print(f"[H3Multishot] VRAM purge skipped: {_e}", flush=True)
            # ----------------------------------------------------------------
            guider = ncs.BasicGuider().get_guider(model, cond)[0]
            guider_hi = (ncs.BasicGuider().get_guider(model, cond_hi)[0]
                         if two_pass_upscale else None)
            shot_seed = (seed + si) if seed_per_shot else seed
            noise = ncs.RandomNoise().get_noise(shot_seed)[0]
            # payload signature: chained shots carry a keyframe; audio refs
            # (voice_ref or self-anchor from shot 2 on) ride every step -
            # same latent shape, materially bigger activation pool
            _auto_set_payload(
                "kf%da%d%s" % (
                    1 if (si > 0 or start_image is not None) else 0,
                    1 if (voice_ref is not None
                          or (self_anchor_voice and si > 0)) else 0,
                    "2p" if two_pass_upscale else ""))
            _mb = _auto_measure_begin()
            try:
                if two_pass_upscale:
                    out1, _d1 = ncs.SamplerCustomAdvanced().sample(
                        noise, guider, sampler, sig_hi, latent)
                    up = _upscale_av_exact(tr, out1, lat_th, lat_tw)
                    s = max(0.0, min(1.0, float(upscale_audio_denoise)))
                    members, was_nested = tr.extract_tensor(up["samples"])
                    if was_nested and len(members) >= 2:
                        if s <= 0.0:
                            ridx, rstr = (0,), None
                        elif s >= 1.0:
                            ridx, rstr = (0, 1), None
                        else:
                            ridx, rstr = (0, 1), {0: 1.0, 1: s}
                    else:
                        ridx = rstr = None
                    noise2 = ncs.RandomNoise().get_noise(shot_seed + 977)[0]
                    up = tr.add_noise_nested_latent(
                        model, noise2, sig_lo, up,
                        renoise_indices=ridx, noise_strengths=rstr)
                    up = tr.finalize_latent_for_handoff(up)
                    out, _denoised = ncs.SamplerCustomAdvanced().sample(
                        ncs.DisableNoise().get_noise()[0], guider_hi,
                        sampler, sig_lo, up)
                else:
                    out, _denoised = ncs.SamplerCustomAdvanced().sample(
                        noise, guider, sampler, sigmas, latent)
            finally:
                # record even on interrupt/OOM: the peak up to that moment is
                # a valid LOWER bound on the pool, and the cache only grows -
                # an aborted thrashing run should still teach the next one
                _auto_measure_end(_mb, model, steps=steps)

            lat = out["samples"]
            if getattr(lat, "is_nested", False):
                lat = lat.unbind()[0]        # AV pair: [0]=video, [-1]=audio
            imgs = video_vae.decode(lat)
            if imgs.ndim == 5:
                imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2],
                                    imgs.shape[-1])
            aud = vae_decode_audio(audio_vae, out)
            sr = aud["sample_rate"]
            wav = aud["waveform"]

            # --- chain gain control: measure this hop, then level it ---
            if chain_gain_control != "off":
                _w = min(_CG_WIN, imgs.shape[0])
                head_lap = _cg_lap_var(imgs[:_w])
                if si == 0:
                    # shot 1 opens with an exposure fade-in (measured luma
                    # 0.227 -> 0.468 over ~1.7s), so the house level comes
                    # from the SETTLED tail, never the opening frames
                    _cg_ref = _cg_lap_var(imgs[-_w:])
                    _cg_frozen = imgs[-1:].clone()
                    print(f"[H3Multishot] chain: house texture level "
                          f"{_cg_ref:.5f} (shot 1 tail)", flush=True)
                    if chain_gain_control in ("flatten", "flatten_pin", "refresh_pin"):
                        # level shot 1 too, but only its post-fade portion:
                        # frames under the target are left alone by _cg_flatten
                        imgs, _s = _cg_flatten(imgs, _cg_ref)
                        if _s > 0:
                            print(f"[H3Multishot] chain: shot 1 levelled "
                                  f"(mean sigma {_s:.2f})", flush=True)
                        _cg_frozen = imgs[-1:].clone()
                else:
                    if _cg_anchor_lap:
                        hop = head_lap / max(_cg_anchor_lap, 1e-9)
                        _cg_gain = hop if _cg_gain is None else (
                            0.5 * _cg_gain + 0.5 * hop)
                        print(f"[H3Multishot] chain: shot {si + 1} hop gain "
                              f"{hop:.3f} (head {head_lap:.5f} / anchor "
                              f"{_cg_anchor_lap:.5f}); vs house "
                              f"{head_lap / max(_cg_ref, 1e-9) - 1.0:+.1%}",
                              flush=True)
                    if chain_gain_control in ("flatten", "flatten_pin", "refresh_pin") and _cg_ref:
                        imgs, _s = _cg_flatten(imgs, _cg_ref)
                        if _s > 0:
                            print(f"[H3Multishot] chain: shot levelled to "
                                  f"house (mean sigma {_s:.2f}; head "
                                  f"{_cg_lap_var(imgs[:_w]):.5f} tail "
                                  f"{_cg_lap_var(imgs[-_w:]):.5f} vs house "
                                  f"{_cg_ref:.5f})", flush=True)
                    if chain_gain_control == "match_output" and _cg_ref:
                        if head_lap > _cg_ref * 1.05:
                            sig = _cg_sigma_for(imgs[:_w], _cg_ref)
                            if sig > 0:
                                imgs = _cg_gauss(imgs, sig)
                                print(f"[H3Multishot] chain: shot matched to "
                                      f"house level, sigma {sig:.2f} (head "
                                      f"{head_lap:.5f} -> "
                                      f"{_cg_lap_var(imgs[:_w]):.5f})",
                                      flush=True)

            prev_last = imgs[-1:].clone()
            if si == 0 and self_anchor_voice and voice_block is None:
                # THE self-anchor: shot 1's own rendered voice becomes the
                # reference for every later shot. The decoded audio is
                # already at the VAE's rate and stereo, so no guard needed -
                # just trim (ref rows cost speed every step) and encode.
                aw = wav[:1] if wav.ndim == 3 else wav.unsqueeze(0)[:1]
                limit = 15 * sr
                if aw.shape[-1] > limit:
                    aw = aw[..., :limit]
                vz = audio_vae.encode(aw.movedim(1, -1))
                voice_block = {"kind": "audio", "ref_audio_t": vz.shape[-1],
                               "audio_latent": vz}
                print(f"[H3Multishot] self-anchor: shot 1's voice "
                      f"({aw.shape[-1]/sr:.1f}s) is now <Audio 1> for the "
                      f"remaining {n - 1} shot(s).", flush=True)
            if si == 0:
                first_frames = imgs.detach().cpu()
                _fw = wav if wav.ndim == 3 else wav.unsqueeze(0)
                first_audio = {"waveform": _fw.detach().cpu(),
                               "sample_rate": sr}
                if preview_first_shot:
                    # write shot 1 NOW - minutes before the chain finishes -
                    # so a bad take can be cancelled instead of waited out
                    _write_shot_mp4(imgs, wav, sr,
                                    "video/H3_FIRSTSHOT/firstshot",
                                    "FIRST-SHOT PREVIEW saved", "H3Multishot")
            if save_every_shot:
                # before the seam trim: this file is the shot as rendered, so
                # a chain that dies at the mux can still be joined by hand
                _write_shot_mp4(imgs, wav, sr, "video/H3_SHOTS/shot",
                                f"shot {si + 1}/{n} saved", "H3Multishot")
            if si > 0:
                imgs = imgs[1:]                       # duplicated seam frame
                trim = int(round(sr / 24.0))          # matching 1/24s audio
                wav = _smart_head_trim(wav, sr, trim)
            # fp16: the encoder quantises to uint8 downstream, and this
            # timeline is what exhausted host RAM at 6 shots x 243f.
            frames_parts.append(imgs.cpu().half())
            audio_parts.append(wav.cpu())

        # Assemble in place. torch.cat allocated a second full timeline
        # while the first was still alive - 2x peak, and a 33.8 GB contiguous
        # request that a 64 GB box cannot satisfy. Same bytes, one buffer.
        _n = sum(int(_p.shape[0]) for _p in frames_parts)
        master = torch.empty((_n,) + tuple(frames_parts[0].shape[1:]),
                             dtype=frames_parts[0].dtype)
        _o = 0
        for _i in range(len(frames_parts)):
            _p = frames_parts[_i]
            master[_o:_o + _p.shape[0]].copy_(_p)
            _o += int(_p.shape[0])
            frames_parts[_i] = None
            del _p
        waveform = _xfade_audio(audio_parts, sr)
        print(f"[H3Multishot] done: {n} shots, {master.shape[0]} frames "
              f"(~{master.shape[0] / 24.0:.1f}s).", flush=True)
        return (master, {"waveform": waveform, "sample_rate": sr}, n,
                first_frames, first_audio)






# ---------------------------------------------------------------------------
# Colour levelling (seamless chains)
#
# MVGD colour-statistics transfer to a FIXED house reference (shot 1's settled
# tail). Chained matching (each shot to its predecessor) re-accumulates drift;
# a fixed reference cannot. Stats are computed in linearised RGB (x**2.2), the
# transform is applied per 8-frame block with EMA smoothing so the correction
# cannot pump frame to frame.
# ---------------------------------------------------------------------------

def _cc_stats(imgs):
    """(mu[3], cov[3,3]) of an IMAGE batch [B,H,W,C] in linear RGB."""
    import torch
    x = imgs.reshape(-1, imgs.shape[-1]).clamp(0, 1) ** 2.2
    # MEDIAN, not mean: the matching statistic must track scene LIGHTING,
    # not the subject. A close-up puts a big lit face where a wide had
    # room, so the mean moves for compositional reasons and mean-matching
    # then "corrects" a change that was never an exposure error
    # (measured: face region +36% between a wide and a close-up while the
    # wall strip held within 2%). The median is dominated by the bulk of
    # the scene and barely moves under reframing.
    mu = x.median(dim=0).values if x.shape[0] > 1 else x.mean(dim=0)
    d = x - mu
    cov = (d.T @ d) / max(x.shape[0] - 1, 1)
    return mu, cov


def _cc_sqrtm(m):
    """Symmetric PSD matrix square root via eigendecomposition."""
    import torch
    vals, vecs = torch.linalg.eigh(m.double())
    vals = vals.clamp_min(1e-12)
    return (vecs @ torch.diag(vals.sqrt()) @ vecs.T)


def _cc_mvgd_T(cov_src, cov_dst):
    """MVGD transfer matrix: src distribution -> dst distribution."""
    import torch
    s_half = _cc_sqrtm(cov_src.double())
    s_ihalf = torch.linalg.inv(s_half)
    inner = _cc_sqrtm(s_half @ cov_dst.double() @ s_half)
    return (s_ihalf @ inner @ s_ihalf)


def _cc_apply_perframe(imgs, target_mu, strength=1.0, smooth=13):
    """Level EVERY FRAME to one fixed colour target.

    A per-shot gain only works if the shot is internally uniform. Under the
    FFLF chain it is not: a shot can render with a warm head and a cooler
    body, and matching medians then multiplies the whole shot by one gain -
    amplifying the head (measured: shot 3 median pulled 2.29 -> 3.11, which
    threw its opening frames to 4.60 and produced a 40% step at the join).

    Correcting each frame independently removes both the between-shot offset
    and the within-shot drift. The gain is smoothed over ~0.5s so a real
    lighting event still reads as an event instead of being tracked away.
    """
    import torch
    if strength <= 0:
        return imgs
    n = imgs.shape[0]
    lin = imgs.clamp(0, 1).double() ** 2.2
    per = lin.reshape(n, -1, imgs.shape[-1]).median(dim=1).values
    gain = (target_mu.double().view(1, 3)
            / per.clamp_min(1e-6)).clamp(0.7, 1.4)
    if not bool(torch.isfinite(gain).all()):
        return imgs
    k = int(smooth) | 1
    if n > k > 1:
        g = torch.nn.functional.pad(gain.T.unsqueeze(0), (k // 2, k // 2),
                                    mode="replicate")
        gain = torch.nn.functional.avg_pool1d(g, k, stride=1).squeeze(0).T[:n]
    out = imgs.clone()
    for i in range(0, n, 8):
        seg = imgs[i:i + 8]
        gs = gain[i:i + 8].view(-1, 1, 1, imgs.shape[-1])
        m = (((seg.clamp(0, 1).double() ** 2.2) * gs).clamp(0, 1)
             ** (1 / 2.2)).to(seg.dtype)
        out[i:i + 8] = seg + strength * (m - seg)
    return out


def _cc_apply(imgs, house_mu, house_cov, strength=1.0, block=8):
    """Level an IMAGE batch to the house colour statistics.

    ONE GLOBAL correction for the whole shot - stats pooled over every frame,
    never per-block. A per-block transfer forces each 8-frame block onto the
    full house distribution, which destroys legitimate local variation
    (render-verified failure: a deep-shadow block got remapped to mid-grey
    and the whole shot posterised into magenta/green).

    The correction is a per-channel GAIN in linear light, not an affine
    transfer: exposure and tint drift between chained shots is multiplicative
    (a gain), and an additive offset in linear space lifts black levels by
    the full drift amount - a 0.02-linear offset turns true black into ~0.17
    gamma, ruinous in dark scenes. Gains map black to exactly black, are
    monotone per channel (posterisation impossible), and correct the axis
    that actually drifts at joins (luma/tint). Covariance matching is
    deliberately dropped; `house_cov` and `block` are kept for signature
    compatibility (`block` only sets the application chunk size).

    Guard rail: gains are clamped to [0.7, 1.4] - a correction beyond that
    means the shot genuinely changed (a light went out, a door opened) and
    colour transfer must not fight real scene changes.
    Measured: colour is CONSTANT within a shot (flat to three decimals
    across a whole shot) and steps hard at the boundary. So one gain per
    shot is the right shape of correction - what was wrong before was the
    TARGET (a rolling per-shot house). Drive every shot to one scene-wide
    reference instead: see color_level="scene".
    """
    import torch
    if strength <= 0:
        return imgs
    mu_s, _cov_s = _cc_stats(imgs)
    gain = (house_mu.double() / mu_s.double().clamp_min(1e-6)).clamp(0.7, 1.4)
    if not bool(torch.isfinite(gain).all()):
        return imgs
    out = imgs.clone()
    for i in range(0, imgs.shape[0], max(1, int(block))):
        seg = imgs[i:i + block]
        lin = seg.clamp(0, 1).double() ** 2.2
        matched = ((lin * gain).clamp(0, 1) ** (1 / 2.2)).to(seg.dtype)
        out[i:i + block] = seg + strength * (matched - seg)
    return out


def _parse_ref_groups(spec, n_image):
    """"3,3" -> [1,1,1,2,2,2]: which subject each reference picture belongs to.

    Empty/invalid returns None, meaning "every picture is <Subject 1>" - the
    behaviour every release before 2.2.4 had. Counts that do not add up to
    n_image are corrected rather than rejected: a trailing shortfall joins the
    last subject, an overshoot is truncated. A user who mutes a reference image
    should not get a hard error out of a text field.
    """
    if not spec or not str(spec).strip() or n_image <= 0:
        return None
    try:
        counts = [int(x) for x in re.split(r"[,\s]+", str(spec).strip()) if x]
    except ValueError:
        print("[H3] reference_subjects %r is not a comma list of counts - "
              "treating every reference as one subject." % spec, flush=True)
        return None
    counts = [c for c in counts if c > 0]
    if len(counts) < 2:
        return None
    out = []
    for si, c in enumerate(counts, 1):
        out.extend([si] * c)
    if len(out) < n_image:
        out.extend([out[-1]] * (n_image - len(out)))
    return out[:n_image]


def _ref2va_task_types(n_image, n_audio, n_video):
    """The summary's square-bracket task-type prefix, per the Ref2VA guide.

    Chosen from the role each reference actually plays here: pictures guide
    generation, our <Video> is an earlier moment of the same take, and our
    <Audio> is referenced for timbre rather than copied.
    """
    t = []
    if n_image:
        t.append("reference generation")
    if n_video:
        t.append("video continuation")
    if n_audio:
        t.append("audio reference")
    return "[%s]" % " + ".join(t or ["reference generation"])


def _ref2va_summary(n_image, n_audio, n_video, n_sub, n_chain=0):
    """One short paragraph naming the subjects and what each reference gives."""
    s = [_ref2va_task_types(n_image, n_audio, n_video)]
    who = ("<Subject 1>" if n_sub <= 1 else
           ", ".join("<Subject %d>" % i for i in range(1, n_sub + 1)))
    s.append("The target video is one continuous shot of %s." % who)
    n_portrait = max(0, n_image - int(n_chain or 0))
    if n_portrait:
        s.append("%s %s the appearance of %s." %
                 (", ".join("<Picture %d>" % k for k in range(1, n_portrait + 1)),
                  "supplies" if n_portrait == 1 else "supply", who))
    if n_chain:
        s.append("%s %s the frame this shot continues from." %
                 (", ".join("<Picture %d>" % k
                            for k in range(n_portrait + 1, n_image + 1)),
                  "is" if n_chain == 1 else "are"))
    if n_video:
        s.append("%s %s the place, framing and light this shot continues from." %
                 (", ".join("<Video %d>" % k for k in range(1, n_video + 1)),
                  "carries" if n_video == 1 else "carry"))
    if n_audio:
        s.append("%s supplies <Subject 1>'s voice timbre." %
                 ", ".join("<Audio %d>" % j for j in range(1, n_audio + 1)))
    return "summary:\n" + " ".join(s)


def _compose_ref2va(defs, summary, retention, body):
    """The six official sections in the guide's order, body in the middle.

    Order is load-bearing: subject_definitions has to establish <Subject N>
    BEFORE detailed_description refers to it. Until 2026-08-21 the two label
    sections were concatenated onto the END of the body instead, so the model
    read the prose first and the definitions afterwards.

    overall_soundscape is deliberately absent: the writer folds ambience into
    the paragraph and is forbidden the word "music", so there is no honest
    source for that section yet. non_diegetic_music: N/A is the guide's own
    value for "no score", and nothing in the conditioning said so before.
    """
    return "\n\n".join([defs, summary, retention,
                        "detailed_description:\n" + body.strip(),
                        "non_diegetic_music: N/A"])


def _subject_defs(n_image, n_audio, n_video, speaker="the person",
                  image_subjects=None, return_parts=False, n_chain=0):
    """Official H3 ref2va subject_definitions + retention_analysis block.

    The tokenizer emits reference items as bare "<Picture k>: ",
    "<Audio j>: " and "<Video k>: " labels BEFORE the prompt text
    (comfy/text_encoders/minimax.py). Without a subject_definitions section
    the model gets labelled references and is never told what they are or
    what to keep - which is why identity, room, colour and especially VOICE
    drift between chained shots. Syntax follows the MiniMax-H3 model card's
    Ref2VA case; retention keywords are fully_preserved / partially_copy /
    reference.

    image_subjects (2.2.4) is an optional list, one entry per reference
    picture, giving the subject number that picture belongs to. Until 2.2.4
    every picture was declared a photograph of <Subject 1> unconditionally, so
    references for two different people told the model all of them showed the
    same individual and it rendered the average - reported on Civitai as
    "the video doesn't contain anything that resembles the reference people".
    None keeps the old single-subject behaviour exactly.
    """
    import os as _os
    if _os.environ.get("H3_NO_SUBJECT_DEFS"):   # A/B switch for testing
        return ""
    if not (n_image or n_audio or n_video):
        return ""
    subs = list(image_subjects or [])
    n_sub = max(subs) if subs else 1
    if n_sub <= 1:
        d = ["subject_definitions:",
             "<Subject 1> is %s speaking in this scene." % speaker]
    else:
        # Only <Subject 1> is described as speaking; H3's voice conditioning is
        # single-speaker and naming several speakers competes for the audio lane.
        d = ["subject_definitions:",
             "<Subject 1> is %s speaking in this scene." % speaker]
        for s in range(2, n_sub + 1):
            d.append("<Subject %d> is a different individual who also appears "
                     "in this scene." % s)
    # Do NOT enumerate accessories here. This block is unconditional text on a
    # BasicGuider path - cfg 1.0, no negative branch - so anything named is
    # ADDED and can never be subtracted by the user's prompt. Naming "glasses"
    # made every ref2va render force thick frames onto the subject no matter
    # what the prompt asked for, and "remove the glasses" only put the word in
    # the conditioning a second time (reported on Civitai 2026-08-13).
    # Face, skin, hair and wardrobe are identity; eyewear, hats and jewellery
    # are wardrobe choices that belong to the prompt.
    r = ["retention_analysis:",
         "<Subject 1> (appears in [Shot 1]): fully_preserved - <Subject 1> "
         "retains the same face, skin and hair, and "
         "stays in the same room under the same lighting and colour "
         "temperature."]
    for s in range(2, n_sub + 1):
        # Affirmative only. The trailing "is never blended with <Subject 1>"
        # this line used to carry violated the rule stated in the comment
        # above: at cfg 1.0 the negation has no branch to land in, so it just
        # names blending in the conditioning. The first clause already carries
        # the whole intent.
        r.append("<Subject %d> (appears in [Shot 1]): fully_preserved - "
                 "<Subject %d> retains their own distinct face, skin and hair." % (s, s))
    # The LAST n_chain pictures are continuation frames, not portraits. Declaring
    # a whole previous-shot frame as "a reference photograph of <Subject 1>" tells
    # the model to treat a scene as a face reference, so they get their own line.
    n_portrait = max(0, n_image - int(n_chain or 0))
    for k in range(1, n_image + 1):          # numeric order, portraits then chain
        if k > n_portrait:
            d.append("<Picture %d> is the first frame of [Shot 1], carried over "
                     "from the previous shot of this same continuous scene." % k)
            r.append("<Picture %d> ([Shot 1] continuation anchor): fully_preserved - "
                     "the shot continues from <Picture %d>, keeping its place, "
                     "framing, colour temperature and light." % (k, k))
            continue
        s = subs[k - 1] if k <= len(subs) else 1
        d.append("<Picture %d> is a reference photograph of <Subject %d>."
                 % (k, s))
    for k in range(1, n_video + 1):
        d.append("<Video %d> is a clip from an earlier moment of this same "
                 "continuous scene, showing <Subject 1> in the same place "
                 "under the same light." % k)
        r.append("<Video %d>: reference - the target video keeps the "
                 "framing, camera distance, room contents and colour "
                 "temperature of <Video %d>." % (k, k))
    for j in range(1, n_audio + 1):
        # Only claim it is a video's soundtrack when that <Video> label exists -
        # the guide forbids unresolved reference labels, and with no reference
        # video this used to point at a <Video 1> that was never defined.
        d.append(("<Audio %d> is the synchronized audio track of <Video %d>, "
                  "containing <Subject 1>'s speaking voice." % (j, j)) if j <= n_video
                 else ("<Audio %d> is a recording of <Subject 1>'s speaking voice."
                       % j))
        r.append("<Audio %d>: reference - the target audio references the "
                 "voice timbre in <Audio %d> so <Subject 1> speaks with the "
                 "same voice." % (j, j))
    if return_parts:
        return ("\n".join(d),
                _ref2va_summary(n_image, n_audio, n_video, n_sub, n_chain),
                "\n".join(r))
    return "\n".join(d) + "\n" + "\n".join(r)


def _vhs_glitch_frames(frames, seed, strength=1.0):
    """Diegetic VHS tracking glitch over a short frame run (join masking).

    Horizontal displacement bands, a slight chroma shift, dropout flecks and
    a noise veil, peaking mid-run and fading at both ends so the artifact
    reads as one tape hiccup rather than a processed boundary. Deterministic
    per seed. frames [N,H,W,C] in 0..1; returns a new tensor.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    out = frames.clone()
    N, H, W, _C = out.shape
    for i in range(N):
        amp = strength * (1.0 - abs(i - (N - 1) / 2.0) / ((N + 1) / 2.0))
        if amp <= 0:
            continue
        for _b in range(2 + int(torch.randint(0, 3, (1,), generator=g))):
            y0 = int(torch.randint(0, max(1, H - 24), (1,), generator=g))
            bh = int(torch.randint(4, 24, (1,), generator=g))
            dx = int(int(torch.randint(-40, 41, (1,), generator=g)) * amp)
            if dx:
                out[i, y0:y0 + bh] = torch.roll(out[i, y0:y0 + bh],
                                                shifts=dx, dims=1)
        dxc = int(6 * amp)
        if dxc:
            out[i, ..., 0] = torch.roll(out[i, ..., 0], dxc, dims=1)
        for _l in range(int(6 * amp)):
            y = int(torch.randint(0, H - 2, (1,), generator=g))
            x0 = int(torch.randint(0, W // 2, (1,), generator=g))
            ln = int(torch.randint(20, W // 2, (1,), generator=g))
            hot = float(torch.rand(1, generator=g)) > 0.5
            out[i, y:y + 2, x0:x0 + ln] = 0.9 if hot else 0.05
        out[i] = (out[i] + amp * 0.06 * torch.randn(
            out[i].shape, generator=g).to(out.device, out.dtype)).clamp(0, 1)
    return out


def _vhs_glitch_audio(wav, sr, at_start, seed, ms=90):
    """Tape head-switch audio hiccup: duck the signal and lay hiss over ~ms
    at the head (at_start=True) or tail of the waveform. Deterministic."""
    import torch
    g = torch.Generator().manual_seed(seed)
    n = min(int(sr * ms / 1000.0), wav.shape[-1])
    if n < 8:
        return wav
    out = wav.clone()
    seg = out[..., :n] if at_start else out[..., -n:]
    t = torch.linspace(0, 1, n)
    env = torch.sin(t * 3.14159265)          # fade the hiccup in and out
    hiss = 0.05 * torch.randn(seg.shape, generator=g).to(seg.device,
                                                         seg.dtype)
    seg = seg * (1.0 - 0.6 * env) + hiss * env
    if at_start:
        out[..., :n] = seg
    else:
        out[..., -n:] = seg
    return out


def _aud_env(x, win):
    """Mono 20ms-window RMS envelope of a waveform tensor [..., T]."""
    import torch
    x = x.reshape(-1, x.shape[-1]).float().mean(dim=0)
    m = (x.shape[-1] // win) * win
    if m < win:
        return torch.zeros(1)
    return x[:m].reshape(-1, win).pow(2).mean(dim=-1).sqrt()


def _jb_grid(n):
    """Largest valid H3 clip length <= n: frames must satisfy n % 17 == 5."""
    n = int(n)
    if n < 5:
        return 5
    while n % 17 != 5 and n > 5:
        n -= 1
    return max(5, n)


def _jb_centre_clip(imgs, want):
    """Centre clip of `want` frames (snapped to the 17k+5 grid).

    JoyEcho selects its slot around the CENTRE of the shot
    (_select_video_clip_around_frame, default mode "center"), not the tail.
    Returns (clip, start_index) so the audio window can be cut to match.
    """
    total = int(imgs.shape[0])
    n = _jb_grid(min(int(want), total))
    start = max(0, (total - n) // 2)
    return imgs[start:start + n], start


def _jb_audio_window(wav, sr, start_frame, num_frames, fps=24.0):
    """The audio under a clip's frame range, as an AUDIO dict."""
    import torch
    a = wav if wav.ndim == 3 else wav.unsqueeze(0)
    s = int(round(start_frame / fps * sr))
    e = int(round((start_frame + num_frames) / fps * sr))
    s = max(0, min(s, a.shape[-1]))
    e = max(s + 1, min(e, a.shape[-1]))
    return {"waveform": a[..., s:e].clone(), "sample_rate": int(sr)}


class _H3ChainBank:
    """Bounded frame bank: pinned earliest entries + recency tail.

    Mirrors the JoyEcho/LTX bank policy that keeps long chains from drifting:
    `frames()` always returns the first `num_fix` entries ever added, plus the
    most recent entries, capped at `max_size` total. Conditioning on a set that
    always contains the beginning of the episode is what breaks the
    shot-to-shot feedback path - each shot is no longer a pure function of the
    one before it.
    """

    def __init__(self, num_fix=1, max_size=3):
        self.num_fix = max(0, int(num_fix))
        self.max_size = max(1, int(max_size))
        self._entries = []

    def add(self, frame):
        self._entries.append(frame)
        # prune to what frames() can ever return, so a long chain does not
        # hold every decoded frame in memory for nothing
        keep_fixed = min(self.num_fix, self.max_size)
        keep_tail = self.max_size - keep_fixed
        if len(self._entries) > keep_fixed + keep_tail:
            head = self._entries[:keep_fixed]
            # keep_tail == 0 must yield NO tail: entries[len-0:] is the whole
            # list (the slice bug that unbounded JoyEcho's bank)
            tail = self._entries[-keep_tail:] if keep_tail > 0 else []
            self._entries = head + tail

    def frames(self):
        fixed = self._entries[:min(self.num_fix, self.max_size)]
        tail = self._entries[len(fixed):]
        keep = self.max_size - len(fixed)
        if keep <= 0:
            return list(fixed)
        # keep_tail == 0 must yield NO tail entries: tail[-0:] is the WHOLE
        # list, which is exactly the bug that let JoyEcho's bank grow unbounded
        return list(fixed) + (list(tail[-keep:]) if keep > 0 else [])

    def latest(self):
        return self._entries[-1] if self._entries else None

    def describe(self):
        f = min(self.num_fix, self.max_size, len(self._entries))
        return f"{len(self.frames())} slot(s) [{f} pinned + {len(self.frames()) - f} recent]"


class H3MultishotMemorySampler:
    """Multishot with a MEMORY BANK - a structural port of JoyEcho multishot.

    There is no keyframe here. Shots are not continued pixel-wise from their
    predecessor; each is generated fresh and held together by a bank of past
    shots injected as reference conditioning. That is JoyEcho's architecture,
    and it is why JoyEcho chains do not accumulate texture drift: a shot is
    never a pure function of the shot before it, because the bank always
    contains the beginning of the episode.

    Bank slot = a short video clip from the MIDDLE of a shot + the audio under
    it, injected as an H3 `video_audio` reference. The first `bank_pinned`
    slots are never evicted; the rest is a bounded recency window.

    REQUIRES a ref2va checkpoint - reference rows are what this node is built
    on, and fl2va was not trained with them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "script": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "One prompt per shot: JSON "
                                             "{\"prompts\": [...]} or plain "
                                             "blocks separated by --- lines."}),
            "shot_count": ("INT", {"default": 0, "min": 0, "max": 64,
                                   "tooltip": "The TOTAL number of shots - not shots per "
                                   "prompt. 0 = one shot per --- block in the script "
                                   "(leave it here). Above 0 forces the total: extra "
                                   "blocks are dropped, a short script repeats its last "
                                   "block."}),
            "width": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32}),
            "frames_per_shot": ("INT", {"default": 243, "min": 5, "max": 1450,
                                        "step": 17,
                                        "tooltip": "Trained range is ~124-362;"
                                        " longer single shots are ladder "
                                        "territory (RoPE extrapolation) - "
                                        "the audio-spine pass runs 719f "
                                        "low-res through exactly this."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
            "seed_per_shot": ("BOOLEAN", {"default": True,
                                          "label_on": "vary per shot",
                                          "label_off": "same seed every shot"}),
            "memory_frames": ("INT", {
                "default": 0, "min": 0, "max": 3,
                "tooltip": "RECENCY slots: how many of the most RECENT shots "
                           "stay in the bank, on top of the pinned one. Total "
                           "bank = pinned + recent, capped at 3 by H3's "
                           "reference limit.\n"
                           "DEFAULT 0, deliberately. The recent slots hand "
                           "each shot's ACCRETED output forward as reference "
                           "images, on top of the latent pin, so invented "
                           "detail compounds. Measured over ten shots at "
                           "960x544, moving 2 -> 0: texture growth 1.055 -> "
                           "1.022 per hop, chroma 1.086 -> 1.039, framing "
                           "correlation at shot 10 0.976 -> 0.995, and the "
                           "drift stops ACCELERATING - 4.2%->6.7% per hop "
                           "became 2.3%->2.0%. At 0 the only reference is "
                           "shot 1, which nothing has been added to yet.\n"
                           "Identity and framing held on a static scene. If a "
                           "busy scene loses motion continuity between shots, "
                           "raise it to 1."}),
            "anchor_frames": ("INT", {
                "default": 1, "min": 0, "max": 9,
                "tooltip": "Identity reference images taken from start_image, "
                           "used on EVERY shot (JoyEcho seeds identity on shot "
                           "1 and lets the bank carry it after that; keep this "
                           "at 0 or 1)."}),
        }, "optional": {
            "start_image": ("IMAGE", {
                "tooltip": "Optional identity reference image. NOT a first "
                           "frame - this node has no keyframe."}),
            "keyframe_images": ("IMAGE", {
                "tooltip": "flf_chain only: N+1 boundary stills for N shots, "
                           "in order. Shot i is generated between image i "
                           "and image i+1. Best source is a single long "
                           "low-res take of the whole scene - its frames at "
                           "the boundary times are already colour-matched, "
                           "identity-matched and correctly posed, so every "
                           "join inherits one consistent look."}),
            "guide_audio": ("AUDIO", {
                "tooltip": "AUDIO SPINE: a continuous audio track for the "
                           "WHOLE take - a voice recording, a song, or a "
                           "low-res long pass. Each shot's audio stream is "
                           "locked to its time-slice of this track at every "
                           "sampling step; the video follows the locked "
                           "audio (lips included) via the model's own "
                           "audio-video attention. Works with EVERY "
                           "continuity mode - the per-shot stride accounts "
                           "for each mode's seam trim (render-verified on "
                           "context_pin). Any sample rate: the track is "
                           "resampled to the audio VAE's rate and mono is "
                           "upmixed. This is the locked-audio music-video "
                           "path."}),
            "sampler_name": (_sampler_names(), {"default": "res_multistep"}),
            "scheduler": (_scheduler_names(), {"default": "simple"}),
            "bank_pinned": ("INT", {
                "default": 1, "min": 0, "max": 3,
                "tooltip": "How many of the EARLIEST shots stay in the bank "
                           "permanently. This is the anti-drift lever: with "
                           "shot 1 pinned, later shots always see where the "
                           "episode started. 0 = pure recency."}),
            "chain_gain_control": (["off", "flatten", "match_output",
                                    "flatten_pin", "refresh_pin"], {
                "default": "off",
                "tooltip": "Texture levelling across the chain. flatten = "
                           "level the DECODED frames and the bank to shot 1 "
                           "(the pin still carries accreted texture; measured "
                           "x1.13-1.15 per hop at 736x1280 with flatten on). "
                           "flatten_pin = EXPERIMENTAL: flatten PLUS level "
                           "the pinned latents' fine-detail energy to shot "
                           "1's before they are pinned. First A/Bs "
                           "(2026-08-17) moved the ratchet only slightly; "
                           "kept for testing, not a default. Long context_pin "
                           "chains and extend takes past ~4 windows still "
                           "sharpen visibly - keep takes short until the fix "
                           "lands."}),
            "continuity": (["cut", "seamless", "seamless_tail",
                            "latent_handoff", "first_frame", "flf_chain",
                            "context_pin"], {
                "default": "cut",
                "tooltip": "cut = JoyEcho-pure: no keyframe, every shot is a "
                           "fresh take held together by the bank. Framing and "
                           "exposure step between shots - correct for "
                           "multishot storytelling with cuts.\n"
                           "seamless = LEGACY, kept for comparison: hands "
                           "the next shot its predecessor's last frame as a "
                           "latent-only keyframe. That is a SOFT hint - no "
                           "vision tokens - and the model often satisfies it "
                           "loosely, so the join can still read as a cut. "
                           "For a real join use context_pin or "
                           "first_frame.\n"
                           "seamless_tail = LEGACY, kept for comparison: "
                           "pins the previous shot's frames -9/-5/-1 at "
                           "keyframe indices 0/4/8 so velocity carries too. "
                           "Needs interior keyframe anchors, which CONFLICT "
                           "with the Motion-Context pack - with that pack "
                           "installed this mode stops with an error naming "
                           "the alternatives instead of crashing mid-chain.\n"
                           "latent_handoff = one denoise trajectory: the next "
                           "shot's first latent block (video AND audio) is "
                           "hard-locked to the previous shot's actual tail "
                           "latents at every sampling step, released only for "
                           "the final detail steps. Speech continues mid-word "
                           "because the model wakes up inside its own "
                           "previous state - no keyframes involved.\n"
                           "first_frame = the model's OWN continuation "
                           "mechanism (fl2va task): the previous shot's "
                           "last frame is handed over the way the stock "
                           "Image-to-Video node does it - as VISION TOKENS "
                           "through the text encoder AND as the frame-0 "
                           "keyframe latent. The new shot literally starts "
                           "on that frame; only the duplicate is trimmed. "
                           "USE AN fl2va CHECKPOINT (ref2va is trained for "
                           "reference rows, not first-frame hand-off) - and "
                           "note the bank is disabled here, because fl2va "
                           "has no reference rows.\n"
                           "flf_chain = TRUE FFLF. Supply N+1 boundary "
                           "keyframes for N shots; shot i renders BETWEEN "
                           "keyframe i and keyframe i+1. Shot i ends on "
                           "exactly the image shot i+1 begins on, so the "
                           "join is one shared picture rather than two "
                           "independent guesses - colour, framing and pose "
                           "match by construction.\n"
                           "context_pin = Motion-Context chaining (needs the "
                           "ComfyUI-H3-Motion-Context pack): the previous "
                           "shot's last 22 frames are pinned into the next "
                           "shot's head AS RAW LATENTS at interior keyframe "
                           "coordinates - bit-identical content, no VAE "
                           "round trip, so velocity AND colour carry - plus "
                           "a timeline-placed audio ref. The regenerated "
                           "head is trimmed on decode. Composes with the "
                           "bank, colour levels and join fx."}),
            "bank_clip_frames": ("INT", {
                "default": 22, "min": 5, "max": 124, "step": 17,
                "tooltip": "Frames per bank slot, taken from the middle of "
                           "each shot (JoyEcho stores a clip, not a single "
                           "frame). Reference rows cost time on every sampling "
                           "step, so keep this small: 22 is ~0.9s."}),
            "color_level": (["off", "scene", "mvgd"], {
                "default": "off",
                "tooltip": "Colour drift across a chain. 'scene' is the one to "
                           "use: ONE reference for the whole piece, applied "
                           "per frame at the very end, so every shot is pulled "
                           "to the same target and there is no step at any "
                           "join. 'mvgd' is DEPRECATED - it matches each shot "
                           "to a rolling house and leaves a hard step at every "
                           "seam (measured 29% warmth step; a render with it "
                           "on drifted +18% brighter over three shots). It is "
                           "kept only so saved graphs still load."}),
            "join_anchor_noise": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 0.05, "step": 0.005,
                "tooltip": "Mix this much seeded noise into every join "
                           "keyframe latent (SkyReels noised-clean-condition). "
                           "The texture ratchet exists because the model "
                           "treats its own output as pristine and adds ~1.2x "
                           "detail on top; a little noise closes that gap at "
                           "the source. 0.02 is the researched setting. The "
                           "noised frames never reach the final cut."}),
            "join_blend": ("BOOLEAN", {
                "default": False, "label_on": "crossfade overlap",
                "label_off": "hard drop",
                "tooltip": "seamless_tail only: instead of hard-dropping the "
                           "9 regenerated overlap frames, crossfade them "
                           "against the previous tail (with a grain guard so "
                           "the blend band does not read as a grain dip) and "
                           "fade audio over the same 375ms. Any residual step "
                           "is spread across 9 frames instead of landing on "
                           "one boundary."}),
            "handoff_release": ("FLOAT", {
                "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "latent_handoff only: sigma below which the locked "
                           "overlap is released so the detail steps can "
                           "reconcile it with the new content. Higher = freer "
                           "(released earlier), 0 = locked to the very end."}),
            "bank_ref_noise": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 0.05, "step": 0.005,
                "tooltip": "Mix this much seeded noise into every bank clip "
                           "before it is stored (SkyReels noised-clean-"
                           "condition, same idea as join_anchor_noise but for "
                           "the references). The texture ratchet rides bank "
                           "clips: the model copies its own output and adds "
                           "~1.2x detail, worst on faces/skin. 0.02 is the "
                           "keyframe-researched setting."}),
            "end_anchor": ("BOOLEAN", {
                "default": False, "label_on": "return to house framing",
                "label_off": "off",
                "tooltip": "Pin shot 1's FIRST frame as a keyframe at the "
                           "LAST frame of every later shot. H3 push-in creep "
                           "compounds across chained shots (each shot "
                           "inherits the previous crept tail and pushes "
                           "further), so tails drift ever further from the "
                           "prompted framing and every join mechanism "
                           "inherits an off-spec tail. The end pin closes "
                           "the loop: a shot may breathe inward mid-take but "
                           "must settle back to house framing by its tail, "
                           "so the next join starts from framing the text "
                           "agrees with."}),
            "join_fx": (["off", "vhs_glitch"], {
                "default": "off",
                "tooltip": "Diegetic join masking: dress every join in a "
                           "short VHS tracking hiccup (displacement bands, "
                           "chroma shift, dropout flecks, audio head-switch "
                           "duck+hiss) peaking on the boundary. For analog-"
                           "horror content the cut stops being an artifact "
                           "to hide and becomes part of the tape. Works with "
                           "every continuity mode."}),
            "audio_lock": ("BOOLEAN", {
                "default": True, "label_on": "locked (replay/spine)",
                "label_off": "free (silent-join)",
                "tooltip": "latent_handoff only. ON: the audio head is a "
                           "locked replay of the previous tail (or the "
                           "spine). OFF = the SILENT-JOIN policy: script "
                           "each shot to land its line and hold still for "
                           "the last beat; audio generates freely, the new "
                           "head's audio is kept in full and the previous "
                           "tail's audio is trimmed instead. Nothing the "
                           "model generates is discarded, so no word can "
                           "be lost BY CONSTRUCTION. A connected "
                           "guide_audio spine overrides this to locked."}),
            "handoff_taper": ("INT", {
                "default": 0, "min": 0, "max": 10,
                "tooltip": "latent_handoff only. Rows AFTER the hard lock "
                           "that are softly biased toward a continuation of "
                           "the previous motion, at linearly decaying "
                           "strength. Without it the lock ends at a cliff: "
                           "the replayed frames are faithful, then the next "
                           "frame follows the shot's OWN pose plan - "
                           "measured as a 3x larger discontinuity on the "
                           "person than on the static room. The taper gives "
                           "the pose a ramp to follow. 3-5 is a good start "
                           "(each row = 4 frames)."}),
            "handoff_depth": (["block", "bootstrap"], {
                "default": "block",
                "tooltip": "latent_handoff overlap depth. block = lock the "
                           "first full 17-frame block (strong video anchor, "
                           "22 frames trimmed, ~0.92s join gap). bootstrap "
                           "= lock only the 2 bootstrap rows to the "
                           "previous last 5 frames (5 frames trimmed, "
                           "~0.21s gap - an ordinary breath), with a "
                           "frame-0 keyframe pin of the previous last "
                           "frame; end_anchor + the bank carry the rest."}),
            # NEW WIDGETS GO LAST, ALWAYS: saved canvases map widgets_values
            # by index, and a widget inserted mid-list silently shifts every
            # value after it on the next load (the v1.4 lesson). Sockets
            # (IMAGE/AUDIO/forceInput STRING) take no widget slot, so their
            # position here is free.
            "reference_images": ("IMAGE", {
                "tooltip": "Optional SUBJECT/CHARACTER reference images (batch "
                           "= multiple refs, e.g. via Batch Images), carried "
                           "into EVERY shot as <Picture 1>, <Picture 2>, ... "
                           "ahead of the bank slots, so their numbering never "
                           "shifts as the bank fills. Distinct from "
                           "start_image, which seeds identity for shot 1 only. "
                           "Bind them in each shot's prompt: 'She looks like "
                           "the woman in <Picture 1>.'"}),
            "voice_ref": ("AUDIO", {
                "tooltip": "Optional VOICE ANCHOR carried into EVERY shot as a "
                           "reference audio (<Audio 1>). Feed a clean solo "
                           "line of the character and the voice is PINNED "
                           "across the chain instead of re-performed from "
                           "text. The bank carries voice too, but only from "
                           "shot 2 on - this covers shot 1 as well. Trimmed to "
                           "15s; reference rows cost speed on every step."}),
            "sampler_override": ("STRING", {
                "forceInput": True,
                "tooltip": "Link a sampler NAME here (e.g. from H3 Studio "
                           "Controls) to drive this widget from one master "
                           "source. Overrides sampler_name when connected."}),
            "scheduler_override": ("STRING", {
                "forceInput": True,
                "tooltip": "Link a scheduler NAME here to single-source it. "
                           "Overrides scheduler when connected."}),
            "self_anchor_voice": ("BOOLEAN", {
                "default": False, "label_on": "anchor to shot 1's voice",
                "label_off": "off",
                "tooltip": "AUTOMATIC voice identity: after shot 1 renders, "
                           "its own audio becomes the reference (<Audio 1>) "
                           "for every later shot - the voice the model "
                           "actually performed is pinned, no file needed. "
                           "Write shot 1 so the character speaks a clean "
                           "solo line. An external voice_ref, if connected, "
                           "takes priority."}),
            "reference_image_size": (["match", "max"], {
                "default": "match",
                "tooltip": "Reference image sizing. 'match' scales each ref "
                           "(down only, keeping aspect) to the generation's "
                           "pixel area; 'max' uses the reference pipeline's "
                           "2048px short edge for best identity fidelity. "
                           "Reference tokens ride through every sampling "
                           "step, so 'max' can be several times slower."}),
            "preview_first_shot": ("BOOLEAN", {
                "default": False, "label_on": "save shot 1 early",
                "label_off": "off",
                "tooltip": "Write shot 1 to output/video/H3_FIRSTSHOT/ the "
                           "MOMENT it finishes decoding - minutes before the "
                           "full chain completes - so a bad take can be "
                           "cancelled early. The full path is printed to the "
                           "console."}),
            "output_scale": ("FLOAT", {
                "default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05,
                "tooltip": "FINAL size multiplier, applied after decode. "
                           "No upscale model: a lanczos resize, 1.0 is off. "
                           "WITH a model: the model runs at its OWN fixed "
                           "factor (usually 4x) and this brings the result to "
                           "source x this value, so 2.0 on a 4x model gives "
                           "2x, not 8x. "
                           "CAREFUL - 1.0 does NOT mean off once a model is "
                           "wired; it means do-not-correct, so you get the "
                           "full 4x. At 1344x768 that is 5376x3072: 94 MB a "
                           "frame, 22 GB a shot, and every shot stays in "
                           "system RAM until the master is joined. The console "
                           "prints the projected size when the model loads - "
                           "read it. "
                           "Adds resolution, not detail. Works with every "
                           "continuity mode; the bank still stores "
                           "base-resolution clips."}),
            "sigmas": ("SIGMAS", {
                "tooltip": "Optional custom sigma schedule, replacing "
                           "sampler/scheduler + steps entirely. Some turbo "
                           "LoRAs ship a schedule they need in order to work "
                           "at all. When this is connected the 'steps' and "
                           "'scheduler' widgets are IGNORED - the step count "
                           "becomes len(sigmas)-1 - and the console says so. "
                           "The two-pass upscale split is taken as a fraction "
                           "of the supplied schedule."}),
            "save_every_shot": ("BOOLEAN", {
                "default": False, "label_on": "write each shot as it decodes",
                "label_off": "off",
                "tooltip": "Write EVERY shot to output/video/H3_SHOTS/ the "
                           "moment it decodes, in addition to the master. "
                           "Insurance for long chains: everything that fails "
                           "after the last shot - a mux OOM, a full disk, a "
                           "cancelled tab - otherwise destroys the whole "
                           "render at once. Shots are written BEFORE the seam "
                           "trim, so consecutive files overlap by ~1s; the "
                           "master is still the clean join. Costs one file "
                           "write per shot."}),
            "upscale_model_name": (["(none)"] + _up_model_list(), {
                "default": "(none)",
                "tooltip": "Pick an upscale model by name instead of wiring a "
                           "loader node. Synthesises detail rather than "
                           "resizing, per shot, at the model's OWN fixed "
                           "factor - usually 4x. "
                           "Set output_scale to the size you actually want; "
                           "leaving it at 1.0 lets the raw 4x through, which "
                           "is rarely what you meant. The console prints the "
                           "projected frame size and RAM cost when the model "
                           "loads, before any sampling. "
                           "Reads models/upscale_models/, the same folder "
                           "ComfyUI's own Load Upscale Model reads."}),
            "master_normalize": (["off", "luma", "luma+contrast"], {
                "default": "luma+contrast",
                "tooltip": "Deflicker the FINISHED chain: every frame driven "
                           "to ONE global luma target, after the master "
                           "exists. Per-shot correction cannot work here - it "
                           "never reaches the raw-latent pin that carries the "
                           "drift, and correcting shots against a rolling "
                           "target leaves a step at every join (measured: all "
                           "per-shot dials ON still gave +142% texture and "
                           "+18% luma over three shots). This runs outside the "
                           "feedback loop and lands every frame on the same "
                           "number, so it cannot create a seam. Brightness "
                           "only: texture drift is NOT fixable after the fact, "
                           "because the only lever is blur and blur destroys "
                           "real detail along with the invented kind."}),
            # APPEND-ONLY from here. Inserting a widget above this line shifts
            # every saved workflow's values by one (v1.2 did exactly that with
            # seed_per_shot, and users got "Value 4 bigger than max of 3:
            # memory_frames" on graphs they had never edited).
            "pin_frames": (["22", "5", "39", "56"], {
                "default": "22",
                "tooltip": "context_pin only: how many frames of the previous "
                           "shot are pinned as raw latents at the head of the "
                           "next one. This is the whole join. 22 is the shipped "
                           "default; the longer settings hold the previous "
                           "shot's composition further into the new one, which "
                           "matters most at the FIRST join - shot 1 has nothing "
                           "pinned behind it, so it is the only shot whose "
                           "framing can disagree with the text. All four values "
                           "are latent-aligned; arbitrary numbers are not."}),
            "pin_noise": ("FLOAT", {
                "default": 0.05, "min": 0.0, "max": 0.10, "step": 0.005,
                "tooltip": "context_pin only: mix this much seeded noise into "
                           "the PINNED LATENT before it conditions the next "
                           "shot. Same noised-clean-condition idea as "
                           "join_anchor_noise, aimed at the thing that "
                           "actually carries the drift here - measured, the "
                           "texture ratchet under context_pin rides the raw "
                           "latent pin, and neither join_anchor_noise "
                           "(keyframes only) nor bank_ref_noise (bank images) "
                           "touches it. "
                           "Small, and scene-dependent: measured -1.8% per hop at "
                           "640x352 and -0.9% at 960x544 on a "
                           "detail-heavy scene, against much larger "
                           "gains on a scene that barely ratcheted at "
                           "all. It cannot touch the dominant drift in a "
                           "busy frame - master_normalize=luma+contrast "
                           "is what does that. Above 0.10 it gets WORSE "
                           "(0.20 measured 1.228 against a 1.211 "
                           "control), which is why the range stops "
                           "there. Set 0 to disable. "
                           "The noised latent conditions the next shot but "
                           "never reaches the final cut."}),
            "pin_renorm": ("BOOLEAN", {
                "default": True,
                "label_on": "hold shot 1's level",
                "label_off": "off",
                "tooltip": "context_pin only: rescale each pinned latent so "
                           "its standard deviation matches the FIRST "
                           "pin's. The pin's own sigma climbs every hop "
                           "(1.0325, 1.0368 against a 1.0220 shot-1 "
                           "anchor), and that inflated pin is what "
                           "conditions the next shot - so it compounds "
                           "upstream of anything a master pass can reach. "
                           "Measured against a matched control, same seed, "
                           "960x544 124f x4: texture growth over the chain "
                           "+15.1% -> +11.5%, and framing correlation to "
                           "shot 1 held 0.985 -> 0.996 by the last shot. "
                           "The framing gain is the bigger one and cannot "
                           "be a metric artifact - post passes do not move "
                           "composition. One seed, one canvas. A scalar "
                           "rescale moves no structure, so unlike a pixel "
                           "correction it cannot blur detail."}),
            # NEW IN 2.2.4 - appended at the very END of the optional block on
            # purpose. widgets_values is a POSITIONAL array, so inserting a
            # widget anywhere but the end silently shifts every value after it
            # in workflows people have already saved. That is what produced the
            # per-shot sharpening incident; never insert mid-list.
            "reference_subjects": ("STRING", {
                "default": "",
                "tooltip": "How your reference pictures group into PEOPLE. "
                           "Empty (default) = every reference picture is "
                           "declared a photograph of the same one person. That "
                           "is correct for a single character and WRONG for "
                           "several - the model is told they are all the same "
                           "individual and renders the average. Comma counts in "
                           "picture order: '3,3' means pictures 1-3 are person "
                           "A and 4-6 are person B; '2,2,2' for three people. "
                           "Only <Subject 1> is described as speaking, because "
                           "H3's voice conditioning is single-speaker."}),
            "reference_video": ("IMAGE", {
                "tooltip": "NEW IN 2.2.4. Frames of an EXISTING clip, handed to "
                           "the model as a video reference alongside the "
                           "picture references. Read what this does before "
                           "wiring it: H3 is told a video reference is 'an "
                           "earlier moment of this same continuous scene' and "
                           "to keep its framing, camera distance, room contents "
                           "and colour temperature. It is SCENE and APPEARANCE "
                           "conditioning. It is NOT motion transfer - there is "
                           "no pose, depth or optical-flow path in H3, so it "
                           "will not make your subject copy the movement in the "
                           "clip. Feed it through H3ReferenceVideo, which trims "
                           "and subsamples to the 2 fps the model actually "
                           "reads; a raw 25-second clip is ~50 reference frames "
                           "riding every sampling step."}),
            "reference_video_audio": ("AUDIO", {
                "tooltip": "Optional soundtrack for reference_video. H3 pairs "
                           "video references with an audio reference, so if you "
                           "leave this empty silence is generated to match. "
                           "Supply the clip's real audio when you want its "
                           "voice timbre referenced too."}),
            # APPENDED LAST (saved-graph widget order).
            "low_ram_master": ("BOOLEAN", {
                "default": False,
                "tooltip": "Stream the master to disk instead of holding every "
                           "decoded shot in host RAM until the join - peak RAM "
                           "becomes ONE shot. Each shot is staged lossless the "
                           "moment it decodes, levelled with the same "
                           "master_normalize math (from stored statistics), "
                           "and the finished file's path comes out of the new "
                           "master_path output; master_frames carries a single "
                           "placeholder frame. v1 streams the default config "
                           "only: join_blend, join_fx and color_level=scene "
                           "fall back to the RAM path with a printed reason. "
                           "Needs ffmpeg on PATH."}),
            "audio_pin_frames": ("INT", {
                "default": 0, "min": 0, "max": 240, "step": 1,
                "tooltip": "context_pin only: frames of the previous shot's "
                           "AUDIO to pin as reference, independent of the "
                           "picture pin. 0 = same as pin_frames (22 = 0.9 s). "
                           "Longer audio context costs conditioning rows but "
                           "NO delivered frames - the head trim stays at "
                           "pin_frames. 96 (4 s) is the audio-memory window "
                           "the JoyEcho ancestor of this sampler carried "
                           "between chunks; try it for continuous speech "
                           "across joins. Experimental."}),
        },
            # hidden inputs are not widgets, so saved workflows are unaffected
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "LATENT", "LATENT", "INT",
                    "STRING")
    RETURN_NAMES = ("master_frames", "master_audio", "shots_rendered",
                    "video_latents", "audio_latents", "head_frames",
                    "master_path")
    OUTPUT_TOOLTIPS = (
        "The joined master, seams trimmed.",
        "Master audio.",
        "How many shots rendered.",
        "Every shot's video latent EXACTLY as sampled, batched along dim 0 "
        "(one entry per shot), UNTRIMMED. Shots after the first open with "
        "head_frames of replayed material from the previous shot's tail - "
        "that is what makes the join seamless, and it is removed at decode, "
        "not in the latent. So this does NOT line up with master_frames "
        "until you trim it yourself. Latent temporal rows for F frames are "
        "5*((F-5)//17)+2.",
        "The matching audio latent per shot, same batching, same caveat.",
        "Frames of replayed head carried by every shot AFTER the first "
        "(0 for shot 1, and 0 for continuity modes that do not pin). Trim "
        "this many frames off the front of each later shot after you decode.",
        "low_ram_master only: the finished master file on disk. Empty string "
        "when low_ram_master is off (use master_frames as always).")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"

    @classmethod
    def VALIDATE_INPUTS(cls, memory_frames):
        # Naming memory_frames here hands ITS range check to us and leaves
        # every other input on ComfyUI's own validation.
        #
        # Out of range almost never means the user typed it. v1.2 inserted
        # seed_per_shot at widget index 8, ahead of memory_frames, so a
        # workflow saved on v1.0/v1.1 reads its old anchor_frames (0-9) into
        # memory_frames (0-3) - and the stock message, "Value 4 bigger than
        # max of 3", sends people hunting a dial they never touched. The JS
        # extension repairs that on load; this is the backstop for anyone
        # running with custom frontend extensions disabled.
        if not 0 <= int(memory_frames) <= 3:
            return (f"memory_frames is {memory_frames}, but the range is 0-3 "
                    f"(H3 holds at most 3 references, pinned + recent).\n"
                    f"If you did not set it: this workflow was saved before "
                    f"v1.2, which added seed_per_shot ahead of memory_frames "
                    f"and shifted every dial after it by one. Re-open the "
                    f"shipped H3_Seamless_Chain_v2.json, or right-click the "
                    f"sampler -> Fix node (recreate) and re-enter your "
                    f"settings, then save.")
        return True

    def run(self, model, clip, video_vae, audio_vae, script, shot_count, width,
            height, frames_per_shot, seed, steps, memory_frames, anchor_frames,
            seed_per_shot=True, start_image=None,
            sampler_name="res_multistep", scheduler="simple",
            bank_pinned=1, chain_gain_control="off", bank_clip_frames=22,
            continuity="cut", color_level="off", join_anchor_noise=0.0,
            join_blend=False, handoff_release=0.30, bank_ref_noise=0.0,
            end_anchor=False, join_fx="off", audio_lock=True,
            handoff_taper=0, handoff_depth="block", guide_audio=None,
            keyframe_images=None, reference_images=None, voice_ref=None,
            sampler_override=None, scheduler_override=None,
            self_anchor_voice=False, reference_image_size="match",
            preview_first_shot=False,
            save_every_shot=False, sigmas=None, output_scale=1.0,
            upscale_model_name="(none)",
            master_normalize="luma+contrast", pin_frames="22", pin_noise=0.0,
            pin_renorm=False, reference_subjects="",
            reference_video=None, reference_video_audio=None,
            low_ram_master=False, audio_pin_frames=0,
            prompt=None, extra_pnginfo=None):
        # Keep the hidden PROMPT before anything can shadow it: the shot loop
        # rebinds `prompt` to this shot's conditioning TEXT, so by finalize()
        # the API graph is gone and the streamed master was tagged with the
        # last shot's script (24 GB test-lab finding F014, 2026-08-16 - core SaveVideo wrote
        # a 42-node dict, the streamed master a 1254-char string, same run).
        _api_prompt, _api_pnginfo = prompt, extra_pnginfo
        # two_pass_upscale is REMOVED as of 2.1.3. It spatially interpolated
        # the raw latent between passes, and H3's latent is not a spatially
        # smooth representation - interpolated values land off-manifold and
        # pass 2, running at low sigma, has no room to pull them back. Three
        # A/Bs at the previously documented recipe (14 steps, beta57) came
        # back as colour-noise mush against clean single-pass controls, with
        # shot 1 - which carries no pin at all - destroyed identically, so it
        # was never about the chaining. Upscaling now happens AFTER decode
        # (output_scale / upscale_model), where it cannot leave the manifold.
        # The branches below are kept only so the diff stays readable; they
        # are unreachable.
        two_pass_upscale = False
        upscale_factor, pass1_fraction, upscale_audio_denoise = 1.5, 0.4, 0.35
        if sampler_override and str(sampler_override).strip():
            sampler_name = str(sampler_override).strip()
        if scheduler_override and str(scheduler_override).strip():
            scheduler = str(scheduler_override).strip()
        import torch
        import node_helpers
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3
        from comfy_extras.nodes_audio import vae_decode_audio
        import comfy.model_management as _mm

        shots = _parse_script(script)
        n = shot_count if shot_count > 0 else len(shots)
        if len(shots) > n:
            shots = shots[:n]
        while len(shots) < n:
            shots.append(shots[-1])

        if sigmas is not None and len(sigmas) > 1:
            # a supplied schedule wins: some turbo LoRAs only converge on the
            # exact curve they shipped with, and silently re-deriving one from
            # steps+scheduler would make them look broken rather than misused
            steps = int(len(sigmas)) - 1
            print("[%s] custom sigmas supplied (%d steps): the steps and "
                  "scheduler widgets are ignored for this run."
                  % ("H3Memory", steps), flush=True)
        else:
            sigmas = ncs.BasicScheduler().get_sigmas(model, scheduler,
                                                     steps, 1.0)[0]
        sampler = ncs.KSamplerSelect().get_sampler(sampler_name)[0]

        # --- voice anchor: encode ONCE, ride in every shot's conditioning ---
        # The bank already carries voice from shot 2 on, but shot 1 renders
        # against an empty bank; an explicit ref covers the whole chain.
        voice_block = None
        if voice_ref is not None:
            _vw = voice_ref["waveform"]
            if _vw.ndim == 2:
                _vw = _vw.unsqueeze(0)
            _vw = _vw[:1]
            if _vw.shape[1] == 1:          # mono crashes the packed layout
                _vw = _vw.repeat(1, 2, 1)
            elif _vw.shape[1] > 2:
                _vw = _vw[:, :2]
            _vsr = int(voice_ref["sample_rate"])
            _vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
            if _vsr != _vae_sr:
                import torchaudio
                _vw = torchaudio.functional.resample(_vw, _vsr, _vae_sr)
                _vsr = _vae_sr
            _lim = 15 * _vsr               # ref rows cost speed EVERY step
            if _vw.shape[-1] > _lim:
                _vw = _vw[..., :_lim]
            _vz = audio_vae.encode(_vw.movedim(1, -1))       # [1, 32, 2, T]
            voice_block = {"kind": "audio", "ref_audio_t": _vz.shape[-1],
                           "audio_latent": _vz}
            print(f"[H3Memory] voice anchor: {_vw.shape[-1] / _vsr:.1f}s ref "
                  f"audio rides in every shot as <Audio 1>.", flush=True)

        # --- subject/character reference images: encode ONCE, fixed slots ---
        import math as _math_ri
        ref_image_items, ref_image_blocks = [], []
        if reference_images is not None:
            for _ri in range(reference_images.shape[0]):
                _img = reference_images[_ri:_ri + 1]
                _h, _w = _img.shape[1], _img.shape[2]
                if reference_image_size == "match":
                    _sc = min(1.0, _math_ri.sqrt((width * height) / (_w * _h)))
                else:
                    _sc = min(1.0, mmh3.REF_IMAGE_SHORT_EDGE / min(_w, _h))
                _tw = max(mmh3.CANVAS_MULTIPLE,
                          round(_w * _sc / mmh3.CANVAS_MULTIPLE)
                          * mmh3.CANVAS_MULTIPLE)
                _th = max(mmh3.CANVAS_MULTIPLE,
                          round(_h * _sc / mmh3.CANVAS_MULTIPLE)
                          * mmh3.CANVAS_MULTIPLE)
                _rz = mmh3._resize(_img, _tw, _th, "disabled")
                ref_image_items.append({"type": "image", "data": _rz})
                ref_image_blocks.append({"kind": "image",
                                         "latent_h": _th // 16,
                                         "latent_w": _tw // 16,
                                         "latent": video_vae.encode(_rz)})
            print(f"[H3Memory] {len(ref_image_blocks)} reference image(s) ride "
                  f"in every shot as <Picture 1..{len(ref_image_blocks)}>.",
                  flush=True)

        # --- two-pass upscale setup: low-res pass 1, exact full-res pass 2 ---
        # The raw-latent continuity modes pin the previous shot's latents into
        # THIS shot's grid. Pass 1 runs on a smaller grid, so the pin does not
        # fit - and resampling it would destroy the bit-identical hand-off that
        # is the entire reason those modes exist. Refuse rather than degrade.
        _tp_tr = _tp_w1 = _tp_h1 = _tp_sig_hi = _tp_sig_lo = None
        _tp_lat_th = _tp_lat_tw = None
        _tp_lat_h1 = _tp_lat_w1 = None   # pass-1 latent grid (pin target)
        if two_pass_upscale:
            if continuity == "latent_handoff":
                raise ValueError(
                    "two_pass_upscale is not compatible with continuity="
                    "'latent_handoff'. That mode renoises the previous tail "
                    "into the CURRENT trajectory at every step, and a "
                    "two-pass render is two trajectories - the handoff would "
                    "be dropped halfway. Use continuity=context_pin, which "
                    "does support two_pass_upscale (the pin is resampled "
                    "onto the pass-1 grid), or turn two_pass_upscale off.")
            _tp_tr = _load_upscaler_utils()
            _f = max(1.0, float(upscale_factor))
            _tp_w1 = max(32, int(round(width / _f / 32)) * 32)
            _tp_h1 = max(32, int(round(height / _f / 32)) * 32)
            _k = int(round(steps * float(pass1_fraction)))
            _k = max(1, min(steps - 1, _k))
            _tp_sig_hi, _tp_sig_lo = sigmas[:_k + 1], sigmas[_k:]
            _probe, _ = mmh3._empty_av_latent(width, height, frames_per_shot)
            _pv = _probe["samples"]
            _pv = _pv.unbind()[0] if getattr(_pv, "is_nested", False) else _pv
            _tp_lat_th, _tp_lat_tw = int(_pv.shape[-2]), int(_pv.shape[-1])
            del _probe, _pv
            print(f"[H3Memory] two-pass: {_tp_w1}x{_tp_h1} for {_k} steps -> "
                  f"latent x{width / _tp_w1:.2f} -> {width}x{height} for "
                  f"{steps - _k} steps (audio_denoise "
                  f"{upscale_audio_denoise}).", flush=True)

        # H3 allows at most 3 video references, so the bank is capped there
        cap = max(1, min(3, int(bank_pinned) + int(memory_frames)))
        # task/checkpoint guard: continuity mode dictates the checkpoint.
        _ckpt = str(getattr(getattr(model, "model", None),
                            "h3_checkpoint_name", "") or "").lower()
        if _ckpt:
            _is_fl = "fl2va" in _ckpt
            _is_ref = "ref2va" in _ckpt
            if continuity == "first_frame" and _is_ref:
                print("[H3Memory] WARNING: continuity=first_frame hands the "
                      "previous last frame over as the fl2va task, but a "
                      "ref2va checkpoint is loaded. The hand-off will be "
                      "weak (soft keyframe only). Load an fl2va checkpoint.",
                      flush=True)
            elif continuity != "first_frame" and _is_fl and bank_pinned >= 0 \
                    and memory_frames > 0:
                print("[H3Memory] WARNING: the memory bank needs reference "
                      "rows (ref2va), but an fl2va checkpoint is loaded. "
                      "Bank slots will be ignored - use continuity="
                      "first_frame with fl2va, or load ref2va.", flush=True)
            if start_image is not None and _is_fl:
                # Reported from the field: start_image connected, fl2va loaded,
                # continuity=first_frame, and shot 1 does not open on the
                # supplied picture. Nothing was misconfigured - this input is an
                # identity REFERENCE ROW here, and fl2va has no reference rows,
                # so it is built and then ignored. The sibling node
                # H3MultishotSampler has an input with the SAME NAME that really
                # is a first frame, which is where the expectation comes from.
                print("[H3Memory] WARNING: start_image on THIS node is an "
                      "identity reference image, NOT a first frame - and an "
                      "fl2va checkpoint has no reference rows, so it is "
                      "ignored entirely. Shot 1 will NOT open on it. To start "
                      "shot 1 on a specific picture: use the H3MultishotSampler "
                      "node, whose start_image is a true I2V first frame, or "
                      "set continuity=flf_chain here and feed keyframe_images "
                      "(N+1 stills for N shots). To use it for identity "
                      "instead, load a ref2va checkpoint.", flush=True)

        bank = _H3ChainBank(num_fix=bank_pinned, max_size=cap)
        frames_parts, audio_parts = [], []
        # ---- low_ram_master eligibility (v1: shipped default config only) --
        _stream_writer = None
        if bool(locals().get("low_ram_master")):
            _blockers = []
            # join_blend is NOT a blocker: it only runs in seamless_tail/
            # latent_handoff, and the writer defers each shot one step so
            # the blend can still mutate the previous tail before staging.
            if str(locals().get("join_fx", "off")) not in ("off", "none", ""):
                _blockers.append("join_fx")
            if str(locals().get("color_level", "")) == "scene":
                _blockers.append("color_level=scene")
            if _blockers:
                print("[H3Memory] low_ram_master: falling back to the RAM "
                      "path - %s read neighbouring shots and are not "
                      "streamable in v1." % "+".join(_blockers), flush=True)
            else:
                try:
                    import os
                    try:
                        from .h3_stream_master import ShotStreamWriter
                    except ImportError:
                        import importlib.util as _ilu
                        _sp = os.path.join(os.path.dirname(
                            os.path.abspath(__file__)), "h3_stream_master.py")
                        _spec = _ilu.spec_from_file_location(
                            "h3_stream_master", _sp)
                        _mod = _ilu.module_from_spec(_spec)
                        _spec.loader.exec_module(_mod)
                        ShotStreamWriter = _mod.ShotStreamWriter
                    import folder_paths as _fp
                    _tdir = os.path.join(_fp.get_output_directory(), "video",
                                         "H3CHAIN_STREAM",
                                         "tmp_%d" % os.getpid())
                    _stream_writer = ShotStreamWriter(_tdir, fps=24)
                    print("[H3Memory] low_ram_master ON: shots stage to %s, "
                          "peak host RAM = one shot." % _tdir, flush=True)
                except Exception as _e:  # noqa: BLE001
                    print("[H3Memory] low_ram_master unavailable (%s) - "
                          "using the RAM path." % _e, flush=True)
                    _stream_writer = None

        _lat_v_parts, _lat_a_parts = [], []   # issue #12: raw per-shot latents
        upscale_model = None
        if upscale_model_name not in ("(none)", "", None):
            # load it NOW, before a single step is sampled. The first version
            # of this loader raised inside the per-shot upscale, i.e. AFTER
            # shot 1 had rendered - six minutes of GPU burned to reach a
            # failure that was knowable at zero.
            try:
                upscale_model = _load_up_model(upscale_model_name)
            except Exception as _e:
                raise RuntimeError(
                    "upscale_model_name=%r could not be loaded: %s. Fix the "
                    "name or set it to (none) - failing now rather than after "
                    "the first shot has rendered." % (upscale_model_name, _e))
            print("[H3Memory] upscale model: %s (loaded and verified)"
                  % upscale_model_name, flush=True)
            # Predict the cost NOW, not after the first shot has been sampled
            # and upscaled. Frames accumulate on the host until the master is
            # joined, so the total is per-shot x shots - and output_scale left
            # at 1.0 lets the model's full factor through, which is how a run
            # ended up projecting 134.6 GB against 93.6 GB of RAM.
            try:
                import psutil as _ps
                _ram = _ps.virtual_memory().total / 2**30
            except Exception:
                _ram = 0.0
            _f = float(output_scale) if output_scale and abs(
                float(output_scale) - 1.0) > 1e-6 else _up_model_factor(
                    upscale_model)
            _ow, _oh = int(round(width * _f)), int(round(height * _f))
            _per = _ow * _oh * 3 * 2 / 2**30          # fp16 on the host
            _tot = _per * frames_per_shot * max(1, len(shots))
            print("[H3Memory] upscale will produce %dx%d (%.1fx): %.0f MB per "
                  "frame, %.1f GB per shot, %.1f GB for %d shot(s)%s"
                  % (_ow, _oh, _f, _per * 1024, _per * frames_per_shot, _tot,
                     max(1, len(shots)),
                     (" against %.1f GB of RAM" % _ram) if _ram else ""),
                  flush=True)
            if _ram and _tot > _ram * 0.8:
                print("[H3Memory] WARNING: that will not fit. Frames are held "
                      "in system RAM until the master is joined, so this dies "
                      "at the join after every shot has been paid for. Set "
                      "output_scale to %.2f or lower, or turn the upscaler off."
                      % max(1.0, (_ram * 0.7 / (frames_per_shot * max(1, len(shots))
                                                * width * height * 3 * 2 / 2**30))
                            ** 0.5), flush=True)
        sr = None
        _cg_ref = None
        _CG_WIN = 24
        last_tail = None       # seamless modes: the physical join frames
        _TAIL_K = 2            # bracket depth: pixel frames -9/-5/-1 -> idx 0/4/8
        _OV = 1 + 4 * _TAIL_K  # overlap frames regenerated by the next shot
        _dbg_pins = []         # (frame_index, clean pinned image) for adherence
        # latent_handoff geometry. Video latent: 2 bootstrap rows for the
        # first 5 frames, then 5 rows per 17-frame block. The bootstrap rows
        # have a different encoding structure than block rows, so the lock
        # covers the first FULL block (rows 2..6 <- prev rows -5:, i.e. new
        # frames 5..21 replay prev frames -17..-1) and the free 5-frame head
        # is warmup that gets trimmed. Audio latent runs at 40 latent-fps
        # with uniform structure: lock the whole head (cols 0..36 <- prev
        # cols -37:, ~0.92s) so no arbitrary audio precedes the replay.
        _HO_ROWS = 5 if handoff_depth == "block" else 2
        _HO_R0 = 2 if handoff_depth == "block" else 0
        _HO_ACOLS = 37 if handoff_depth == "block" else 8
        _OV_HO = 22 if handoff_depth == "block" else 5
        _HO_GUARD = 16         # onset-guard cols (0.4s of locked room tone)
        _ho_v = _ho_a = None   # previous shot's tail latents
        _ho_taper_src = None   # last row, for the post-lock pose taper
        _ho_guard = None       # encoded room tone for the onset guard
        _ho_wav_tail = None    # previous shot's tail waveform (fidelity log)
        _house_frame = None    # shot 1 frame 0: the canonical framing
        _spine = None          # encoded audio spine (guide_audio)
        # per-join output trim, by mode - also the spine's per-shot stride
        # context_pin trims a fixed 22-frame regenerated head per join and
        # flf_chain drops the 1 duplicated boundary frame - both MUST be in
        # this table or the audio-spine stride walks ahead of the picture
        # by the trim amount per join (cold-read verified 2026-08-10).
        _TRIM = {"cut": 0, "seamless": 1, "first_frame": 1, "flf_chain": 1,
                 "seamless_tail": _OV, "latent_handoff": _OV_HO,
                 "context_pin": 22}.get(continuity, 0)
        if guide_audio is not None:
            _gw3, _g_sr = _wav_for_vae(audio_vae, guide_audio, "audio spine")
            _spine = audio_vae.encode(_gw3.movedim(1, -1)).detach()
            print("[H3Memory] audio spine: %d cols (%.1fs) - every shot's "
                  "audio is locked to a slice of it, so the VOICE cannot "
                  "change between shots (fl2va has no reference rows to "
                  "carry a voice; this is how you keep one performance)."
                  % (_spine.shape[-1],
                     _gw3.shape[-1] / float(_g_sr)),
                  flush=True)
            if two_pass_upscale:
                # the spine lock lives in a predict_noise patch built around
                # ONE guider; pass 2 runs its own guider, so half the
                # trajectory would sample unlocked audio and the voice would
                # move exactly where the spine exists to hold it still
                raise ValueError(
                    "two_pass_upscale cannot be combined with an audio spine "
                    "(guide_audio). The spine locks audio through every "
                    "sampling step of a single trajectory; a two-pass render "
                    "is two trajectories. Disconnect guide_audio, or turn "
                    "two_pass_upscale off.")
        _cc_mu = _cc_cov = None  # house colour stats (shot 1 settled tail)
        _cp_prev = None   # context_pin: previous shot's full AV latent
        _rp_gain = 1.15   # refresh_pin: running per-hop gain estimate (seed =
                          # the measured 1.13-1.15x hop of 2026-08-17)
        _rp_div_last = 1.0  # refresh_pin: divisor actually applied to the
                            # LAST pin - v5.1: the EMA must track the model's
                            # RAW gain (measured x divisor), not the residual
                            # (A/B 2026-08-23: raw gain is stable at ~1.16
                            # while the residual-chasing EMA under-dosed)
        _rp_ref = None    # refresh_pin: house texture in THIS block's units
        _pin_sig0 = None  # first pin's sigma - the renorm anchor
        _pin_hf0 = None   # first pin's fine-detail energy - the flatten_pin anchor
        _cg_last_raw = None  # previous shot's raw tail texture (pixel domain)
        _cp_trim = 0

        if bank_pinned == 0 and n > 4:
            print("[H3Memory] WARNING: bank_pinned=0 on a %d-shot chain. "
                  "With no pinned slot the conditioning is pure recency - "
                  "each shot hears only the one before it - and audio "
                  "COLLAPSES: measured 84-92%% loss of 4-10 kHz energy by "
                  "shot 8 (five-arm A/B, 2026-08-11). Set bank_pinned=1, "
                  "and keep chains short if the voice matters." % n, flush=True)
        print(f"[H3Memory] JoyEcho-style memory bank: no keyframe, "
              f"{bank_pinned} pinned + {cap - min(bank_pinned, cap)} recent "
              f"slot(s), {_jb_grid(bank_clip_frames)}f clips. Needs a ref2va "
              f"checkpoint.", flush=True)

        for si, prompt in enumerate(shots):
            if two_pass_upscale:
                latent, frame_count = mmh3._empty_av_latent(
                    _tp_w1, _tp_h1, frames_per_shot)
                _p1v, _p1n = _tp_tr.extract_tensor(latent["samples"])
                _tp_lat_h1 = int(_p1v[0].shape[-2])
                _tp_lat_w1 = int(_p1v[0].shape[-1])
            else:
                latent, frame_count = mmh3._empty_av_latent(width, height,
                                                            frames_per_shot)
            ref_items, ref_blocks = [], []
            kf_vision = []     # first_frame mode: images -> vision tokens

            # operator-supplied refs go FIRST and never move: the bank grows
            # from shot to shot, so anything appended after it would change
            # <Picture n> / <Audio n> numbering mid-chain and break the
            # prompt's bindings. Items and blocks stay in the same sequence.
            for _it, _bl in zip(ref_image_items, ref_image_blocks):
                ref_items.append(_it)
                ref_blocks.append(_bl)
            if voice_block is not None:
                ref_items.append({"type": "audio"})
                ref_blocks.append(voice_block)

            # identity reference image(s) - JoyEcho seeds identity, the bank
            # carries it afterwards
            if start_image is not None and anchor_frames > 0:
                img = start_image[:1]
                ih, iw = int(img.shape[1]), int(img.shape[2])
                import math as _math
                sc = min(1.0, _math.sqrt((width * height) / max(iw * ih, 1)))
                tw = max(32, round(iw * sc / 32) * 32)
                th = max(32, round(ih * sc / 32) * 32)
                rz = mmh3._resize(img, tw, th, "disabled")
                ref_items.append({"type": "image", "data": rz})
                ref_blocks.append({"kind": "image", "latent_h": th // 16,
                                   "latent_w": tw // 16,
                                   "latent": video_vae.encode(rz)})

            # bank slots -> video_audio references, built the way core does.
            # first_frame mode runs on an fl2va checkpoint, which has no
            # reference rows - the hand-off frame carries continuity.
            #
            # 2.2.4: a user-supplied reference_video is prepended as just
            # another (frames, audio) pair, so it travels this exact proven
            # path instead of a parallel one. H3 pairs every video reference
            # with an audio reference, so silence is synthesised when the user
            # has none - stereo, because the ref audio encoder wants two
            # channels, and any rate is fine since _encode_ref_audio resamples.
            _extra_clips = []
            if reference_video is not None and int(reference_video.shape[0]):
                _rv_a = reference_video_audio
                _rv_n = int(reference_video.shape[0])
                if _rv_a is None:
                    import torch as _t
                    _sr = 32000
                    _dur = _rv_n / float(mmh3.FPS)
                    _rv_a = {"waveform": _t.zeros(1, 2, max(1, int(_dur * _sr))),
                             "sample_rate": _sr}
                    if si == 1:
                        print("[H3Memory] reference_video: %d frame(s), no "
                              "audio supplied - pairing with silence" % _rv_n,
                              flush=True)
                elif si == 1:
                    print("[H3Memory] reference_video: %d frame(s) + audio"
                          % _rv_n, flush=True)
                if si == 1 and _rv_n > 64:
                    print("[H3Memory] reference_video is %d frames; it is "
                          "subsampled to 2 fps but that is still a lot of "
                          "reference tokens on EVERY step. Run it through "
                          "H3ReferenceVideo to trim." % _rv_n, flush=True)
                _extra_clips.append((reference_video, _rv_a))
            for clip_frames, clip_audio in (
                    _extra_clips
                    + ([] if continuity == "first_frame" else bank.frames())):
                vh, vw = int(clip_frames.shape[1]), int(clip_frames.shape[2])
                cw, ch = mmh3.adapt_canvas(vw, vh)
                if vw * vh < cw * ch:
                    cw = max(32, round(vw / 32) * 32)
                    ch = max(32, round(vh / 32) * 32)
                fr = mmh3._resize(clip_frames, cw, ch, "disabled")
                fr = fr[:_jb_grid(fr.shape[0])]
                z = video_vae.encode(fr)
                a_lat, a_t = _mmh3_encode_ref_audio(audio_vae, clip_audio)
                # the soundtrack takes its own <Audio j>, emitted before <Video k>
                ref_items.append({"type": "audio"})
                idx = list(range(0, fr.shape[0], mmh3.FPS // 2))
                ref_items.append({"type": "video", "data": fr[idx],
                                  "timestamps": [i / 2.0 for i in range(len(idx))]})
                ref_blocks.append({"kind": "video_audio", "latent_t": z.shape[2],
                                   "latent_h": ch // 16, "latent_w": cw // 16,
                                   "ref_audio_t": a_t, "latent": z,
                                   "audio_latent": a_lat})

            keyframes = []
            if continuity == "flf_chain" and keyframe_images is None:
                # a silent no-op here renders a full unanchored chain and
                # the operator finds out hours later - fail loudly instead
                raise ValueError(
                    "continuity=flf_chain but keyframe_images is empty. "
                    "Wire N+1 boundary plates (and enable their gate) for "
                    "N shots, or switch continuity to context_pin.")
            if continuity == "flf_chain" and keyframe_images is not None:
                # TRUE FFLF: shot i runs between boundary image i and i+1.
                # The join is ONE shared picture used as the end of one shot
                # and the start of the next, so there is nothing to drift
                # and nothing to colour-correct at the boundary.
                _n_kf = keyframe_images.shape[0]
                _a = keyframe_images[min(si, _n_kf - 1):min(si, _n_kf - 1) + 1]
                _kf_a = mmh3._resize(_a, width, height, "disabled")
                kf_vision.append(_kf_a)
                keyframes.append({"resolved_frame_index": 0, "image": _kf_a})
                if si + 1 < _n_kf:
                    _b = keyframe_images[si + 1:si + 2]
                    _kf_b = mmh3._resize(_b, width, height, "disabled")
                    kf_vision.append(_kf_b)
                    keyframes.append(
                        {"resolved_frame_index": frame_count - 1,
                         "image": _kf_b})
                    # the documented FL2VA alignment instruction, first line
                    # (em dash and wording verbatim from the base guide)
                    prompt = (
                        "How the reference pictures align with the target "
                        "video — Picture 1 (from Shot 1) aligns with the "
                        "0.00-second mark of the target video; Picture 2 "
                        "(from Shot 1) aligns with the %.2f-second mark of "
                        "the target video.\n\n" % (frame_count / 24.0)
                    ) + prompt
                elif not __import__("os").environ.get("H3_NO_KF_ALIGN"):
                    # final plate: one keyframe at frame 0 - the documented
                    # I2VA alignment line, same first-line contract
                    prompt = ("For the target video, at 0.00 seconds into "
                              "the target video, <Picture 1> (from [Shot 1]) "
                              "is fully referenced.\n\n") + prompt
                print("[H3Memory] FFLF shot %d: pinned between boundary "
                      "keyframes %d and %d" % (si + 1, si, min(si + 1,
                                                               _n_kf - 1)),
                      flush=True)
            elif last_tail is not None and continuity == "first_frame":
                # the model's own hand-off: the previous last frame goes in
                # BOTH ways the stock Image-to-Video node sends it - vision
                # tokens through the text encoder AND the frame-0 keyframe
                # latent. A keyframe latent alone is a weak hint; the vision
                # path is the conditioning fl2va was trained on.
                kf_img = mmh3._resize(last_tail[-1:], width, height,
                                      "disabled")
                kf_vision.append(kf_img)
                keyframes.append({"resolved_frame_index": 0,
                                  "image": kf_img})
                # the documented I2VA alignment line for the handed-off frame
                if not __import__("os").environ.get("H3_NO_KF_ALIGN"):
                    prompt = ("For the target video, at 0.00 seconds into "
                              "the target video, <Picture 1> (from [Shot 1]) "
                              "is fully referenced.\n\n") + prompt
            elif last_tail is not None and continuity == "seamless":
                kf_img = mmh3._resize(last_tail[-1:], width, height, "disabled")
                keyframes.append({"resolved_frame_index": 0, "image": kf_img})
            elif last_tail is not None and continuity == "seamless_tail":
                # tail bracket: previous pixel frames -9/-5/-1 pinned at
                # keyframe indices 0/4/8 (one per latent block on the 4x
                # temporal grid). The join is over-determined: position,
                # exposure and velocity are all specified by real frames.
                #
                # Indices 4 and 8 are INTERIOR anchors, which stock comfy
                # rejects. Our layout patch generalises the math, but it
                # stands down when ComfyUI-H3-Motion-Context is installed -
                # and MC's layout patch only serves rows carrying its own
                # marker, so THESE keyframes fall through to stock and the
                # chain dies mid-render with "only first/last keyframe
                # anchors are supported" (user-reported, 2026-08-11). Fail
                # BEFORE any sampling, with the fix in the message.
                try:
                    from .h3_interior_patch import (_motion_context_present,
                                                    ensure_interior_keyframes)
                except ImportError:
                    try:
                        from h3_interior_patch import (
                            _motion_context_present, ensure_interior_keyframes)
                    except ImportError:
                        # loose install: the module sits beside this file but
                        # is not importable by name (same fallback as
                        # h3_keyframes.py)
                        import importlib.util as _ilu
                        import os as _os
                        _p = _os.path.join(_os.path.dirname(
                            _os.path.abspath(__file__)),
                            "h3_interior_patch.py")
                        _s = _ilu.spec_from_file_location(
                            "h3_interior_patch", _p)
                        _m = _ilu.module_from_spec(_s)
                        _s.loader.exec_module(_m)
                        _motion_context_present = _m._motion_context_present
                        ensure_interior_keyframes = _m.ensure_interior_keyframes
                _mc_pack = _motion_context_present()
                if _mc_pack:
                    raise ValueError(
                        "continuity=seamless_tail needs interior keyframe "
                        f"anchors, and {_mc_pack} owns that patch site but "
                        "only serves its own nodes - the chain would crash "
                        "mid-render. Use continuity=context_pin (better, and "
                        "it is what that pack is for), or first_frame, or "
                        "remove that pack to use seamless_tail.")
                _ik_ok, _ik_msg = ensure_interior_keyframes(verbose=False)
                if not _ik_ok:
                    raise ValueError(
                        "continuity=seamless_tail needs interior keyframe "
                        f"anchors and the layout patch failed: {_ik_msg}. "
                        "Use continuity=first_frame or context_pin instead.")
                for j in range(_TAIL_K + 1):
                    pi = -(1 + 4 * (_TAIL_K - j))          # -9, -5, -1
                    src = last_tail[pi:pi + 1] if pi != -1 else last_tail[-1:]
                    kf_img = mmh3._resize(src, width, height, "disabled")
                    _dbg_pins.append((4 * j, kf_img[0].detach().cpu().clone()))
                    keyframes.append({"resolved_frame_index": 4 * j,
                                      "image": kf_img})
            if (last_tail is not None and continuity == "latent_handoff"
                    and handoff_depth == "bootstrap"):
                # bootstrap depth: the 2-row latent lock is a weak video
                # anchor - back it with a frame-0 keyframe pin of the
                # previous last frame (soft, but end_anchor and the bank
                # carry the rest)
                kf_img = mmh3._resize(last_tail[-1:], width, height,
                                      "disabled")
                keyframes.append({"resolved_frame_index": 0,
                                  "image": kf_img})
            if (end_anchor and continuity == "first_frame" and si > 0
                    and _house_frame is not None):
                # fl2va reads first+last as "travel from A to B" and
                # invents a camera move to fill the middle (render-verified:
                # shot 2 pushed into an extreme close-up and back out).
                # Hand it ONLY the first frame and let it continue.
                if si == 1:
                    print("[H3Memory] end_anchor ignored in first_frame "
                          "mode: a last-frame pin makes fl2va plan a camera "
                          "MOVE between the two frames. Control drift with "
                          "prompt wording instead.", flush=True)
            elif end_anchor and _house_frame is not None and si > 0:
                # return-to-house DOUBLE pin at the shot's tail: closes the
                # compounding push-in creep so the next join inherits a tail
                # the text agrees with. One pin at the last frame gets
                # outvoted by committed motion (render-verified: a tail
                # lean-in ran straight through it); a second pin half a
                # second earlier makes the hold bracket-strength and reads
                # as her settling for the beat. Rides through the same
                # encode loop below, so join_anchor_noise applies too.
                kf_img = mmh3._resize(_house_frame, width, height, "disabled")
                keyframes.append(
                    {"resolved_frame_index": frames_per_shot - 1,
                     "image": kf_img})
                if frames_per_shot > 21:
                    keyframes.append(
                        {"resolved_frame_index": frames_per_shot - 13,
                         "image": kf_img})

            print("[H3Memory] shot %d/%d (%df @ %dx%d) | bank %s%s"
                  % (si + 1, n, frames_per_shot, width, height, bank.describe(),
                     " + identity ref" if (start_image is not None
                                           and anchor_frames > 0) else ""),
                  flush=True)

            # Symmetric eviction: free the DiT before the encoder loads.
            # Without this the previous shot's DiT is still resident when a
            # 15.69 GB encoder is requested, and on a 24 GB card it cannot fit
            # beside it - ComfyUI streams the encoder off disk and the reader
            # eventually fails (hostbuf_file_reader_read). Costs nothing: the
            # DiT is reloaded every shot regardless.
            if si > 0:
                try:
                    import comfy.model_management as _mm2
                    _d2 = _mm2.get_torch_device()
                    _b4 = _mm2.get_free_memory(_d2) / (1024 ** 3)
                    _mm2.free_memory(_mm2.get_total_memory(_d2) * 0.9, _d2)
                    _mm2.soft_empty_cache()
                    _af = _mm2.get_free_memory(_d2) / (1024 ** 3)
                    print("[H3Memory] DiT evicted before the text encoder; "
                          "%.1f -> %.1f GB free" % (_b4, _af), flush=True)
                except Exception as _e:
                    print("[H3Memory] could not evict before the encoder (%s) - "
                          "if the encoder streams from disk this is why" % _e,
                          flush=True)

            if kf_vision:
                tokens = clip.tokenize(prompt, images=kf_vision)
            elif ref_items:
                _n_img = sum(1 for i in ref_items if i["type"] == "image")
                _groups = _parse_ref_groups(reference_subjects, _n_img)
                _n_aud = sum(1 for i in ref_items if i["type"] == "audio")
                _n_vid = sum(1 for i in ref_items if i["type"] == "video")
                # H3_LEGACY_SECTION_ORDER=1 restores the pre-2026-08-21
                # behaviour (label sections concatenated after the prose) so
                # the ordering change can be A/B'd on one seed.
                _legacy_order = bool(__import__("os").environ.get(
                    "H3_LEGACY_SECTION_ORDER"))
                _parts = _subject_defs(_n_img, _n_aud, _n_vid,
                                       image_subjects=_groups,
                                       return_parts=True)
                # "" when there are no references at all, else a 3-tuple.
                _parts = _parts if isinstance(_parts, tuple) else None
                _sd = "\n".join(_parts) if _parts else ""
                if _sd:
                    if si == 1:
                        print("[H3Memory] subject_definitions added for %d "
                              "reference item(s) - the model is now told "
                              "what the refs ARE and to preserve identity, "
                              "room, colour and voice timbre"
                              % len(ref_items), flush=True)
                        if _groups:
                            print("[H3Memory] reference_subjects %r -> %d "
                                  "distinct subject(s) across %d picture(s); "
                                  "each is declared its own person instead of "
                                  "all being <Subject 1>"
                                  % (reference_subjects, max(_groups), _n_img),
                                  flush=True)
                        elif _n_img > 1:
                            print("[H3Memory] %d reference pictures are all "
                                  "declared <Subject 1> (one person). If they "
                                  "show DIFFERENT people, set reference_"
                                  "subjects (e.g. '3,3') or they will blend."
                                  % _n_img, flush=True)
                    if _legacy_order:
                        prompt = prompt.rstrip() + "\n" + _sd
                    else:
                        prompt = _compose_ref2va(_parts[0], _parts[1],
                                                 _parts[2], prompt)
                        if si == 1:
                            print("[H3Memory] Ref2VA sections in guide order: "
                                  "subject_definitions, summary, "
                                  "retention_analysis, detailed_description, "
                                  "non_diegetic_music: N/A "
                                  "(H3_LEGACY_SECTION_ORDER=1 to revert)",
                                  flush=True)
                tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            else:
                tokens = clip.tokenize(prompt)
            cond = clip.encode_from_tokens_scheduled(tokens)
            cond_hi = cond if two_pass_upscale else None

            def _encode_kfs(kfs, tw, th):
                """Encode a keyframe list onto a tw x th grid, in place."""
                for kf_i, kf_ in enumerate(kfs):
                    z = video_vae.encode(
                        mmh3._resize(kf_.pop("image"), tw, th, "disabled"))
                    if join_anchor_noise > 0:
                        # noised clean condition: the model must not treat its
                        # own output as pristine (that is the 1.2x ratchet).
                        # Seeded so same-seed A/B arms stay clean.
                        g = torch.Generator(device=z.device).manual_seed(
                            (shot_seed ^ 0x5EED) + kf_i)
                        t_add = float(join_anchor_noise)
                        z = (1.0 - t_add) * z + t_add * torch.randn(
                            z.shape, generator=g, device=z.device, dtype=z.dtype)
                    kf_["latent"] = z
                return kfs

            if keyframes:
                # two-pass pins the SAME pixels on both grids - encoding twice
                # is cheap and exact, where resampling a latent is neither
                kfs_hi = ([{"resolved_frame_index": k["resolved_frame_index"],
                            "image": k["image"]} for k in keyframes]
                          if two_pass_upscale else None)
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_keyframes": _encode_kfs(
                        keyframes,
                        _tp_w1 if two_pass_upscale else width,
                        _tp_h1 if two_pass_upscale else height),
                    "minimax_frame_count": frame_count,
                })
                if kfs_hi is not None:
                    cond_hi = node_helpers.conditioning_set_values(cond_hi, {
                        "minimax_keyframes": _encode_kfs(kfs_hi, width, height),
                        "minimax_frame_count": frame_count,
                    })
            if ref_blocks:
                cond = node_helpers.conditioning_set_values(
                    cond, {"minimax_refs": ref_blocks})
                if cond_hi is not None:
                    cond_hi = node_helpers.conditioning_set_values(
                        cond_hi, {"minimax_refs": ref_blocks})

            # context_pin: reuse the Motion-Context node as a library via
            # the registry - OUR features (bank, colour levels, join fx)
            # stay; THEIR mechanism (interior latent pin + timeline audio
            # ref + payload coexistence patches) rides on the conditioning.
            _cp_trim = 0
            if (continuity == "context_pin" and si > 0
                    and _cp_prev is not None):
                import nodes as _nodes_mod
                _mc_cls = _nodes_mod.NODE_CLASS_MAPPINGS.get(
                    "MiniMaxH3MotionContext")
                if _mc_cls is None:
                    # ethanfel's ComfyUI-MiniMaxH3-Contex-Loop is a fork of the
                    # same project, but it deliberately does NOT re-register
                    # MiniMaxH3MotionContext - that id stays with upstream, by
                    # its own design. Someone who installed the fork INSTEAD of
                    # upstream has a pack full of H3 chain nodes and still no
                    # context_pin, and the old message sent them looking for a
                    # pack they thought they already had (user-reported).
                    _fork = any(
                        k in _nodes_mod.NODE_CLASS_MAPPINGS
                        for k in ("MiniMaxH3ChainLoopStart",
                                  "MiniMaxH3LoopTrim",
                                  "MiniMaxH3ChainAssemble"))
                    raise RuntimeError(
                        "continuity=context_pin needs the "
                        "ComfyUI-H3-Motion-Context pack by NikoDemon80 "
                        "(github.com/NikoDemon80/ComfyUI-H3-Motion-Context) - "
                        "it provides the MiniMaxH3MotionContext node."
                        + (" You appear to have ethanfel's "
                           "ComfyUI-MiniMaxH3-Contex-Loop fork installed. That "
                           "is a COMPLEMENT, not a replacement: it "
                           "deliberately leaves the MiniMaxH3MotionContext id "
                           "to upstream, so install NikoDemon80's pack as well "
                           "- the two are designed to coexist and this pack "
                           "works with either one's runtime patches."
                           if _fork else
                           " If you installed a fork instead, note that forks "
                           "may leave that node id to upstream on purpose."))
                # The pin is raw latents from the PREVIOUS shot, sampled at
                # full resolution. Pass 1 runs on a smaller grid, so the pin
                # has to be resampled onto that grid or the shapes simply do
                # not meet - which is what used to make this combination an
                # error. Pass 2 is unpinned by construction (cond_hi is
                # snapshotted before this line), so it inherits the seam
                # through the upscaled pass-1 latent rather than re-deriving
                # it, and never sees a pin at the wrong size.
                _pin_src = _cp_prev
                if two_pass_upscale and _cp_prev is not None:
                    _pin_src = _upscale_av_exact(_tp_tr, _cp_prev,
                                                 _tp_lat_h1, _tp_lat_w1)
                    print("[H3Memory] two-pass: pin resampled to the pass-1 "
                          "grid (%dx%d latent)" % (_tp_lat_h1, _tp_lat_w1),
                          flush=True)
                _pf = str(pin_frames) if str(pin_frames) in (
                    "5", "22", "39", "56") else "22"
                if (pin_renorm and isinstance(_pin_src, dict)
                        and "samples" in _pin_src):
                    # BEFORE the noise: pin_noise is scaled to sigma, so the
                    # renorm has to land first or it would be measuring a
                    # sigma the noise is about to change.
                    _zr = _pin_src["samples"]
                    _v0 = (_zr.unbind()[0] if getattr(_zr, "is_nested", False)
                           else _zr)
                    _sg = float(_v0.float().std())
                    if _pin_sig0 is None:
                        _pin_sig0 = _sg
                        print("[H3Memory] context_pin: pin sigma anchor %.4f "
                              "(shot 1)" % _sg, flush=True)
                    elif _sg > 1e-6:
                        _k = _pin_sig0 / _sg
                        if abs(_k - 1.0) > 1e-4:
                            _pin_src = dict(_pin_src)
                            if getattr(_zr, "is_nested", False):
                                import comfy.nested_tensor as _nt2
                                _cc = list(_zr.unbind())
                                _cc[0] = (_cc[0].float() * _k).to(_cc[0].dtype)
                                _pin_src["samples"] = _nt2.NestedTensor(_cc)
                            else:
                                _pin_src["samples"] = (
                                    _zr.float() * _k).to(_zr.dtype)
                            print("[H3Memory] context_pin: pin renormed "
                                  "sigma %.4f -> %.4f (x%.4f)"
                                  % (_sg, _pin_sig0, _k), flush=True)

                if (chain_gain_control == "flatten_pin"
                        and isinstance(_pin_src, dict)
                        and "samples" in _pin_src):
                    # THE OTHER HALF OF FLATTEN. chain_gain_control=flatten
                    # levels decoded frames and the bank, but context_pin
                    # carries the previous shot's RAW LATENTS - and their
                    # accreted fine detail is what the next shot conditions
                    # on, so the ratchet rode the pin regardless (measured
                    # 2026-08-17 at 736x1280 with flatten ON: x1.13-1.15 per
                    # hop, +66% texture by window 5, +77% by window 6 of an
                    # extend take). Level the pin's high-frequency energy to
                    # shot 1's tail before it is pinned: hp = z - mean3x3(z)
                    # over (H,W); gain = sqrt(anchor/current), clamped to
                    # [0.5, 1] so it only ever softens invented detail, never
                    # sharpens. Video half only. After renorm (sigma anchor),
                    # before pin_noise.
                    import torch.nn.functional as _F
                    _zr = _pin_src["samples"]
                    _nested = getattr(_zr, "is_nested", False)
                    _v0 = _zr.unbind()[0] if _nested else _zr
                    _vf = _v0.float()
                    _b, _c, _tt, _hh, _ww = _vf.shape
                    _flat = _vf.permute(0, 2, 1, 3, 4).reshape(-1, _c, _hh, _ww)
                    _low = _F.avg_pool2d(_flat, 3, stride=1, padding=1,
                                         count_include_pad=False)
                    _hp = _flat - _low
                    # measure on the tail (what gets pinned): last ~8 latent frames
                    _tail = max(1, min(_tt, 8))
                    _hp_t = _hp.reshape(_b, _tt, _c, _hh, _ww)[:, -_tail:]
                    _e = float(_hp_t.pow(2).mean())
                    if _pin_hf0 is None:
                        _pin_hf0 = _e
                        print("[H3Memory] context_pin: fine-detail anchor "
                              "%.5f (shot 1 tail)" % _e, flush=True)
                    elif _e > 1e-12:
                        _g_lat = (_pin_hf0 / _e) ** 0.5
                        # The latent high-pass under-reads what the VAE turns
                        # into visible sharpness (measured 2026-08-17: latent
                        # HF +7.5% at a hop where decoded texture rose +19%),
                        # so also take the PIXEL-domain ratio the frame flatten
                        # measured on this shot's raw tail - Laplacian variance
                        # is amplitude^2, so amplitude gain = sqrt(ref/raw) -
                        # and apply the stronger (smaller) of the two.
                        _g_pix = 1.0
                        try:
                            if _cg_ref and _cg_last_raw and _cg_last_raw > _cg_ref:
                                _g_pix = (float(_cg_ref) / float(_cg_last_raw)) ** 0.5
                        except Exception:
                            _g_pix = 1.0
                        _g = max(0.5, min(1.0, _g_lat, _g_pix))
                        print("[H3Memory] context_pin: flatten_pin gains - "
                              "latent %.3f, pixel %.3f -> using %.3f"
                              % (_g_lat, _g_pix, _g), flush=True)
                        if _g < 0.999:
                            _newv = (_low + _hp * _g).reshape(
                                _b, _tt, _c, _hh, _ww).permute(0, 2, 1, 3, 4)
                            _newv = _newv.to(_v0.dtype)
                            _pin_src = dict(_pin_src)
                            if _nested:
                                import comfy.nested_tensor as _nt3
                                _cc = list(_zr.unbind()); _cc[0] = _newv
                                _pin_src["samples"] = _nt3.NestedTensor(_cc)
                            else:
                                _pin_src["samples"] = _newv
                            print("[H3Memory] context_pin: flatten_pin - pin "
                                  "fine-detail energy %.5f vs anchor %.5f -> "
                                  "high-pass x%.3f (only softens)"
                                  % (_e, _pin_hf0, _g), flush=True)
                        else:
                            print("[H3Memory] context_pin: flatten_pin - pin "
                                  "already at anchor (%.5f vs %.5f), untouched"
                                  % (_e, _pin_hf0), flush=True)

                if (pin_noise > 0 and isinstance(_pin_src, dict)
                        and "samples" in _pin_src):
                    # noised clean condition, applied to the carrier. The pin
                    # is this model's own output; left pristine the model
                    # treats it as ground truth and adds detail on top of
                    # detail, once per hop. Seeded so same-seed A/B arms are
                    # comparable.
                    #
                    # The pin is a NestedTensor - [0] video, [-1] audio - so
                    # it has to be unbound first, and only the VIDEO half is
                    # noised. Noising the audio component dulls the voice,
                    # which is the drift we already fight elsewhere.
                    _t = float(pin_noise)
                    _sig = []

                    def _noise_one(_z):
                        # Variance-preserving, scaled to the latent's OWN
                        # standard deviation. Unit-variance noise would make
                        # this dial resolution- and content-dependent: the
                        # latent's magnitude is not fixed, so a fixed noise
                        # magnitude is a different SNR at every render size.
                        # sqrt(1-t^2) keeps total variance at sigma^2 rather
                        # than attenuating the pin, so t changes ONLY the
                        # noise fraction and not the pin's strength.
                        _g = torch.Generator(device=_z.device).manual_seed(
                            shot_seed ^ 0x91EE)
                        _s = _z.float().std()
                        _sig.append(float(_s))
                        _n = torch.randn(_z.shape, generator=_g,
                                         device=_z.device, dtype=torch.float32)
                        _out = ((1.0 - _t * _t) ** 0.5) * _z.float() \
                            + _t * _s * _n
                        return _out.to(_z.dtype)

                    _zs = _pin_src["samples"]
                    _pin_src = dict(_pin_src)
                    if getattr(_zs, "is_nested", False):
                        import comfy.nested_tensor as _nt
                        _cps = list(_zs.unbind())
                        _cps[0] = _noise_one(_cps[0])
                        _pin_src["samples"] = _nt.NestedTensor(_cps)
                        _what = "video half of the AV pin"
                    else:
                        _pin_src["samples"] = _noise_one(_zs)
                        _what = "pin"
                    print("[H3Memory] context_pin: %s noised %.3f of its "
                          "own sigma=%.4f (anti-ratchet, variance-preserving)"
                          % (_what, _t, _sig[0] if _sig else float("nan")),
                          flush=True)
                _apf = int(audio_pin_frames or 0) or int(_pf)
                if _apf != int(_pf):
                    print("[H3Memory] context_pin: audio reference window %d "
                          "frames (%.1f s), picture pin %s frames - the head "
                          "trim follows the picture pin only." %
                          (_apf, _apf / 24.0, _pf), flush=True)
                cond, _cp_trim = _mc_cls().apply(
                    conditioning=cond, vae=video_vae, latent=latent,
                    context_length=_pf, audio_context_length=_apf,
                    context_latent=_pin_src)
                print("[H3Memory] context_pin: previous shot's tail pinned "
                      "as raw latents (%sf video + %sf audio ref, trim %d "
                      "on decode)" % (_pf, _pf, _cp_trim), flush=True)

            # issue #8: separate TE device -> nothing to reclaim, keep it hot
            _te_dev = getattr(clip.patcher, "load_device", None)
            _dit_dev = getattr(model, "load_device", None)
            if (_te_dev is not None and _dit_dev is not None
                    and str(_te_dev) != str(_dit_dev)):
                if si == 0:
                    print(f"[H3Memory] TE on {_te_dev}, DiT on {_dit_dev} - "
                          f"separate devices, TE stays resident.", flush=True)
                    # A remote TE loads nothing locally - but models left over
                    # from the PREVIOUS run may still be resident, and skipping
                    # the sweep hands AutoReserve a card that looks nearly full
                    # (measured 2026-08-21: a stale 15 GB TE -> TIGHT path ->
                    # driver spill -> shot 1 at 65 s/it vs 27 s/it clean).
                    try:
                        _dev0 = _mm.get_torch_device()
                        _b40 = _mm.get_free_memory(_dev0) / (1024 ** 3)
                        try:
                            _mm.unload_all_models()
                        except Exception:
                            pass
                        _mm.free_memory(_mm.get_total_memory(_dev0) * 0.9, _dev0)
                        _mm.soft_empty_cache()
                        _af0 = _mm.get_free_memory(_dev0) / (1024 ** 3)
                        if _af0 - _b40 > 0.5:
                            print("[H3Memory] cleared %.1f GB of leftovers "
                                  "from earlier runs before the first DiT "
                                  "load." % (_af0 - _b40), flush=True)
                    except Exception:
                        pass
            else:
                try:
                    clip.patcher.model.to(_mm.text_encoder_offload_device())
                except Exception:
                    pass
                # The VAEs are dead weight during sampling - encode already
                # happened, decode has not. They were staying resident (5.5 GB
                # between them) while the DiT loaded, and on shot 2+ the larger
                # conditioning payload raises the activation reserve enough
                # that the DiT then misses a FULL load by a few hundred MB.
                # Measured: shot 1 full load at 18.5 s/it, shot 2 with 399 MB
                # offloaded at 267 s/it - a 14x collapse for 2% of the weights,
                # because every offloaded layer streams over PCIe every step.
                _vfreed = 0
                for _v in (video_vae, audio_vae):
                    if getattr(_v, "patcher", None) is not None:
                        _vfreed += 1
                try:
                    _dev = _mm.get_torch_device()
                    # Unload through model management, NOT module.to(). On the
                    # DynamicVRAM path (0.33+) weights live in a comfy_aimdo
                    # vbar arena; a bare .to(cpu) moves the module and leaves
                    # the arena's pages resident (~8 GB observed on a 24 GB
                    # card), after which free_memory reports "0 models
                    # unloaded" because nothing LOOKS loaded any more. Proper
                    # unload goes through the patcher and tears the arena down.
                    # The DiT is not loaded yet, and TE/VAEs are re-requested
                    # every shot anyway, so this costs nothing extra.
                    try:
                        _mm.unload_all_models()
                    except Exception as _ue:
                        print("[H3Memory] unload_all_models failed (%s) - "
                              "falling back to module moves" % _ue, flush=True)
                        try:
                            clip.patcher.model.to(_mm.text_encoder_offload_device())
                        except Exception:
                            pass
                        for _v in (video_vae, audio_vae):
                            try:
                                _v.patcher.model.to(_mm.vae_offload_device())
                            except Exception:
                                pass
                    _mm.free_memory(_mm.get_total_memory(_dev) * 0.9, _dev)
                    _mm.soft_empty_cache()
                    # Name whoever is still holding the card. This is the
                    # instrument that settles the "mystery 8 GB": if the vbar
                    # theory is right this prints nothing on 0.33.1 any more;
                    # if it is wrong, the culprit is named instead of guessed.
                    try:
                        import torch as _t
                        _freeb, _totb = _t.cuda.mem_get_info(_dev.index
                                        if hasattr(_dev, "index") else 0)
                        if (_totb - _freeb) > _totb * 0.25:
                            print("[H3Memory] post-evict residents: driver "
                                  "%.1f GB held | torch alloc %.1f reserved %.1f"
                                  % ((_totb - _freeb) / 2**30,
                                     _t.cuda.memory_allocated(_dev) / 2**30,
                                     _t.cuda.memory_reserved(_dev) / 2**30),
                                  flush=True)
                            for _lm in list(getattr(_mm, "current_loaded_models", [])):
                                try:
                                    _sz = _lm.model.loaded_size()
                                    if _sz > 256 * 1024**2:
                                        print("[H3Memory]   resident: %s  %.2f GB"
                                              % (_lm.model.model.__class__.__name__,
                                                 _sz / 2**30), flush=True)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    print("[H3Memory] TE%s evicted; %.1f GB free for the DiT"
                          % (" + %d VAE(s)" % _vfreed if _vfreed else "",
                             _mm.get_free_memory(_dev) / (1024 ** 3)), flush=True)
                except Exception:
                    pass

            guider = ncs.BasicGuider().get_guider(model, cond)[0]
            guider_hi = (ncs.BasicGuider().get_guider(model, cond_hi)[0]
                         if two_pass_upscale else None)
            if (_spine is not None
                    or (continuity == "latent_handoff"
                        and _ho_v is not None)):
                # one denoise trajectory: every model call sees the previous
                # shot's tail latents, renoised to the CURRENT sigma, sitting
                # in the overlap slots of both streams. The prediction then
                # continues that state - motion, exposure, and the word that
                # was mid-air. Released below handoff_release so the final
                # detail steps can reconcile the boundary.
                # CRITICAL: comfy PACKS the nested AV latent into one flat
                # [B,1,N] tensor before sampling (CFGGuider.sample ->
                # pack_latents), so predict_noise receives the pack, never a
                # NestedTensor. The lock is written through reshaped views
                # of the pack. (An is_nested check here silently no-ops -
                # render-verified failure.)
                _ms_obj = model.get_model_object("model_sampling")
                # ComfyUI 0.32.0+ (ModelSamplingAV) carries the AUDIO half of
                # the pack SCALED onto the video schedule:
                # model_base.process_latent_in multiplies the audio slice by
                # shift/audio_shift - 12/3 = 4 for H3 - and divides it back on
                # the way out. Everything we inject below is in the stream's
                # NATIVE domain (straight from audio_vae.encode, or from a
                # sampler output that has already been divided back), so it
                # must be multiplied up before being written into x.
                # User-reported as "the Audio Spine outputs static" on 0.32.0,
                # with voice_ref - which never touches the sampler's latent -
                # fine on the same file. 0.30.0 has no such scaling and
                # getattr's 1.0 default leaves it byte-identical.
                try:
                    _asc = float(getattr(_ms_obj, "audio_scale", 1.0) or 1.0)
                except Exception:
                    _asc = 1.0
                if abs(_asc - 1.0) > 1e-6:
                    print("[H3Memory] audio injections scaled x%.3f onto the "
                          "sampler's audio domain (ModelSamplingAV)" % _asc,
                          flush=True)
                # THE TWO CLOCKS: H3 video rides the sampler's shift-12
                # sigma; the audio stream lives on a shift-3 schedule
                # internally. An audio lock injected at the VIDEO sigma is
                # 2.5-4x noisier than the model's timestep declares during
                # the plan-forming steps - the model reads it as hiss, not
                # content (source-verified: comfy/ldm/minimax/model.py).
                from comfy.ldm.minimax.model import time_shift_sigma as _tss
                _dm = getattr(getattr(model, "model", None),
                              "diffusion_model", None)
                _shv = float(getattr(_dm, "sigma_shift_video", 12.0))
                _sha = float(getattr(_dm, "sigma_shift_audio", 3.0))
                _orig_pn = guider.predict_noise
                _comps0 = latent["samples"].unbind()
                _vshape = tuple(_comps0[0].shape)
                _ashape = (tuple(_comps0[1].shape)
                           if len(_comps0) > 1 else None)
                _spine_seg = None
                if _spine is not None and _ashape is not None:
                    # this shot's time-slice of the spine: shots advance by
                    # (frames_per_shot - trim) in output time, and the trim
                    # depends on the continuity mode
                    _a0 = int(round(si * (frames_per_shot - _TRIM)
                                    / 24.0 * 40.0))
                    _a0 = max(0, min(_a0, max(0, _spine.shape[-1] - 1)))
                    _spine_seg = _spine[..., _a0:_a0 + _ashape[-1]]

                _Nv = 1
                for _d in _vshape[1:]:
                    _Nv *= _d
                _Na = 0
                if _ashape is not None:
                    _Na = 1
                    for _d in _ashape[1:]:
                        _Na *= _d

                _alock = bool(audio_lock) or _spine_seg is not None

                def _pn(x, timestep, model_options={}, seed=None,
                        _o=_orig_pn, _hv=_ho_v, _ha=_ho_a, _hg=_ho_guard,
                        _hs=_spine_seg, _ms=_ms_obj, _al=_alock,
                        _r0=_HO_R0, _tss=_tss, _shv=_shv, _sha=_sha,
                        _tp=int(handoff_taper), _tsrc=_ho_taper_src,
                        _rel=float(handoff_release), _vs=_vshape,
                        _as=_ashape, _Nv=_Nv, _Na=_Na, _asc=_asc,
                        _state={"logged": False}):
                    try:
                        sig = float(timestep.flatten()[0])
                    except Exception:
                        sig = float(timestep)
                    if (x.ndim == 3 and x.shape[1] == 1
                            and x.shape[2] >= _Nv + _Na):
                        try:
                            x = x.clone()
                            st = torch.tensor([sig], device=x.device,
                                              dtype=x.dtype)
                            sa = _tss(sig, _shv, _sha)
                            sta = torch.tensor([sa], device=x.device,
                                               dtype=x.dtype)
                            if sig > _rel and _hv is not None:
                                xv = x[:, 0, :_Nv].reshape(
                                    (x.shape[0],) + _vs[1:])
                                tv = _hv.to(device=x.device, dtype=x.dtype)
                                xv[:, :, _r0:_r0 + tv.shape[2]] = \
                                    _ms.noise_scaling(
                                        st, torch.randn_like(tv), tv)
                                if _tp > 0 and _tsrc is not None:
                                    # graded taper: bias the rows AFTER the
                                    # hard lock toward the previous tail at
                                    # linearly decaying strength, so the
                                    # pose has a ramp instead of a cliff
                                    # (a hard lock end re-plans the body -
                                    # measured 3x the room's discontinuity)
                                    _t0 = _r0 + tv.shape[2]
                                    _tn = min(_tp, xv.shape[2] - _t0)
                                    if _tn > 0:
                                        ts_ = _tsrc.to(device=x.device,
                                                       dtype=x.dtype)
                                        for _j in range(_tn):
                                            _w = (_tn - _j) / (_tn + 1.0)
                                            _tgt = _ms.noise_scaling(
                                                st, torch.randn_like(ts_),
                                                ts_)[:, :, 0]
                                            xv[:, :, _t0 + _j] = (
                                                (1.0 - _w) * xv[:, :, _t0 + _j]
                                                + _w * _tgt)
                            # AUDIO stays locked through EVERY step: video
                            # release exists so detail steps can reconcile
                            # texture, but audio content must be exact -
                            # released late steps can still move speech
                            # ONSETS into the replay window, and the trim
                            # then chops the line's opening words
                            # (review-verified: "If a civilization" and
                            # "And before you say it" both swallowed).
                            if _Na and _hs is not None:
                                # spine mode: the WHOLE audio stream is a
                                # locked slice of one continuous track -
                                # nothing left for the model to plan.
                                # Injected at the AUDIO clock.
                                xa = x[:, 0, _Nv:_Nv + _Na].reshape(
                                    (x.shape[0],) + _as[1:])
                                ts = _hs.to(device=x.device,
                                            dtype=x.dtype) * _asc
                                _nn = min(ts.shape[-1], xa.shape[-1])
                                xa[..., :_nn] = _ms.noise_scaling(
                                    sta, torch.randn_like(ts[..., :_nn]),
                                    ts[..., :_nn])
                            elif _Na and _al and _ha is not None:
                                xa = x[:, 0, _Nv:_Nv + _Na].reshape(
                                    (x.shape[0],) + _as[1:])
                                ta = _ha.to(device=x.device,
                                            dtype=x.dtype) * _asc
                                if _hg is not None:
                                    ta = torch.cat(
                                        [ta, _hg.to(device=x.device,
                                                    dtype=x.dtype) * _asc],
                                        dim=-1)
                                xa[..., :ta.shape[-1]] = _ms.noise_scaling(
                                    sta, torch.randn_like(ta), ta)
                            if not _state["logged"]:
                                _state["logged"] = True
                                print("[H3Memory] handoff injection ACTIVE "
                                      "(packed path, sigma %.3f)" % sig,
                                      flush=True)
                        except Exception as _e:
                            print("[H3Memory] handoff injection FAILED: %r"
                                  % (_e,), flush=True)
                    return _o(x, timestep, model_options=model_options,
                              seed=seed)

                guider.predict_noise = _pn
                print("[H3Memory] latent handoff armed: %d video rows, "
                      "%d audio cols + %d guard cols%s, release below "
                      "sigma %.2f"
                      % (0 if _ho_v is None else _ho_v.shape[2],
                         0 if _ho_a is None else _ho_a.shape[-1],
                         0 if _ho_guard is None else _ho_guard.shape[-1],
                         "" if _spine_seg is None else
                         " + SPINE %d cols" % _spine_seg.shape[-1],
                         float(handoff_release)), flush=True)
            shot_seed = (seed + si) if seed_per_shot else seed
            noise = ncs.RandomNoise().get_noise(shot_seed)[0]
            # payload signature: continuity mode + position + bank/spine
            # decide the conditioning payload, and with it the real pool
            _auto_set_payload(
                "%s%d_k%dr%d%s%s" % (
                    continuity[:4], 1 if si > 0 else 0,
                    len(keyframes), len(ref_blocks),
                    "s" if _spine is not None else "",
                    "2p" if two_pass_upscale else ""))
            _mb = _auto_measure_begin()
            try:
                if two_pass_upscale:
                    out1, _d1 = ncs.SamplerCustomAdvanced().sample(
                        noise, guider, sampler, _tp_sig_hi, latent)
                    up = _upscale_av_exact(_tp_tr, out1, _tp_lat_th,
                                           _tp_lat_tw)
                    _s = max(0.0, min(1.0, float(upscale_audio_denoise)))
                    _members, _was_nested = _tp_tr.extract_tensor(up["samples"])
                    if _was_nested and len(_members) >= 2:
                        if _s <= 0.0:
                            _ridx, _rstr = (0,), None
                        elif _s >= 1.0:
                            _ridx, _rstr = (0, 1), None
                        else:
                            _ridx, _rstr = (0, 1), {0: 1.0, 1: _s}
                    else:
                        _ridx = _rstr = None
                    noise2 = ncs.RandomNoise().get_noise(shot_seed + 977)[0]
                    up = _tp_tr.add_noise_nested_latent(
                        model, noise2, _tp_sig_lo, up,
                        renoise_indices=_ridx, noise_strengths=_rstr)
                    up = _tp_tr.finalize_latent_for_handoff(up)
                    out, _d = ncs.SamplerCustomAdvanced().sample(
                        ncs.DisableNoise().get_noise()[0], guider_hi,
                        sampler, _tp_sig_lo, up)
                else:
                    out, _d = ncs.SamplerCustomAdvanced().sample(
                        noise, guider, sampler, sigmas, latent)
            finally:
                _auto_measure_end(_mb, model, steps=steps)

            lat = out["samples"]
            if continuity == "context_pin":
                # the WHOLE AV latent, exactly as sampled - the next shot
                # pins its tail bit-identically, no decode in the path
                _cp_prev = {"samples": out["samples"]}
            _a_lat = None
            if getattr(lat, "is_nested", False):
                _comps = lat.unbind()
                _a_lat = _comps[1] if len(_comps) > 1 else None
                lat = _comps[0]
            # issue #12: keep each shot's latent exactly as sampled. Trimming
            # here would be throwing the pin material away on the user's
            # behalf, and the pin is the part that cannot be recovered later.
            _lat_v_parts.append(lat.detach().cpu())
            if _a_lat is not None:
                _lat_a_parts.append(_a_lat.detach().cpu())
            if continuity == "latent_handoff":
                _ho_taper_src = (lat[:, :, -1:].detach().clone()
                                 if handoff_taper > 0 else None)
                _ho_v = lat[:, :, -_HO_ROWS:].detach().clone()
                _ho_a = (_a_lat[..., -_HO_ACOLS:].detach().clone()
                         if _a_lat is not None and audio_lock else None)
            imgs = video_vae.decode(lat)
            if imgs.ndim == 5:
                imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2],
                                    imgs.shape[-1])
            if (continuity == "context_pin"
                    and chain_gain_control == "refresh_pin"):
                # REFRESH THE CARRIER IN THE PIXEL DOMAIN. Every earlier lever
                # either cleaned shipped frames after the loop (flatten,
                # master_normalize) or softened the pin through a latent
                # high-pass that under-reads decoded sharpness ~2.5x
                # (flatten_pin). Here the pin's video tail is decoded pixels
                # levelled to house/(running hop gain) and RE-ENCODED, so the
                # model's own ~1.15x gain lands the next shot ON the house
                # level instead of above it: L * (1/g) * g = L by
                # construction. Audio lane stays bit-identical raw - the
                # voice chain must never round-trip a VAE. Any failure falls
                # back to the raw pin.
                try:
                    import torch as _tt_rp
                    _zr = out["samples"]
                    _nested_rp = getattr(_zr, "is_nested", False)
                    _vlat = _zr.unbind()[0] if _nested_rp else _zr
                    _pf_rp = int(str(pin_frames)) if str(pin_frames) in (
                        "5", "22", "39", "56") else 22
                    # pixel tail long enough to encode >= _pf_rp latent frames
                    _tratio = max(1.0, imgs.shape[0] / float(_vlat.shape[2]))
                    _npix = min(imgs.shape[0],
                                int(_pf_rp * _tratio) + int(_tratio) + 4)
                    _ptail = imgs[-_npix:].detach().float()
                    # v5 = the proposal as written: measure AND level in the
                    # pack's own calibrated units (_cg_lap_var over the same
                    # 24-frame window the chain flatten uses), target
                    # house/g via _cg_flatten (blur-only, never sharpens).
                    # v4's own-units band-gain miscalibrated the hop gains
                    # and made every join visible (operator-called
                    # 2026-08-23: obvious cuts + speech misalignment).
                    _w_rp = min(24, imgs.shape[0])
                    _lap_now = _cg_lap_var(imgs[-_w_rp:])
                    if si == 0 and _lap_now > 1e-12:
                        _rp_ref = _lap_now
                        print("[H3Memory] refresh_pin: house ref %.5f "
                              "(shot 1 tail, _cg_lap_var units)" % _rp_ref,
                              flush=True)
                    if si > 0 and _rp_ref and _lap_now > 1e-12:
                        _g_meas = _lap_now / float(_rp_ref)
                        _g_raw = _g_meas * _rp_div_last
                        if _g_raw > 1.0:
                            _rp_gain = 0.5 * _rp_gain + 0.5 * _g_raw
                        print("[H3Memory] refresh_pin: hop gain measured "
                              "%.3f (raw %.3f, EMA %.3f)"
                              % (_g_meas, _g_raw, _rp_gain), flush=True)
                    _lvl, _sig_rp = _cg_flatten(
                        _ptail, float(_rp_ref) / _rp_gain if _rp_ref
                        else 0.0)
                    _rp_div_last = _rp_gain
                    _lvl = _lvl.clamp(0, 1)
                    _z_new = video_vae.encode(_lvl)
                    if _z_new.ndim == 4:
                        _z_new = _z_new.unsqueeze(0)
                    if _z_new.shape[2] >= _pf_rp:
                        _vnew = _vlat.clone()
                        _vnew[:, :, -_pf_rp:] = _z_new[
                            :, :, -_pf_rp:].to(_vnew.dtype).to(_vnew.device)
                        if _nested_rp:
                            import comfy.nested_tensor as _nt_rp
                            _cc_rp = list(_zr.unbind())
                            _cc_rp[0] = _vnew
                            _cp_prev = {"samples":
                                        _nt_rp.NestedTensor(_cc_rp)}
                        else:
                            _cp_prev = {"samples": _vnew}
                        print("[H3Memory] refresh_pin: tail levelled to "
                              "house/%.3f (flatten sigma %.2f) and "
                              "re-encoded; %d latent frames spliced, "
                              "audio raw" % (_rp_gain, _sig_rp, _pf_rp),
                              flush=True)
                    else:
                        print("[H3Memory] refresh_pin: encode returned %d "
                              "latent frames < pin %d - raw pin kept"
                              % (_z_new.shape[2], _pf_rp), flush=True)
                except Exception as _rp_e:
                    print("[H3Memory] refresh_pin FAILED (%s) - raw pin "
                          "kept for this join" % _rp_e, flush=True)
                # free the encode's working set NOW. The full-res float
                # tails plus the encoder pass's cached blocks otherwise sit
                # in the pool and push the next shot's DiT plan into
                # streaming (13 min/shot vs 5.5, measured 2026-08-22).
                # DiT stays loaded - only dead locals and allocator cache go.
                try:
                    del _lvl, _ptail
                except NameError:
                    pass
                try:
                    del _z_new, _vnew
                except NameError:
                    pass
                try:
                    import comfy.model_management as _mm_rp
                    _dev_rp = _mm_rp.get_torch_device()
                    _b4_rp = _mm_rp.get_free_memory(_dev_rp) / (1024 ** 3)
                    _mm_rp.soft_empty_cache()
                    _af_rp = _mm_rp.get_free_memory(_dev_rp) / (1024 ** 3)
                    print("[H3Memory] refresh_pin: encode working set freed "
                          "(%.1f -> %.1f GB free)" % (_b4_rp, _af_rp),
                          flush=True)
                except Exception:
                    pass
            aud = vae_decode_audio(audio_vae, out)
            sr = aud["sample_rate"]
            wav = aud["waveform"]

            # --- colour levelling to the FIXED house reference -----------
            # before anchor extraction and bank ingest, so the next shot's
            # conditioning inherits corrected statistics (closed loop)
            if si == 0:
                _house_frame = imgs[0:1].detach().clone()
            if color_level == "mvgd":   # per-shot, rolling house reference
                if si == 0:
                    _cc_mu, _cc_cov = _cc_stats(imgs[-min(24, imgs.shape[0]):])
                    print("[H3Memory] colour house stats set (shot 1 settled "
                          "tail)", flush=True)
                else:
                    imgs = _cc_apply(imgs, _cc_mu, _cc_cov)
                    print("[H3Memory] colour levelled to house", flush=True)


            if (si == 0 and chain_gain_control != "off"
                    and continuity in ("context_pin", "latent_handoff")):
                print("[H3Memory] NOTE: chain_gain_control corrects the "
                      "decoded frames and the bank, but continuity=%s carries "
                      "the previous shot's RAW LATENTS, which it cannot reach "
                      "- the texture ratchet rides the pin regardless "
                      "(measured +142%% over 3 shots with flatten ON). It is "
                      "effective on frame-carried modes (first_frame, cut)."
                      % continuity, flush=True)
            if chain_gain_control != "off":
                _w = min(_CG_WIN, imgs.shape[0])
                if si == 0:
                    _cg_ref = _cg_lap_var(imgs[-_w:])
                    print(f"[H3Memory] chain: house texture level "
                          f"{_cg_ref:.5f}", flush=True)
                if _cg_ref and chain_gain_control in ("flatten", "flatten_pin", "refresh_pin"):
                    # the RAW tail texture of this shot, before levelling: the
                    # pixel-domain ratchet flatten_pin has to undo on the pin
                    _cg_last_raw = _cg_lap_var(imgs[-_w:]) if si > 0 else _cg_ref
                    imgs, _s = _cg_flatten(imgs, _cg_ref)
                    if _s > 0:
                        print(f"[H3Memory] chain: levelled (sigma {_s:.2f})",
                              flush=True)
                elif _cg_ref and chain_gain_control == "match_output" and si > 0:
                    if _cg_lap_var(imgs[:_w]) > _cg_ref * 1.05:
                        _sig = _cg_sigma_for(imgs[:_w], _cg_ref)
                        if _sig > 0:
                            imgs = _cg_gauss(imgs, _sig)

            if continuity == "first_frame" and si > 0 and last_tail is not None:
                # did the model actually START on the handed-over frame?
                _m0 = float((imgs[0].detach().cpu().float()
                             - last_tail[-1].detach().cpu().float())
                            .abs().mean())
                print("[H3Memory] first_frame handover: frame0 vs prev last "
                      "mad %.4f -> %s" % (_m0, "HELD" if _m0 < 0.03 else
                                          "IGNORED (wrong checkpoint? "
                                          "fl2va is required)"), flush=True)

            if _dbg_pins:
                # bracket adherence: the regenerated head frames should
                # reproduce the pinned tail frames. Catches weak holds (a
                # prompt fighting the bracket) and index misalignment (each
                # pin also scored one frame early/late).
                _msgs = []
                for _idx, _src in _dbg_pins:
                    _sc = {}
                    for _d in (-1, 0, 1):
                        _k = _idx + _d
                        if 0 <= _k < imgs.shape[0]:
                            _sc[_d] = float((imgs[_k].detach().cpu().float()
                                             - _src.float()).abs().mean())
                    if _sc:
                        _best = min(_sc, key=_sc.get)
                        _msgs.append("idx %d mad %.4f (best %+d: %.4f)"
                                     % (_idx, _sc.get(0, float("nan")),
                                        _best, _sc[_best]))
                print("[H3Memory] bracket adherence: " + "; ".join(_msgs),
                      flush=True)
                _dbg_pins = []

            if (continuity == "latent_handoff" and si > 0
                    and last_tail is not None):
                # replay fidelity: the locked span should re-diffuse the
                # previous tail; a high mad means the lock is too weak (or
                # the row mapping is off).
                if handoff_depth == "block":
                    _pt, _fo, _ks = last_tail[-17:], 5, (0, 8, 16)
                else:
                    _pt, _fo, _ks = last_tail[-5:], 0, (0, 2, 4)
                _msgs = []
                for _k in _ks:
                    if _fo + _k < imgs.shape[0] and _k < _pt.shape[0]:
                        _msgs.append("f%d mad %.4f" % (_fo + _k, float(
                            (imgs[_fo + _k].detach().cpu().float()
                             - _pt[_k].detach().cpu().float()).abs().mean())))
                print("[H3Memory] handoff replay fidelity: "
                      + "; ".join(_msgs), flush=True)
                if _ho_wav_tail is not None:
                    # audio replay fidelity + speech-onset check: the head
                    # of this shot should be a replay of the previous tail;
                    # a speech onset inside it means the trim will chop the
                    # line's opening words (review-verified failure mode)
                    _win = max(1, int(sr * 0.02))
                    _n = min(_ho_wav_tail.shape[-1], wav.shape[-1])
                    _ep = _aud_env(_ho_wav_tail[..., :_n].cpu(), _win)
                    _en = _aud_env(wav[..., :_n].detach().cpu(), _win)
                    _m = min(_ep.shape[0], _en.shape[0])
                    _mad = float((_ep[:_m] - _en[:_m]).abs().mean())
                    def _onset(e):
                        idx = (e > 0.02).nonzero()
                        return (float(idx[0]) * 0.02) if idx.numel() else -1.0
                    print("[H3Memory] audio replay: env mad %.4f | speech "
                          "onset prev-tail %.2fs vs new-head %.2fs"
                          % (_mad, _onset(_ep[:_m]), _onset(_en[:_m])),
                          flush=True)
                    # the number that decides the join: first speech AFTER
                    # the replay+guard span. < 1.33s means the model planned
                    # speech under the lock and its opening was suppressed.
                    _e2 = _aud_env(wav[..., :int(sr * 2.5)].detach().cpu(),
                                   _win)
                    _post = _e2[int(0.925 / 0.02):]
                    _pi = (_post > 0.02).nonzero()
                    _t2 = (0.925 + float(_pi[0]) * 0.02) if _pi.numel() \
                        else -1.0
                    print("[H3Memory] new-line onset %.2fs (guard ends "
                          "1.33s, trim keeps from 0.88s) -> %s"
                          % (_t2, "CLEAN" if (_t2 < 0 or _t2 >= 1.30)
                             else "SUPPRESSED-START RISK"), flush=True)

            if si == 0 and self_anchor_voice and voice_block is None:
                # THE self-anchor: shot 1's own rendered voice becomes the
                # reference for every later shot. The bank carries voice as
                # part of a video_audio slot that keeps rolling; this pins
                # the ORIGINAL performance and never moves. The decoded audio
                # is already at the VAE's rate and stereo - just trim and
                # encode.
                _aw = wav[:1] if wav.ndim == 3 else wav.unsqueeze(0)[:1]
                _alim = 15 * sr
                if _aw.shape[-1] > _alim:
                    _aw = _aw[..., :_alim]
                _avz = audio_vae.encode(_aw.movedim(1, -1))
                voice_block = {"kind": "audio", "ref_audio_t": _avz.shape[-1],
                               "audio_latent": _avz}
                print("[H3Memory] self-anchor: shot 1's voice (%.1fs) is now "
                      "<Audio 1> for the remaining %d shot(s)."
                      % (_aw.shape[-1] / sr, n - 1), flush=True)
            # store this shot as a bank slot: centre clip + the audio under it
            clip_imgs, clip_start = _jb_centre_clip(imgs, bank_clip_frames)
            if bank_ref_noise > 0:
                # noised-clean-condition for the bank: the texture ratchet
                # rides reference clips exactly as it rode keyframes - the
                # model copies its own "pristine" output and adds ~1.2x
                # detail (worst on faces). A little seeded noise makes the
                # clip read as capture, so texture is regenerated, not
                # enhanced. The noised clip never reaches the final cut.
                _gn = torch.Generator().manual_seed(shot_seed ^ 0xBA9C)
                clip_imgs = (clip_imgs + bank_ref_noise * torch.randn(
                    clip_imgs.shape, generator=_gn).to(
                    clip_imgs.device, clip_imgs.dtype)).clamp(0, 1)
            bank.add((clip_imgs.clone(),
                      _jb_audio_window(wav, sr, clip_start,
                                       clip_imgs.shape[0])))

            # Upscale AFTER the bank has taken its clip: the bank must keep
            # base-resolution reference clips or the conditioning payload -
            # and the VRAM it costs - grows with output_scale for no gain.
            # Downstream of the VAE, so unlike the old two-pass path this
            # cannot leave the latent manifold. Frame COUNT is untouched, so
            # every seam-trim index below still means what it meant.
            imgs = _upscale_frames(imgs, output_scale, upscale_model,
                                   "H3Memory")
            if si == 0 and preview_first_shot:
                # shot 1 as early as possible, so a bad take can be cancelled
                # instead of waited out
                _write_shot_mp4(imgs, wav, sr,
                                "video/H3_FIRSTSHOT/firstshot",
                                "FIRST-SHOT PREVIEW saved", "H3Memory")
            if save_every_shot:
                # before the seam trim, so consecutive files overlap ~1s - a
                # chain that dies at the mux can still be joined by hand
                _write_shot_mp4(imgs, wav, sr, "video/H3_SHOTS/shot",
                                f"shot {si + 1}/{n} saved", "H3Memory")

            _tail_n = _OV_HO if continuity == "latent_handoff" else _OV
            last_tail = imgs[-max(_tail_n, 1):].clone()
            if continuity == "latent_handoff":
                _ho_wav_tail = wav[..., -int(round(sr * 0.925)):] \
                    .detach().cpu().clone()
                # onset guard: encode this shot's quietest 0.4s as room
                # tone. Locked into the next shot's audio JUST PAST the
                # trim point, it stops the model planning speech under the
                # replay - a lock alone merely masks the plan, and the free
                # region then resumes MID-WORD at the boundary
                # (waveform-verified: onset 25ms after the trim, word
                # truncated because its start was suppressed, not trimmed).
                try:
                    if not audio_lock:
                        _ho_guard = None
                        raise StopIteration
                    _gw = max(1, int(sr * 0.02))
                    _ge = _aud_env(wav.detach().cpu(), _gw)
                    _gn = int(sr * 0.4)
                    _k = max(1, _gn // _gw)
                    if wav.shape[-1] > _gn and _ge.shape[0] > _k + 1:
                        _cs = torch.cumsum(
                            torch.cat([torch.zeros(1), _ge]), 0)
                        _i = int((_cs[_k:] - _cs[:-_k]).argmin()) * _gw
                        _seg = wav[..., _i:_i + _gn]
                        _w3 = _seg if _seg.ndim == 3 else _seg.unsqueeze(0)
                        _ho_guard = audio_vae.encode(
                            _w3.movedim(1, -1)).detach().clone()
                    else:
                        _ho_guard = None
                except StopIteration:
                    pass
                except Exception as _e:
                    _ho_guard = None
                    print("[H3Memory] onset guard skipped: %r" % (_e,),
                          flush=True)

            # join handling. seamless: 1 duplicated frame. seamless_tail: the
            # next shot REGENERATES _OV frames that duplicate this tail - drop
            # them hard, or crossfade them (join_blend) so any residual step
            # is spread across the band instead of landing on one boundary.
            if si > 0 and continuity == "context_pin" and _cp_trim > 0:
                # the head is a regeneration of the previous shot's tail,
                # there purely to carry motion/colour/sound - drop it whole
                # (video frames + the exact matching audio span)
                imgs = imgs[min(_cp_trim, imgs.shape[0] - 1):]
                wav = wav[..., int(round(sr * _cp_trim / 24.0)):]
            elif si > 0 and continuity in ("seamless", "first_frame",
                                           "flf_chain"):
                # frame 0 IS the previous last frame - drop the duplicate
                # (audio via the quietest-gap cut so a head word survives)
                imgs = imgs[1:]
                wav = _smart_head_trim(wav, sr, int(round(sr / 24.0)))
            elif si > 0 and continuity in ("seamless_tail", "latent_handoff"):
                ov = min(_OV_HO if continuity == "latent_handoff" else _OV,
                         imgs.shape[0] - 1)
                _blend_prev = (frames_parts[-1] if frames_parts else
                               (_stream_writer.pending
                                if _stream_writer is not None else None))
                if join_blend and _blend_prev is not None:
                    prev = _blend_prev
                    band = min(ov, prev.shape[0])
                    w = torch.linspace(1.0, 0.0, band).view(-1, 1, 1, 1)
                    new_band = imgs[:band].cpu()
                    blend = w * prev[-band:] + (1.0 - w) * new_band
                    # grain guard: uncorrelated grain averages down in a
                    # blend; re-inject it so the band has no grain dip
                    hp = prev[-band:] - _cg_gauss(prev[-band:], 1.0)
                    g_sigma = float(hp.std())
                    gg = torch.Generator().manual_seed(shot_seed ^ 0xB1E0D)
                    blend = blend + g_sigma * torch.sqrt(
                        2.0 * w * (1.0 - w)) * torch.randn(
                        blend.shape, generator=gg, dtype=blend.dtype)
                    prev[-band:] = blend.clamp(0, 1)
                imgs = imgs[ov:]
                _xf = max(1, int(sr * 40 / 1000.0))
                if (continuity == "latent_handoff"
                        and (audio_lock or _spine is not None)):
                    # Symmetric trim: the overlap audio is a locked REPLAY of
                    # the previous tail - the real words already live in the
                    # previous part's kept audio. Keeping the replay instead
                    # put imperfect-replay audio under the previous video
                    # tail, heard as the next shot starting BEFORE the video
                    # cut and resyncing at the trim point. Drop it with the
                    # replayed frames (weld-compensated) so any residual step
                    # lands simultaneously with the video join.
                    keep_from = int(round(sr * ov / 24.0)) - _xf
                    if keep_from > 0:
                        wav = wav[..., keep_from:]
                else:
                    # silent-join (audio free) and seamless_tail: the new
                    # head's audio is genuine content - keep it in full and
                    # trim the PREVIOUS tail instead. With the join scripted
                    # into held silence, nothing generated is ever lost.
                    # seamless_tail: the new head's audio is fresh content
                    # (only video is bracket-pinned) and its opening words
                    # live there - keep it intact and trim the PREVIOUS tail
                    # instead, weld-compensated so A/V stay sample-locked.
                    cut = int(round(sr * ov / 24.0)) - _xf
                    if (audio_parts and cut > 0
                            and audio_parts[-1].shape[-1] > cut):
                        audio_parts[-1] = audio_parts[-1][..., :-cut]
                        # micro fade-out on the trimmed tail: the cut can
                        # land mid-phoneme and the 40ms weld lets a clipped
                        # fragment tick through ("ree" at the join,
                        # review-verified); 100ms to silence reads as a
                        # natural decay instead
                        _fn = min(int(sr * 0.10),
                                  audio_parts[-1].shape[-1])
                        if _fn > 8:
                            _fade = (torch.linspace(1.0, 0.0, _fn) ** 0.5) \
                                .to(audio_parts[-1].dtype)
                            audio_parts[-1][..., -_fn:] = \
                                audio_parts[-1][..., -_fn:] * _fade

            if join_fx == "vhs_glitch" and si > 0 and frames_parts:
                # dress the boundary: 2 tail frames of the previous part +
                # 3 head frames of this one, plus the audio hiccup on both
                # sides of the weld. Seeded per join for reproducibility.
                _fx_seed = (seed ^ 0x7A9E) + si
                _pt = frames_parts[-1]
                _ptn = min(2, _pt.shape[0])
                if _ptn:
                    _pt[-_ptn:] = _vhs_glitch_frames(_pt[-_ptn:], _fx_seed)
                _hn = min(3, imgs.shape[0])
                if _hn:
                    imgs[:_hn] = _vhs_glitch_frames(
                        imgs[:_hn], _fx_seed + 1).to(imgs.device, imgs.dtype)
                audio_parts[-1] = _vhs_glitch_audio(
                    audio_parts[-1], sr, at_start=False, seed=_fx_seed)
                wav = _vhs_glitch_audio(wav, sr, at_start=True,
                                        seed=_fx_seed + 1)
                print("[H3Memory] join %d dressed as VHS glitch" % si,
                      flush=True)

            # fp16: the encoder quantises to uint8 downstream, and this
            # timeline is what exhausted host RAM at 6 shots x 243f.
            if _stream_writer is not None:
                # streaming: the shot goes to lossless disk NOW and its RAM is
                # returned; only 1-D statistics stay behind.
                _stream_writer.add(imgs.cpu().float())
            else:
                frames_parts.append(imgs.cpu().half())
            audio_parts.append((wav if wav.ndim == 3 else wav.unsqueeze(0)).cpu())

        if color_level == "scene" and len(frames_parts) > 1:
            # SCENE-WIDE match: ONE reference for the whole piece, applied
            # once at the end. The per-shot mode matched each shot to a
            # rolling "house" and still left a hard step at every join
            # (measured 29% warmth, constant within each shot). Driving
            # every shot to a single scene-wide median removes the step,
            # because all shots are pulled to the same target rather than
            # to each other.
            # Reference: the BOUNDARY KEYFRAMES when we have them. They come
            # from one continuous take, agree to a few percent, and are the
            # colour each shot is supposed to arrive at - far better than a
            # median of the generated shots, which is itself drifted.
            if keyframe_images is not None:
                _s_mu, _s_cov = _cc_stats(keyframe_images)
                _src = "boundary keyframes"
            else:
                _s_mu, _s_cov = _cc_stats(torch.cat(
                    [p[::max(1, p.shape[0] // 24)] for p in frames_parts],
                    dim=0))
                _src = "scene median"
            _before = [float(_cc_stats(p)[0][0] / _cc_stats(p)[0][2]
                             .clamp_min(1e-6)) for p in frames_parts]
            for _i in range(len(frames_parts)):
                # PER-FRAME: a per-shot gain cannot fix a shot with a warm
                # head and a cool body - it scales the head too
                frames_parts[_i] = _cc_apply_perframe(frames_parts[_i],
                                                      _s_mu)
            _after = [float(_cc_stats(p)[0][0] / _cc_stats(p)[0][2]
                            .clamp_min(1e-6)) for p in frames_parts]
            print("[H3Memory] scene colour match (per-frame, target = %s) "
                  "| warmth before " % _src
                  + "/".join("%.2f" % v for v in _before) + "  after "
                  + "/".join("%.2f" % v for v in _after), flush=True)

        # issue #12: batch the raw per-shot latents along dim 0. Shapes match
        # when every shot shares one grid, which is the normal case; if a run
        # ever mixes grids the batch is impossible, and saying so beats
        # returning something that silently is not what it claims to be.
        def _batch_latents(parts, what):
            if not parts:
                return {"samples": torch.zeros(0)}
            shapes = {tuple(x.shape[1:]) for x in parts}
            if len(shapes) > 1:
                print(f"[H3Memory] {what}: shots do not share a grid "
                      f"({sorted(shapes)}) - returning shot 1 only.",
                      flush=True)
                return {"samples": parts[0]}
            return {"samples": torch.cat(parts, dim=0)}

        _lat_v = _batch_latents(_lat_v_parts, "video_latents")
        _lat_a = _batch_latents(_lat_a_parts, "audio_latents")
        print("[H3Memory] latents out: video %s, audio %s, head_frames=%d "
              "(UNTRIMMED - shots 2+ open with the replayed head)"
              % (tuple(_lat_v["samples"].shape),
                 tuple(_lat_a["samples"].shape) if _lat_a_parts else "none",
                 _cp_trim), flush=True)

        # always the short 40ms weld: a long crossfade CONSUMES its overlap,
        # which shortened audio 375ms per join and walked lip sync off from
        # shot 2 onward. The seamless_tail trim above pre-compensates 40ms.
        waveform = _xfade_audio(audio_parts, sr, ms=40)

        if _stream_writer is not None:
            # Streaming finish: gains from stored statistics (same math as
            # _mn_normalize), temps re-streamed one at a time into the master
            # encoder. Peak RAM this whole branch: one decode chunk.
            import os
            import folder_paths as _fp
            _mdir = os.path.join(_fp.get_output_directory(),
                                 "video", "H3CHAIN_STREAM")
            os.makedirs(_mdir, exist_ok=True)
            _i = 1
            while os.path.exists(os.path.join(_mdir, "master_%05d.mp4" % _i)):
                _i += 1
            _mpath = os.path.join(_mdir, "master_%05d.mp4" % _i)
            _stream_writer.finalize(_mpath, master_normalize, waveform, sr,
                                    prompt=_api_prompt,
                                    extra_pnginfo=_api_pnginfo)
            _ph = torch.zeros((1, _stream_writer.shots[0]["h"],
                               _stream_writer.shots[0]["w"], 3),
                              dtype=torch.half)
            print(f"[H3Memory] done (streamed): {n} shots -> {_mpath}. "
                  "master_frames is a 1-frame placeholder; wire master_path.",
                  flush=True)
            return (_ph, {"waveform": waveform, "sample_rate": sr}, n,
                    _lat_v, _lat_a, int(_cp_trim), _mpath)

        if master_normalize != "off":
            frames_parts, _mn_msg = _mn_normalize(frames_parts, master_normalize)
            if _mn_msg:
                print("[H3Memory] master normalize (%s): %s"
                      % (master_normalize, _mn_msg), flush=True)

        # Assemble in place. torch.cat allocated a second full timeline
        # while the first was still alive - 2x peak, and a 33.8 GB contiguous
        # request that a 64 GB box cannot satisfy. Same bytes, one buffer.
        _n = sum(int(_p.shape[0]) for _p in frames_parts)
        master = torch.empty((_n,) + tuple(frames_parts[0].shape[1:]),
                             dtype=frames_parts[0].dtype)
        _o = 0
        for _i in range(len(frames_parts)):
            _p = frames_parts[_i]
            master[_o:_o + _p.shape[0]].copy_(_p)
            _o += int(_p.shape[0])
            frames_parts[_i] = None
            del _p
        print(f"[H3Memory] done: {n} shots, {master.shape[0]} frames "
              f"(~{master.shape[0] / 24.0:.1f}s).", flush=True)
        return (master, {"waveform": waveform, "sample_rate": sr}, n,
                _lat_v, _lat_a, int(_cp_trim), "")

class H3OptionalImage:
    """An on/off gate for an OPTIONAL image input.

    A normal switch node cannot express "no image": both of its branches are
    required, so turning I2V "off" ends up feeding a placeholder (usually a
    black EmptyImage) into start_image - which is not text-to-video, it is
    video that starts from a black frame.

    This node passes the image through when enabled, and emits nothing (None)
    when disabled, which is exactly what an optional input expects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True, "label_on": "image ON",
                    "label_off": "no image (T2V)",
                    "tooltip": "Off = emits nothing, so the downstream optional "
                               "input behaves as if it were unconnected."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image to pass through when enabled."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "gate"
    CATEGORY = "utils/minimax"
    DESCRIPTION = ("Pass an image through, or nothing at all. Use to toggle an "
                   "optional input such as start_image (I2V) without feeding a "
                   "placeholder frame.")

    def gate(self, enabled, image=None):
        if not enabled:
            print("[H3OptionalImage] disabled - passing nothing (T2V).",
                  flush=True)
            return (None,)
        if image is None:
            print("[H3OptionalImage] enabled but no image connected - passing "
                  "nothing.", flush=True)
        return (image,)


class H3InfiniteTakeSampler:
    """ONE denoise trajectory over an arbitrary-length AV latent.

    Temporal MultiDiffusion for MiniMax H3: the sampler integrates a single
    full-length latent; the model only ever attends over one window at a
    time inside predict_noise, and overlapping windows' predictions blend
    with raised-cosine ramps. There are no shot boundaries anywhere in the
    trajectory - no seams to fix - and VRAM stays constant in duration
    because each model eval is one ordinary window.

    Latent geometry: 2 bootstrap rows encode the first 5 frames, then 5
    rows per 17-frame block. Mid-take windows are fed 2 rows of preceding
    context in the bootstrap slots (their prediction for those rows is
    discarded - weight 0), so every window looks like a normal clip to the
    model. Audio (40 latent-fps) is windowed to the same global timeline.

    The script carries one prompt per window; overlapping windows share
    their overlap content, so per-window dialogue becomes one continuous
    performance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "script": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "JSON {\"prompts\": [...]} - ONE prompt per "
                           "window (the node prints the window count and "
                           "time spans if the count is wrong)."}),
            "width": ("INT", {"default": 768, "min": 32, "max": 4096,
                              "step": 32}),
            "height": ("INT", {"default": 1344, "min": 32, "max": 4096,
                               "step": 32}),
            "total_frames": ("INT", {
                "default": 719, "min": 39, "max": 3600, "step": 17,
                "tooltip": "Total take length at 24 fps, snapped up to the "
                           "17k+5 grid. 719 = ~30s, 1450 = ~60s. VRAM does "
                           "NOT grow with this - only render time does."}),
            "window_frames": ("INT", {
                "default": 243, "min": 90, "max": 480, "step": 17,
                "tooltip": "Frames per attention window (snapped to 17k+5). "
                           "Stay inside the trained range (124-362); the "
                           "model never sees more than this at once."}),
            "overlap_frames": ("INT", {
                "default": 34, "min": 17, "max": 170, "step": 17,
                "tooltip": "Overlap between consecutive windows (multiples "
                           "of 17). More overlap = stronger agreement, more "
                           "compute."}),
            "seed": ("INT", {"default": 0, "min": 0,
                             "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 10, "min": 1, "max": 100}),
            "sampler_name": (_sampler_names(), {
                "default": "euler",
                "tooltip": "euler-family only for this node: H3's audio "
                           "velocity is chain-rule-scaled, so 'denoised' is "
                           "not x0 for the audio stream - multistep/"
                           "exponential samplers that extrapolate on it "
                           "mis-integrate the audio."}),
            "scheduler": (_scheduler_names(), {"default": "simple"}),
            "activation_reserve_gb": ("FLOAT", {
                "default": 8.0, "min": 2.0, "max": 22.0, "step": 0.5,
                "tooltip": "Activation VRAM reserved for ONE window eval. "
                           "This node pins its own reserve because the "
                           "loader's auto-reserve sees TWO latent shapes "
                           "here (full take at load, window per eval) and "
                           "double-reserves, forcing the DiT to offload "
                           "(render-verified on a 3090: 750s/it from a 93% "
                           "offloaded model). 7-8 GB suits 768x1344/243f; "
                           "raise only on OOM inside the model forward. "
                           "NOTE: a progress step is one pass over ALL "
                           "windows, so 0/N sits still for window_count x "
                           "per-window time before the first tick - that is "
                           "not a hang."}),
        },
        # OPTIONAL on purpose. A browser tab opened before this widget
        # existed has no widget for it, so it serialises the node without
        # one; as a REQUIRED input that fails validation until the page is
        # reloaded. Optional inputs still render as widgets, and still
        # append after every required widget, so slot order is unchanged.
        # NEW WIDGETS APPEND LAST - inserting above shifts every saved
        # workflow's values by one slot
        "optional": {
            "derive_length_from_script": ("BOOLEAN", {
                "default": False, "label_on": "script sets the length",
                "label_off": "total_frames sets the length",
                "tooltip": "ON: count the prompts and compute total_frames "
                           "from them (one prompt per window), ignoring the "
                           "total_frames widget. Lets an LLM rewriter decide "
                           "how long the piece is - write N shots, get N "
                           "windows. OFF: total_frames rules and the script "
                           "must contain exactly the matching number of "
                           "prompts (the node prints the window time-spans "
                           "if it does not)."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("frames", "audio", "windows")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"

    def run(self, model, clip, video_vae, audio_vae, script, width, height,
            total_frames, window_frames, overlap_frames, seed, steps,
            sampler_name="euler", scheduler="simple",
            derive_length_from_script=False, activation_reserve_gb=8.0):
        import torch
        import comfy.samplers as cs
        import comfy.sampler_helpers as csh
        import comfy.model_management as _mm
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3
        from comfy_extras.nodes_audio import vae_decode_audio

        _sc = json.loads(script)
        # accept the rewriter's key OR a bare list, so LLMEnhance output and
        # its passthrough (raw JSON) mode both drop straight in
        if isinstance(_sc, list):
            prompts = [str(p) for p in _sc]
        else:
            prompts = [str(p) for p in (_sc.get("prompts")
                                        or _sc.get("shots") or [])]
        if not prompts:
            raise ValueError(
                "script contained no prompts. Expected {\"prompts\": [...]} "
                "(also accepts {\"shots\": [...]} or a bare JSON list).")

        # window geometry must be known before the length can be derived
        wf = _jb_grid(window_frames)
        Wb = (wf - 5) // 17
        Ob = max(1, int(overlap_frames) // 17)
        Sb = Wb - Ob
        if Sb < 1:
            raise ValueError("overlap must be smaller than the window")

        if derive_length_from_script:
            # the SCRIPT decides the runtime: one prompt per window, so
            # N prompts need Sb*(N-1)+Wb blocks. Inverts the usual
            # dependency and lets an LLM rewriter pick the length.
            _n = len(prompts)
            total_frames = 17 * (Sb * (_n - 1) + Wb) + 5
            print("[H3Infinite] length derived from script: %d prompt(s) -> "
                  "%d frames (%.1fs)"
                  % (_n, total_frames, total_frames / 24.0), flush=True)

        latent, F = mmh3._empty_av_latent(width, height, total_frames)
        comps = latent["samples"].unbind()
        v0, a0_t = comps[0], comps[1]
        R, Ta = v0.shape[2], a0_t.shape[-1]
        B_total = (F - 5) // 17
        rows_w = 5 * Wb + 2
        A_w = int(round((17 * Wb + 5) / 24.0 * 40.0))
        c_skip = int(round(5 / 24.0 * 40.0))          # bootstrap audio span
        n_ov_r = 5 * Ob                               # overlap rows
        n_ov_a = int(round(17 * Ob / 24.0 * 40.0))    # overlap audio cols

        starts = [0]
        while starts[-1] + Wb < B_total:
            # snap starts DOWN to multiples of 3 blocks: audio cols per
            # block = 17/24*40 = 28.333, exact only at 3-block strides -
            # an unsnapped start carries ~8ms A/V phase error into a band
            # where its audio is averaged with a correctly phased
            # neighbour. COVERAGE BEATS PHASE: the final clamped window
            # must reach the end of the take even if its start cannot
            # snap; it just gets the warning. Pick total_frames so that
            # ((frames-5)/17 - Wb) % 3 == 0 to avoid it entirely.
            nxt = starts[-1] + Sb
            if nxt + Wb >= B_total:
                nxt = B_total - Wb
                if nxt % 3 != 0:
                    print("[H3Infinite] final window start %d is not a "
                          "multiple of 3 blocks: ~8ms A/V phase error in "
                          "its overlap. Use a total_frames where "
                          "((F-5)/17 - %d) %% 3 == 0 to avoid."
                          % (nxt, Wb), flush=True)
            else:
                nxt = (nxt // 3) * 3
            if nxt <= starts[-1]:
                break
            starts.append(nxt)
        if len(prompts) == 1 and len(starts) > 1:
            prompts = prompts * len(starts)
        if len(prompts) != len(starts):
            spans = ["window %d: %.1fs-%.1fs" % (
                k, (17 * b) / 24.0, (17 * b + 17 * Wb + 5) / 24.0)
                for k, b in enumerate(starts)]
            raise ValueError(
                "script has %d prompts but the take needs %d windows:\n%s"
                % (len(prompts), len(starts), "\n".join(spans)))

        wins = []
        for k, b0 in enumerate(starts):
            r0 = 0 if b0 == 0 else 5 * b0
            ac0 = 0 if b0 == 0 else min(
                int(round(17 * b0 / 24.0 * 40.0)), Ta - A_w)
            wins.append({
                "b0": b0, "r0": r0, "r1": r0 + rows_w,
                "a0": ac0, "a1": ac0 + A_w,
                "skip_r": 0 if b0 == 0 else 2,
                "skip_a": 0 if b0 == 0 else c_skip,
                "ramp_up": b0 != 0,
                "ramp_dn": k != len(starts) - 1,
            })

        print("[H3Infinite] %d frames (~%.1fs) as %d windows of %d frames, "
              "%d-frame overlap. VRAM = one window."
              % (F, F / 24.0, len(starts), 17 * Wb + 5, 17 * Ob),
              flush=True)

        conds_raw = []
        for p in prompts:
            tokens = clip.tokenize(p)
            conds_raw.append(clip.encode_from_tokens_scheduled(tokens))

        # evict the TE unless it lives on its own device (all text encoding
        # is done by this point; window conds are processed from tensors)
        _te_dev = getattr(clip.patcher, "load_device", None)
        _dit_dev = getattr(model, "load_device", None)
        if not (_te_dev is not None and _dit_dev is not None
                and str(_te_dev) != str(_dit_dev)):
            try:
                clip.patcher.model.to(_mm.text_encoder_offload_device())
                _dev = _mm.get_torch_device()
                _mm.free_memory(_mm.get_total_memory(_dev) * 0.9, _dev)
                _mm.soft_empty_cache()
                print("[H3Infinite] TE evicted; %.1f GB free for the DiT"
                      % (_mm.get_free_memory(_dev) / (1024 ** 3)),
                      flush=True)
            except Exception:
                pass

        vws = (1, v0.shape[1], rows_w, v0.shape[3], v0.shape[4])
        aws = (1, a0_t.shape[1], a0_t.shape[2], A_w)
        Nv_w = vws[1] * vws[2] * vws[3] * vws[4]
        Na_w = aws[1] * aws[2] * aws[3]
        Nv = v0.shape[1] * R * v0.shape[3] * v0.shape[4]
        Na = a0_t.shape[1] * a0_t.shape[2] * Ta

        def _ramp(n, up, dtype):
            t = torch.linspace(0, 1, max(2, n), dtype=dtype)
            w = 0.5 * (1.0 - torch.cos(t * 3.14159265))
            return w if up else (1.0 - w)

        guider = ncs.BasicGuider().get_guider(model, conds_raw[0])[0]
        _state = {"wconds": None}
        _node_seed = seed

        _orig_class_pn = type(guider).predict_noise

        def _pn(x, timestep, model_options={}, seed=None):
            if not (x.ndim == 3 and x.shape[1] == 1
                    and x.shape[2] == Nv + Na):
                # not the full AV pack (unexpected) - fall through untouched
                return _orig_class_pn(guider, x, timestep,
                                      model_options=model_options,
                                      seed=seed)
            dev, dt = x.device, x.dtype
            if _state["wconds"] is None:
                _state["wconds"] = []
                dummy = torch.zeros((1, 1, Nv_w + Na_w), device=dev,
                                    dtype=dt)
                for c in conds_raw:
                    d = {"positive": csh.convert_cond(c)}
                    for kk in d:
                        d[kk] = list(map(lambda a: a.copy(), d[kk]))
                    d = cs.process_conds(
                        guider.inner_model, dummy, d, dev, None, None,
                        _node_seed, latent_shapes=[vws, aws])
                    _state["wconds"].append(d)
                print("[H3Infinite] %d window conds processed"
                      % len(_state["wconds"]), flush=True)
            xv = x[:, 0, :Nv].reshape((1,) + tuple(v0.shape[1:]))
            xa = x[:, 0, Nv:].reshape((1,) + tuple(a0_t.shape[1:]))
            acc_v = torch.zeros_like(xv)
            acc_a = torch.zeros_like(xa)
            wsum_v = torch.zeros((1, 1, R, 1, 1), device=dev, dtype=dt)
            wsum_a = torch.zeros((1, 1, 1, Ta), device=dev, dtype=dt)
            for k, w in enumerate(wins):
                v_w = xv[:, :, w["r0"]:w["r1"]]
                a_w = xa[..., w["a0"]:w["a1"]]
                x_w = torch.cat([v_w.reshape(1, 1, -1),
                                 a_w.reshape(1, 1, -1)], dim=-1)
                d_w = cs.sampling_function(
                    guider.inner_model, x_w, timestep, None,
                    _state["wconds"][k]["positive"], 1.0,
                    model_options=model_options, seed=seed)
                dv = d_w[:, 0, :Nv_w].reshape((1,) + tuple(vws[1:]))
                da = d_w[:, 0, Nv_w:].reshape((1,) + tuple(aws[1:]))
                # row weights for this window's contribution
                nr = rows_w - w["skip_r"]
                wr = torch.ones(nr, device=dev, dtype=dt)
                if w["ramp_up"]:
                    wr[:n_ov_r] = _ramp(n_ov_r, True, dt).to(dev)
                if w["ramp_dn"]:
                    wr[-n_ov_r:] = torch.minimum(
                        wr[-n_ov_r:], _ramp(n_ov_r, False, dt).to(dev))
                g_r0 = w["r0"] + w["skip_r"]
                wrv = wr.view(1, 1, nr, 1, 1)
                acc_v[:, :, g_r0:w["r1"]] += dv[:, :, w["skip_r"]:] * wrv
                wsum_v[:, :, g_r0:w["r1"]] += wrv
                na = A_w - w["skip_a"]
                wa = torch.ones(na, device=dev, dtype=dt)
                if w["ramp_up"]:
                    wa[:n_ov_a] = _ramp(n_ov_a, True, dt).to(dev)
                if w["ramp_dn"]:
                    wa[-n_ov_a:] = torch.minimum(
                        wa[-n_ov_a:], _ramp(n_ov_a, False, dt).to(dev))
                g_a0 = w["a0"] + w["skip_a"]
                wav_ = wa.view(1, 1, 1, na)
                acc_a[..., g_a0:w["a1"]] += da[..., w["skip_a"]:] * wav_
                wsum_a[..., g_a0:w["a1"]] += wav_
            out_v = acc_v / wsum_v.clamp_min(1e-4)
            out_a = acc_a / wsum_a.clamp_min(1e-4)
            return torch.cat([out_v.reshape(1, 1, -1),
                              out_a.reshape(1, 1, -1)], dim=-1)

        guider.predict_noise = _pn

        sigmas = ncs.BasicScheduler().get_sigmas(model, scheduler, steps,
                                                 1.0)[0]
        sampler = ncs.KSamplerSelect().get_sampler(sampler_name)[0]
        noise = ncs.RandomNoise().get_noise(seed)[0]
        # pin a stable window-sized activation reserve for the whole run:
        # the loader's shape-keyed auto-reserve sees the full pack at load
        # AND the window pack per eval, double-reserving and mis-attributing
        # measurements. One fixed answer for every shape query, restored
        # after sampling.
        _res_bytes = int(float(activation_reserve_gb) * (1024 ** 3))
        _orig_memreq = getattr(model.model, "memory_required", None)

        def _win_reserve(input_shape, *a, **k):
            return _res_bytes

        model.model.memory_required = _win_reserve
        print("[H3Infinite] activation reserve pinned at %.1f GB per "
              "window eval" % float(activation_reserve_gb), flush=True)
        try:
            out, _d = ncs.SamplerCustomAdvanced().sample(
                noise, guider, sampler, sigmas, latent)
        finally:
            if _orig_memreq is not None:
                model.model.memory_required = _orig_memreq

        lat = out["samples"]
        a_lat = None
        if getattr(lat, "is_nested", False):
            _c = lat.unbind()
            a_lat = _c[1] if len(_c) > 1 else None
            lat = _c[0]

        # chunked video decode with the same fake-bootstrap context trick,
        # so decode VRAM stays constant in take length too
        frames_out = []
        db = max(2, Wb)
        b = 0
        while b < B_total:
            take = min(db, B_total - b)
            if b == 0:
                seg = lat[:, :, :2 + 5 * take]
            else:
                # 2 rows of preceding context in the bootstrap slots, first
                # 5 decoded frames dropped - same trick as the windows
                seg = lat[:, :, 5 * b:5 * (b + take) + 2]
            px = video_vae.decode(seg)
            if px.ndim == 5:
                px = px.reshape(-1, px.shape[-3], px.shape[-2],
                                px.shape[-1])
            frames_out.append(px if b == 0 else px[5:])
            b += take
        master = torch.cat([f.cpu() for f in frames_out], dim=0)

        aud = vae_decode_audio(audio_vae, {"samples": a_lat})
        print("[H3Infinite] done: %d frames (~%.1fs), %d windows."
              % (master.shape[0], master.shape[0] / 24.0, len(starts)),
              flush=True)
        return (master, aud, len(starts))


NODE_CLASS_MAPPINGS = {"H3ScriptSplit": H3ScriptSplit,
                       "H3ModelLoaderAny": H3ModelLoaderAny,
                       "H3ClipLoaderAny": H3ClipLoaderAny,
                       "H3AudioTrimStart": H3AudioTrimStart,
                       "H3MultishotSampler": H3MultishotSampler,
                       "H3MultishotMemorySampler": H3MultishotMemorySampler,
                       "H3InfiniteTakeSampler": H3InfiniteTakeSampler,
                       "H3OptionalImage": H3OptionalImage}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ScriptSplit": "H3 Shot List",
    "H3ModelLoaderAny": "H3 Model Loader (safetensors + GGUF)",
    "H3ClipLoaderAny": "H3 CLIP Loader (safetensors + GGUF)",
    "H3AudioTrimStart": "H3 Audio Trim Start",
    "H3MultishotSampler": "H3 Multishot Sampler (one node)",
    "H3MultishotMemorySampler": "H3 Multishot Sampler + Memory (long form)",
    "H3InfiniteTakeSampler": "H3 Infinite Take (one trajectory, any length)",
    "H3OptionalImage": "H3 Optional Image (I2V on/off)"}
