"""Extend take: one prompt, one continuous speech, N pinned windows.

The chain samplers already do the hard part - context_pin carries the
previous window's tail latents plus an audio reference into the next window,
which is the same shape of mechanism the LTX extend sampler used to continue
a take. What this node adds is the ergonomics the JoyEcho ancestor had and
the chain did not: the operator states a TAKE LENGTH, the node picks a
window that fits THIS card (or takes one), and derives the window count with
the replay trim accounted, so the writer can be told "one continuous speech,
N windows of W seconds" instead of "N shots with per-shot lines". The LLM
never does arithmetic; it gets the numbers.

Verified 2026-08-16/17: six writer-driven renders on a 5090 at 141/192/243f
windows, 17/17 joins continued the speech.

Nothing here samples. Arithmetic plus a summary string for the canvas.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# H3 frame grid: valid clip lengths satisfy (F - 5) % 17 == 0
_GRID = (5, 17)
# candidate windows, largest first (fewest joins wins when it fits)
_WINDOWS = (243, 226, 209, 192, 175, 158, 141, 124, 107, 90)
# latent cells the reserve system keys on: 24 ch x T_lat x H/16 x W/16, with
# T_lat ~ 0.2961*F + 0.35 (fit to logged cells: 192f->5054720, 243f->6384960
# at 736x1280; 243f->6993216 at 768x1344)
_GGUF_RATE_GB_PER_MCELL = 2.4      # measured 2.2-2.5 on 3090/5090 caches
_NATIVE_RATE_GB_PER_MCELL = 4.3    # w4a8/int8 measured 3.0-5.7 (24 GB lab cache)
_KEEPOUT_GB = 1.0
_WDDM_FRAC = 0.09                  # the driver-headroom margin the reserve uses
_DEFAULT_WEIGHTS_GB = 15.0         # curve-Q5_1 when no MODEL is wired
_RESIDENT_TARGET = 0.6             # keep at least this fraction of weights on-card


def _snap_grid(frames):
    a, b = _GRID
    f = max(a + b, int(frames))
    return a + b * max(1, int(round((f - a) / float(b))))


def _cells(width, height, frames):
    t_lat = 0.2961 * frames + 0.35
    return 24.0 * t_lat * (height / 16.0) * (width / 16.0)


def _card_total_gb():
    try:
        import comfy.model_management as mm
        return mm.get_total_memory(mm.get_torch_device()) / 2**30
    except Exception:
        return 24.0


def _weights_gb(model):
    if model is None:
        return _DEFAULT_WEIGHTS_GB, "assumed (wire the MODEL for a real figure)"
    try:
        return model.model_size() / 2**30, "from the wired model"
    except Exception:
        return _DEFAULT_WEIGHTS_GB, "assumed (model_size unavailable)"


def _rate_gb_per_mcell(model):
    """Pool per million latent cells: the card's own measurements when it has
    any, else the family default. Family from the model's stem when wired."""
    fam = "gguf"
    try:
        name = getattr(getattr(model, "model", None), "__class__", None)
        # ModelPatcher does not carry the file name; the reserve system does,
        # via its cache keys - use those to pick a measured rate.
        from .h3_multishot_utils import _auto_cache_load, _model_family
        import torch
        dev = torch.cuda.get_device_name(0)
        rates = []
        for k, v in _auto_cache_load().items():
            try:
                d2, m2, c2, s2 = str(k).split("|", 3)
                c2 = int(c2)
            except ValueError:
                continue
            if d2 != dev or not v or not c2:
                continue
            if _model_family(m2) == "gguf":
                rates.append(v / 2**30 / (c2 / 1e6))
        if rates:
            rates.sort()
            return rates[len(rates) // 2], "measured on this card"
    except Exception:
        pass
    return _GGUF_RATE_GB_PER_MCELL, "default (no measurements yet)"


def plan_take(take_seconds, window, width, height, fps, pin_frames, model=None):
    """Shared by H3ExtendTake and H3StudioControls(take_seconds>0)."""
    want = int(round(float(take_seconds) * int(fps)))
    why = ""
    if window == "auto":
        total = _card_total_gb()
        w_gb, w_src = _weights_gb(model)
        rate, r_src = _rate_gb_per_mcell(model)
        # "Fits" = the activation pool fits with at least ~60% of the
        # weights resident; the rest streams (the reserve system's own
        # rule - streamed weights are cheap, a paged pool is fatal).
        # Demanding fully-resident weights would hand a 24 GB card 90-
        # frame windows when 141-192f is measured fine there.
        margins = _KEEPOUT_GB + total * _WDDM_FRAC
        budget = total - margins - _RESIDENT_TARGET * w_gb
        chosen = None
        for cand in _WINDOWS:
            pool = rate * _cells(width, height, cand) / 1e6
            if pool <= budget:
                chosen = cand
                break
        if chosen is None:
            chosen = _WINDOWS[-1]
            why = (" | WARNING: even %d frames estimates a %.1f GB pool "
                   "against %.1f GB of room - expect heavy streaming; "
                   "lower the resolution or load a smaller model"
                   % (chosen, rate * _cells(width, height, chosen) / 1e6,
                      budget))
        f = chosen
        est = rate * _cells(width, height, f) / 1e6
        stream = max(0.0, est + w_gb - (total - margins))
        why = (" | auto: %.0f GB card, weights %.1f GB (%s), pool rate %.2f "
               "GB/Mcell (%s) -> %d-frame window estimates a %.1f GB pool%s"
               % (total, w_gb, w_src, rate, r_src, f, est,
                  (", ~%.1f GB of weights will stream" % stream)
                  if stream > 0.3 else ", weights fully resident")) + why
    else:
        f = _snap_grid(int(window))
    step = max(1, f - int(pin_frames))          # delivered frames per later window
    n = 1
    while f + (n - 1) * step < want:
        n += 1
    total_frames = f + (n - 1) * step
    summary = ("EXTEND TAKE: %.1f s requested -> %d windows x %d frames "
               "(pin %d) = %d delivered frames = %.1f s%s"
               % (float(take_seconds), n, f, int(pin_frames), total_frames,
                  total_frames / float(fps), why))
    print("[H3ExtendTake] " + summary, flush=True)
    return (n, f, total_frames, summary)


class H3ExtendTake:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "take_seconds": ("FLOAT", {
                    "default": 30.0, "min": 3.0, "max": 600.0, "step": 0.5,
                    "tooltip": "How long the finished take should be. Windows "
                               "are derived; the last one may run a little "
                               "long rather than short."}),
                "window": (["auto"] + [str(w) for w in _WINDOWS], {
                    "default": "auto",
                    "tooltip": "Frames per window. auto = the largest window "
                               "whose estimated activation pool fits beside "
                               "the model's weights on THIS card (fewest "
                               "joins that will not thrash). Wire the MODEL "
                               "for a real weight size; otherwise 15 GB is "
                               "assumed. Pick a number to override."}),
                "width": ("INT", {"default": 736, "min": 64, "max": 2048, "step": 16,
                                  "tooltip": "Render width - wire from MASTER CONTROLS."}),
                "height": ("INT", {"default": 1280, "min": 64, "max": 2048, "step": 16,
                                   "tooltip": "Render height - wire from MASTER CONTROLS."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "pin_frames": ("INT", {
                    "default": 22, "min": 0, "max": 56,
                    "tooltip": "The sampler's picture pin (head trim per join). "
                               "22 unless you changed it on the sampler."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional: the loaded H3 model, "
                                    "so auto can size the weights."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("num_shots", "frames_per_shot", "total_frames", "summary")
    FUNCTION = "plan"
    CATEGORY = "H3/episode"
    DESCRIPTION = ("Turns a take length into window length + window count for "
                   "the chain samplers + writer (extend take join style).")

    def plan(self, take_seconds, window, width, height, fps, pin_frames,
             model=None):
        return plan_take(take_seconds, window, width, height, fps,
                         pin_frames, model)


NODE_CLASS_MAPPINGS["H3ExtendTake"] = H3ExtendTake
NODE_DISPLAY_NAME_MAPPINGS["H3ExtendTake"] = "H3 Extend Take (seconds -> windows)"
