"""Report whether interior keyframe anchors are available, and from where.

As of 2.7.1 this module DETECTS the capability; it no longer installs it.
Earlier versions rewrote `PackedLayout.__init__` in memory and recompiled it,
which the Comfy Registry's security scan now flags - and a flagged version is
not served to Manager at all, so the whole release becomes uninstallable for
the sake of a code path that is dead on any current ComfyUI.

Three outcomes, in the order they are tried:

  1. ComfyUI 0.34+  - the engine honours a non-zero `resolved_frame_index`
     itself (`cond_t = cursor + FRAME_RESCALE * index`). Nothing to do.
  2. ComfyUI-H3-Motion-Context installed - that pack owns the layout site on
     older cores and its version is a superset (per-row coordinates plus the
     audio timeline move). It is already required for continuity=context_pin.
  3. Neither - interior anchors are unavailable and callers are told to
     update ComfyUI or install that pack.

Anchors at the FIRST and LAST frame are stock behaviour and work on every
core in all three cases; only the in-between ones depend on this.

Historical note, kept because it explains why interior anchors are sound at
all: stock's two supported cases are

    p = 0        ->  cond_t = text_len
    p = fc - 1   ->  cond_t = text_len + sum(_video_t_spans(latent_t)) - FRAME_RESCALE

and since `sum(_video_t_spans(latent_t)) == FRAME_RESCALE * frame_count` on
every valid 17k+5 frame count, both collapse into `text_len + FRAME_RESCALE * p`,
which is defined for every p. Verified on an RTX 5090 (243 frames, anchor at
pixel frame 121): the rendered frame most resembling the anchor was frame 122
of 243 - the requested position, off by one - reached by continuous motion with
no cut (peak frame-to-frame delta 2.3x median), audio unbroken across it.

One caveat worth knowing: if two images have no continuous path between them
(different rooms, say), H3 satisfies the anchor with a hard CUT rather than a
move. Stock first/last does the same with such a pair, so that is the model
being sensible, not a defect of interior anchors.

"""

import inspect
import logging

_MARK = "_h3_interior_keyframes_patched"
_state = {"done": False, "ok": False, "msg": ""}


def _motion_context_present():
    """ComfyUI-H3-Motion-Context patches the SAME layout site and refuses to
    stack on a foreign patch (its self-test compares against stock and sees
    ours as a position mismatch). Its version is a superset - per-row
    coordinates plus audio timeline placement - so when it is installed we
    stand down and let it own the site. When it is absent we fill the gap,
    which is the only reason this module still exists."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    # This file ships two ways: inside a pack folder (custom_nodes/<pack>/
    # this.py - siblings live one level up) and loose (custom_nodes/this.py -
    # siblings live right here). Guessing one level was the bug that made a
    # loose install never detect Motion-Context, never stand down, and mask
    # the collision on the dev machine while every packaged install hit it.
    # Walk up to the custom_nodes directory instead of assuming a depth.
    cn = here
    while cn and os.path.basename(cn).lower() != "custom_nodes":
        parent = os.path.dirname(cn)
        if parent == cn:
            cn = here                          # fell off the root - fall back
            break
        cn = parent
    try:
        for name in os.listdir(cn):
            if "motion-context" in name.lower().replace("_", "-"):
                return name
    except Exception:
        pass
    return None


def ensure_interior_keyframes(verbose=True):
    """Idempotent. Returns (ok, message)."""
    if _state["done"]:
        return _state["ok"], _state["msg"]
    _state["done"] = True

    # ComfyUI 0.34.0 honours a non-zero resolved_frame_index by itself, so
    # there is nothing left for this module to add. Probing the engine beats
    # assuming either way: on an older core the probe fails and the patch
    # installs as before.
    try:
        import torch
        import comfy.ldm.minimax.model as _mm
        _native = True
        for _idx in (0, 17):
            _lay = _mm.PackedLayout(7, 7, 22, 38, 16,
                                    keyframes=[{"resolved_frame_index": _idx,
                                                "latent": torch.zeros(1, 4, 1, 22, 38)}])
            _c = [(a, b) for a, b, k in _lay.segments if k == "cond"]
            if not _c or abs(float(_lay.position_ids[_c[0][0], 0])
                             - (7.0 + _mm.FRAME_RESCALE * _idx)) > 1e-6:
                _native = False
                break
    except Exception:
        _native = False
    if _native:
        _state["ok"] = True
        _state["msg"] = ("standing down: this ComfyUI places interior keyframe "
                         "anchors natively, so no layout patch is required.")
        if verbose:
            logging.info("[H3Keyframes] " + _state["msg"])
        return True, _state["msg"]

    _mc = _motion_context_present()
    if _mc:
        _state["ok"] = True
        _state["msg"] = (
            f"standing down: {_mc} is installed and owns the keyframe layout "
            f"patch (its version is a superset). Interior anchors come from "
            f"that pack; first/last anchors work either way.")
        if verbose:
            logging.info("[H3Keyframes] " + _state["msg"])
        return True, _state["msg"]

    try:
        import comfy.ldm.minimax.model as mm
    except Exception as e:                                    # pragma: no cover
        _state["msg"] = f"MiniMax H3 not present in this ComfyUI ({e})"
        return False, _state["msg"]

    cls = getattr(mm, "PackedLayout", None)
    if cls is None:
        _state["msg"] = "comfy.ldm.minimax.model.PackedLayout not found"
        return False, _state["msg"]

    if getattr(cls, _MARK, False):
        _state["ok"] = True
        _state["msg"] = "already patched"
        return True, _state["msg"]

    try:
        # match against the RAW source: the anchor is indented as a method body,
        # and dedenting first would strip the class indent off it and never match
        raw = inspect.getsource(cls.__init__)
    except Exception:                                         # pragma: no cover
        raw = ""

    if "FRAME_RESCALE * float(pixel_index)" in raw:
        # someone already generalised it (e.g. a hand-edited core file)
        cls._h3_interior_keyframes_patched = True
        _state["ok"] = True
        _state["msg"] = "core already supports interior anchors; nothing to do"
        return True, _state["msg"]

    # Older core, and no pack installed that owns the layout. This module used
    # to rewrite PackedLayout.__init__'s source in memory and recompile it.
    # That is gone as of 2.7.1: the Comfy Registry's security scan flags
    # dynamic code execution in a published node, and a flagged version is not
    # served to Manager at all. The capability is not lost - it moved to the
    # two places that can provide it without recompiling anything:
    #   * ComfyUI 0.34+, which places interior anchors natively (the branch
    #     at the top of this function), or
    #   * ComfyUI-H3-Motion-Context, which owns the layout patch on older
    #     cores and is already required for continuity=context_pin.
    # Anchors at frame 0 and the last frame are stock behaviour and keep
    # working on every core, patch or no patch.
    _state["msg"] = (
        "interior keyframe anchors need either ComfyUI 0.34+ (which places "
        "them natively) or the ComfyUI-H3-Motion-Context pack (which places "
        "them on older cores). This ComfyUI is older than 0.34 and that pack "
        "is not installed, so interior anchors are unavailable - update "
        "ComfyUI or install that pack. Anchors at the first and last frame "
        "work normally either way.")
    if verbose:
        logging.info("[H3Keyframes] " + _state["msg"])
    return False, _state["msg"]
