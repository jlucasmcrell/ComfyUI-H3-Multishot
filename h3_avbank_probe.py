# -*- coding: utf-8 -*-
"""Make ref2va reference rows coexist with keyframe anchors in one generation.

Comfy's PackedLayout ALREADY composes both - keyframe cond rows pack right
after text, ref rows after them, one shared row cursor
(comfy/ldm/minimax/model.py:297). The refs+keyframes crash is a plumbing bug
in MiniMaxH3.extra_conds (comfy/model_base.py:2098): when both keys are
present, the refs branch OVERWRITES payload["cond_video_latents"] instead of
appending, so the packed sequence has more cond rows than latents. This module
monkey-patches extra_conds to MERGE (keyframes first, matching the layout's
segment order), and adds H3KeyframeInject.

OWNERSHIP - this site is shared with ComfyUI-H3-Motion-Context. Their
patch_payload.py wraps the same method for the same reason and REFUSES to
stack on an unrecognised wrapper: if we get here first, their pack errors out
and continuity=context_pin dies (user-reported, 2026-08-11). Their detection
honours a shared marker ABI - a wrapper carrying
`_h3_motion_context_payload_patch = True` is recognised as a compatible copy
and they stand down. So this wrapper:

  1. does everything their wrapper does (video merge + audio latents +
     frame_count passthrough, for ANY keyframes+refs graph - theirs only
     fires on payloads carrying their marker keys), and
  2. declares the marker, so whichever pack loads first owns the site and
     the other stands down cleanly. Load order stops mattering.
"""

import comfy.model_base as _mb
import node_helpers

# ComfyUI-H3-Motion-Context's shared ABI (patch_payload.py). Declaring the
# marker promises our wrapper subsumes theirs - keep the body a superset.
_MC_MARKER = "_h3_motion_context_payload_patch"


def _apply_merge_patch():
    cur = _mb.MiniMaxH3.extra_conds
    if getattr(cur, "_h3_avbank_merge", False):
        return
    if getattr(cur, _MC_MARKER, False):
        # Motion-Context (or a pack vendoring its patch) got here first and
        # its wrapper handles the same collision - stand down.
        print("[H3AVBank] extra_conds already owned by a Motion-Context-"
              "compatible patch - standing down.", flush=True)
        return
    _orig = cur

    def extra_conds(self, **kwargs):
        out = _orig(self, **kwargs)
        kf = kwargs.get("minimax_keyframes")
        refs = kwargs.get("minimax_refs")
        if not kf or not refs:
            return out           # one mechanism in play - stock is correct
        cond = out.get("minimax_payload")
        payload = getattr(cond, "cond", None) if cond is not None else None
        if not isinstance(payload, dict):
            print("[H3AVBank] could not reach the H3 payload - keyframe "
                  "latents may be overwritten by refs", flush=True)
            return out
        # layout packs keyframe cond rows FIRST, then ref rows - the latent
        # list must match that order
        payload["cond_video_latents"] = (
            [k["latent"] for k in kf if "latent" in k]
            + [r["latent"] for r in refs if "latent" in r])
        # audio reference latents ride the same payload (Motion-Context's
        # timeline audio ref depends on this; harmless when absent)
        # keyframe audio (e.g. a motion-context audio tail) first, like the
        # video list above and the layout's segment order; refs after
        aud = ([k["audio_latent"] for k in kf
                if k.get("audio_latent") is not None]
               + [r["audio_latent"] for r in refs
                  if r.get("audio_latent") is not None])
        if aud:
            payload["cond_audio_latents"] = aud
        # frame_count only when the graph supplied one - overwriting a value
        # the original already set with None breaks the last-frame anchor
        fc = kwargs.get("minimax_frame_count")
        if fc is not None:
            payload["frame_count"] = fc
        return out

    extra_conds._h3_avbank_merge = True
    setattr(extra_conds, _MC_MARKER, True)
    _mb.MiniMaxH3.extra_conds = extra_conds
    print("[H3AVBank] extra_conds merge patch active (refs + keyframes "
          "compose; Motion-Context-compatible)", flush=True)


_apply_merge_patch()


class H3KeyframeInject:
    """Add a frame-0 start keyframe to existing conditioning (ref2va etc.).

    The keyframe is what the video PHYSICALLY continues from; the refs on
    the incoming conditioning stay what the encoder looks at for identity.
    Requires the extra_conds merge patch in this module (applied at import).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "vae": ("VAE",),
            "start_image": ("IMAGE",),
            "width": ("INT", {"default": 544, "min": 32, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 960, "min": 32, "max": 4096, "step": 32}),
            "length": ("INT", {"default": 362, "min": 5, "max": 3600, "step": 17,
                "tooltip": "Must match the generation's frame count."}),
        }, "optional": {
            "end_image": ("IMAGE", {"tooltip": "Optional last-frame anchor."}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "inject"
    CATEGORY = "model/conditioning/minimax"

    def inject(self, conditioning, vae, start_image, width, height, length,
               end_image=None):
        from comfy_extras.nodes_minimax_h3 import _resize, align_frame_count
        frame_count = align_frame_count(max(5, length))
        keyframes = [{"resolved_frame_index": 0,
                      "latent": vae.encode(_resize(start_image[:1], width, height, "disabled"))}]
        if end_image is not None:
            keyframes.append({"resolved_frame_index": frame_count - 1,
                              "latent": vae.encode(_resize(end_image[:1], width, height, "center"))})
        out = node_helpers.conditioning_set_values(conditioning, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })
        print(f"[H3AVBank] injected {len(keyframes)} keyframe(s) at "
              f"{[k['resolved_frame_index'] for k in keyframes]} "
              f"(frame_count {frame_count})", flush=True)
        return (out,)


NODE_CLASS_MAPPINGS = {"H3KeyframeInject": H3KeyframeInject}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3KeyframeInject": "H3 Keyframe Inject (experimental: refs + start frame)"}
