"""H3 Q-Graft - bias H3's attention envelope toward a donor model's, at a dose.

Applies per-block multiplicative gains to each block's attn.q_norm.weight -
scaling q scales that block's attention logits, i.e. its sharpness/temperature.
The gains come from comparing per-block Q magnitude profiles of a donor
(Z-Image for the realism experiment) against H3's own, depth-resampled.

Runtime patch, no weight surgery: works on every quant (norm weights are
stored full-precision even in GGUF), dose is a slider, K/V/MLP untouched -
the same protections TenStrip's graft used, expressed as a patch instead of
a merged checkpoint. dose 0 = exactly the base model.
"""
import json
import os
import re

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_KEY = re.compile(r"blocks\.(\d+)\.attn\.q_norm\.weight$")


class H3QGraftPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "gains_file": ("STRING", {"default": "", "tooltip":
                "Path to a gains json ({\"gains\": {\"<block>\": g, ...}}) built by "
                "comparing donor-vs-H3 Q magnitude profiles. Empty = passthrough."}),
            "dose": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5, "step": 0.05,
                "tooltip": "0 = base model untouched. 1 = the full measured gain "
                "curve. The effective per-block multiplier is 1 + dose*(gain-1)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "H3/experimental"
    DESCRIPTION = "Per-block attention-sharpness graft (donor Q envelope onto H3) via q_norm scaling."

    def patch(self, model, gains_file, dose):
        if dose <= 0 or not (gains_file or "").strip():
            return (model,)
        p = gains_file.strip().strip('"')
        if not os.path.isfile(p):
            raise FileNotFoundError("H3QGraftPatch: gains file not found: %s" % p)
        gains = {int(k): float(v) for k, v in json.load(open(p))["gains"].items()}
        m = model.clone()
        sd = m.model.state_dict()
        patches, lo, hi = {}, 10.0, 0.0
        for k, w in sd.items():
            mt = _KEY.search(k)
            if not mt:
                continue
            g = gains.get(int(mt.group(1)))
            if g is None:
                continue
            eff = 1.0 + dose * (g - 1.0)
            if abs(eff - 1.0) < 1e-4:
                continue
            lo, hi = min(lo, eff), max(hi, eff)
            # plain tensor value = additive diff on the weight
            patches[k] = (w.to("cpu", copy=True) * (eff - 1.0),)
        if not patches:
            print("[H3QGraft] no matching q_norm keys - model unchanged", flush=True)
            return (model,)
        m.add_patches(patches, 1.0)
        print("[H3QGraft] dose %.2f: %d blocks patched, effective q gain %.3f..%.3f"
              % (dose, len(patches), lo, hi), flush=True)
        return (m,)


NODE_CLASS_MAPPINGS["H3QGraftPatch"] = H3QGraftPatch
NODE_DISPLAY_NAME_MAPPINGS["H3QGraftPatch"] = "H3 Q-Graft (attention envelope, experimental)"
