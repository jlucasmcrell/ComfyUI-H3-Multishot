"""H3 Speed Boosters: one switch panel for the third-party accelerators.

Why this node exists instead of shipping the accelerator nodes in the canvas:
ComfyUI refuses to queue a workflow containing ANY node class that is not
installed - even bypassed (the v2 sol-attn lesson, reported by users). This
class ships in the pack, so the graph always queues; each booster is applied
at runtime only if its pack is installed, and a missing pack costs a printed
install pointer, never the render.

Measured on the same seed, ship shape (640x1152, 192f, 14 steps, 2026-08-16):
  baseline 621s | Spectrum 441s (-29%) | TeaCache@0.15 531s (-14%)
  | TeaCache@0.30 491s (-21%) | blockcache 551s (-11%)
  | Spectrum+TeaCache stacked 450s - NOT faster than Spectrum alone.
CORRECTION 2026-08-17: the block cache logs "cached 0/14 model forwards" on
EVERY run at 14 steps, on both a 5090 and a 3090 (11 of 11 runs checked) -
it is inert at the shipped step count; the -11% above did not reproduce and
its "indistinguishable" output was indistinguishable because it did nothing.
Ships OFF now; useful only at 30+ steps where consecutive residuals get
under its 0.12 threshold. Spectrum and TeaCache:
visible distortion on PEOPLE (rooms fine). Stacked Spectrum+TeaCache:
severely damaged. Ambient audio fine on all arms.
"""

_PACKS = {
    "easycache": ("EasyCache", None),   # core ComfyUI - no install needed
    "spectrum": ("SpectrumApplyMiniMaxH3",
                 "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3"),
    "teacache": ("MiniMaxH3TeaCache",
                 "https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache"),
    "blockcache": ("MiniMaxH3BlockCacheT8",
                   "https://github.com/T8mars/comfyui-minimax-h3-blockcache-T8"),
}


def _apply(model, class_name, url, label, **kw):
    import nodes as core_nodes
    cls = core_nodes.NODE_CLASS_MAPPINGS.get(class_name)
    if cls is None:
        print("[H3SpeedBoosters] %s is ON but its pack is not installed - "
              "rendering at normal speed. Install: %s" % (label, url),
              flush=True)
        return model
    inst = cls()
    fn = getattr(inst, getattr(cls, "FUNCTION"))
    # fill required inputs we don't surface with their declared defaults
    spec = cls.INPUT_TYPES().get("required", {})
    args = {}
    for name, s in spec.items():
        if name == "model":
            args[name] = model
        elif name in kw:
            args[name] = kw[name]
        else:
            d = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
            if "default" in d:
                args[name] = d["default"]
            elif isinstance(s[0], list) and s[0]:
                args[name] = s[0][0]
    out = fn(**args)
    print("[H3SpeedBoosters] %s applied." % label, flush=True)
    if isinstance(out, tuple):
        return out[0]
    # V3-schema nodes (ComfyUI 0.33.1+) return a NodeOutput carrying the
    # values in .args - passing it through as the MODEL crashes downstream
    a = getattr(out, "args", None)
    if isinstance(a, tuple):
        return a[0] if a else model
    return out


class H3SpeedBoosters:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "spectrum": ("BOOLEAN", {
                "default": False,
                "tooltip": "About 29% quicker, but can distort people in "
                           "side-by-side comparison (rooms stay fine). Try "
                           "it on your scene before a long render. Needs "
                           "ComfyUI-Spectrum-MiniMax-H3 installed."}),
            "teacache": ("BOOLEAN", {
                "default": False,
                "tooltip": "About 14% quicker at default strength, 21% at "
                           "0.30 - can distort people, like Spectrum. Test "
                           "on your scene first. Needs "
                           "ComfyUI-MiniMaxH3-TeaCache installed."}),
            "teacache_strength": ("FLOAT", {
                "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                "tooltip": "Higher = faster and the image drifts further "
                           "from what you'd get without it. 0.15 is the "
                           "measured default."}),
            "blockcache": ("BOOLEAN", {
                "default": False,
                "tooltip": "Skips model blocks whose input barely changed. "
                           "At the shipped 14 steps it never fires - measured "
                           "0 hits on every run on two cards - because 14 "
                           "steps are too far apart for its 0.12 threshold. "
                           "Only worth turning on at 30+ steps. Ships OFF. "
                           "Needs comfyui-minimax-h3-blockcache-T8 installed "
                           "(without it: a printed link, normal speed)."}),
            "easycache": ("BOOLEAN", {
                "default": False,
                "tooltip": "About 21% quicker and built into ComfyUI - "
                           "nothing to install. Same advice as the others: "
                           "compare one scene with it on and off."}),
            # appended last (2.6.0) so saved panels load unchanged
            "blockcache_threshold": ("FLOAT", {
                "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "How different two consecutive steps may be before "
                           "the block cache stops reusing work. 0.12 (the "
                           "pack's default) never fires at 14 steps; higher "
                           "values fire more often and drift the image more. "
                           "Only meaningful with blockcache ON."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "boost"
    CATEGORY = "video/minimax"
    DESCRIPTION = ("Optional speed-ups from community packs, behind switches "
                   "that are safe to leave off. Every booster changes the "
                   "image slightly (like changing the seed) rather than "
                   "degrading it - test on your own content. Boosters whose "
                   "pack is not installed print an install link and do "
                   "nothing. Measured: stacking Spectrum + TeaCache is no "
                   "faster than Spectrum alone - pick one.")

    def boost(self, model, spectrum, teacache, teacache_strength, blockcache,
              easycache=False, blockcache_threshold=0.12):
        if easycache:
            model = _apply(model, "EasyCache",
                           "built into ComfyUI - update if missing",
                           "EasyCache")
        if teacache:
            model = _apply(model, *_PACKS["teacache"], "TeaCache",
                           rel_l1_thresh=float(teacache_strength))
        if blockcache:
            model = _apply(model, *_PACKS["blockcache"], "block cache",
                           residual_diff_threshold=float(blockcache_threshold))
        if spectrum:
            model = _apply(model, *_PACKS["spectrum"], "Spectrum",
                           enabled=True)
        return (model,)


NODE_CLASS_MAPPINGS = {"H3SpeedBoosters": H3SpeedBoosters}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SpeedBoosters": "H3 Speed Boosters (optional packs)"}
