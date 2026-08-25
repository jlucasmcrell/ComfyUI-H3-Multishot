"""H3ChainNormalize - level a finished chain's texture, colour and exposure.

Chained multishot drifts: each window's anchor is the previous window's tail,
so structure energy creeps up and the grade wanders. In-loop controls damp it
but never fully remove it; this node fixes what survives, on the decoded
frames, right before they become a video.

Method (the best of two approaches, measured on master_00131 2026-08-25):
  * every frame is HISTOGRAM-MATCHED to a shot-1 reference frame, which pulls
    colour and exposure drift back - the half of the problem no sharpening
    control ever touched (idea from Gemini's colormatch.py);
  * the softening amount is EMA-smoothed so corrections ease in rather than
    stepping at a window boundary (same source);
  * the correction is applied ONLY to the 5-17px STRUCTURE band, so grain and
    sensor noise pass through untouched. A whole-frame blur flattens drift
    just as well but costs ~20%% of the fine band - the camcorder texture;
  * the baseline is the MEDIAN of shot 1 AFTER its opening exposure fade
    settles, with a deadband, so shot 1 is never softened against its own
    average.

Measured on the 84s lamp-lit gauge: drift 1.49x -> 1.22x with fine-band grain
at or above the original, versus 1.21x with 20%% grain loss for the
whole-frame version.

Drop between the sampler's IMAGE output and CreateVideo.
"""
import torch

_CAT = "video/minimax"


def _box(x, r):
    """Box blur over [B,H,W,C] via separable cumulative sums (edge-padded)."""
    b, h, w, c = x.shape
    pad = torch.nn.functional.pad(
        x.permute(0, 3, 1, 2), (r, r, r, r), mode="replicate").permute(0, 2, 3, 1)
    cs = pad.cumsum(1)
    cs = torch.cat([torch.zeros_like(cs[:, :1]), cs], 1)
    x = (cs[:, 2 * r + 1:] - cs[:, :-2 * r - 1]) / float(2 * r + 1)
    cs = x.cumsum(2)
    cs = torch.cat([torch.zeros_like(cs[:, :, :1]), cs], 2)
    return (cs[:, :, 2 * r + 1:] - cs[:, :, :-2 * r - 1]) / float(2 * r + 1)


def _norm_lap(gray):
    """Contrast-normalised Laplacian energy for one [H,W] frame."""
    k = (gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1]
         - gray[1:-1, :-2] - gray[1:-1, 2:])
    return float((k ** 2).mean() / torch.clamp(gray.std() ** 2, min=1e-9))


def _cdf(frame_u8_channel):
    hist = torch.bincount(frame_u8_channel.reshape(-1), minlength=256).float()
    return torch.cumsum(hist, 0) / torch.clamp(hist.sum(), min=1.0)


def _hist_match(frame, ref_cdfs):
    """Match one [H,W,3] float frame (0..1) to precomputed reference CDFs."""
    out = torch.empty_like(frame)
    idx = torch.arange(256, device=frame.device, dtype=torch.float32)
    for c in range(3):
        src = torch.clamp(frame[..., c] * 255.0, 0, 255).to(torch.uint8)
        s_cdf = _cdf(src)
        # for each source level, the reference level with the nearest CDF
        pos = torch.searchsorted(ref_cdfs[c].contiguous(), s_cdf.contiguous())
        lut = torch.clamp(pos.float(), 0, 255)
        lut = torch.where(torch.isfinite(lut), lut, idx)
        out[..., c] = lut[src.long()] / 255.0
    return out


class H3ChainNormalize:
    """Level texture, colour and exposure across a chained take."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "baseline_seconds": ("FLOAT", {
                    "default": 10.0, "min": 2.0, "max": 30.0, "step": 0.5,
                    "tooltip": "How much of shot 1 sets the house level."}),
                "skip_seconds": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Skipped at the very start. H3 opens with an "
                               "exposure fade-in (luma climbs for ~1.7s); "
                               "including it drags the baseline dark and "
                               "makes the whole piece get softened."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "How hard excess structure is pulled back. "
                               "1.0 matched the eye-approved result."}),
                "deadband": ("FLOAT", {
                    "default": 1.06, "min": 1.0, "max": 1.5, "step": 0.01,
                    "tooltip": "Frames within this ratio of the baseline are "
                               "left alone, so shot 1 stays untouched."}),
                "ema": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Smoothing on the correction. Low values ease "
                               "it in so nothing pops at a window join."}),
                "colour_match": ("BOOLEAN", {
                    "default": True, "label_on": "match colour to shot 1",
                    "label_off": "texture only",
                    "tooltip": "Histogram-match every frame to a shot-1 "
                               "reference. This is what fixes the grade "
                               "wandering across a long chain."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = _CAT

    def run(self, images, baseline_seconds, skip_seconds, fps, strength,
            deadband, ema, colour_match):
        n = images.shape[0]
        if n < 8:
            print("[H3ChainNormalize] %d frame(s) - nothing to level" % n,
                  flush=True)
            return (images,)
        dev = images.device
        skip = min(n - 1, int(skip_seconds * fps))
        take = min(n, skip + max(8, int(baseline_seconds * fps)))

        laps = [_norm_lap(images[i].mean(-1)) for i in range(skip, take)]
        laps_sorted = sorted(laps)
        ref = laps_sorted[len(laps_sorted) // 2]
        ref_idx = skip + (take - skip) // 2
        ref_cdfs = None
        if colour_match:
            rf = torch.clamp(images[ref_idx] * 255.0, 0, 255).to(torch.uint8)
            ref_cdfs = [_cdf(rf[..., c]) for c in range(3)]

        out = torch.empty_like(images)
        sigma = 0.0
        applied = []
        for i in range(n):
            fr = images[i]
            if ref_cdfs is not None:
                fr = _hist_match(fr, ref_cdfs)
            cur = _norm_lap(fr.mean(-1))
            target = max(0.0, (cur / max(ref, 1e-9)) - deadband) * strength
            sigma = ema * target + (1.0 - ema) * sigma
            s = min(0.85, sigma)
            if s > 0.02:
                b = fr.unsqueeze(0)
                band = _box(b, 2) - _box(b, 8)          # ~5px minus ~17px
                fr = torch.clamp(b - s * band, 0, 1).squeeze(0)
            applied.append(s)
            out[i] = fr
        tail = applied[-int(2 * fps):] or [0.0]
        print("[H3ChainNormalize] baseline %.5f (median of shot 1 after %.1fs)"
              " | colour %s | correction max %.3f, last-2s mean %.3f over %d "
              "frames" % (ref, skip_seconds, "matched" if colour_match else "off",
                          max(applied), sum(tail) / len(tail), n), flush=True)
        return (out.to(dev),)


NODE_CLASS_MAPPINGS = {"H3ChainNormalize": H3ChainNormalize}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ChainNormalize": "H3 Chain Normalize (texture + colour)"}
