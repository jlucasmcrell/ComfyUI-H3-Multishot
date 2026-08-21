"""Small in-process fixes for ComfyUI's LTX latent upsampler.

Rational (x1.5 / x0.75) upsamplers end in a BlurDownsample whose 5x5 kernel
is a plain registered buffer. When the upsampler is only *partially* loaded
(a 24-32 GB card that still has the 20 GB DiT resident), ComfyUI casts the
conv weights to the GPU on the fly but leaves that buffer on the CPU, and
the pass-2 upscale dies with "Expected all tensors to be on the same device
... conv_depthwise2d". Seen 2026-08-17 on a 30 s 1280x736 take, x1.5.
Fix: move the kernel to the input's device/dtype at call time.
"""
import logging

try:
    import torch.nn.functional as F
    from einops import rearrange
    from comfy.ldm.lightricks import latent_upsampler as _lu

    def _blur_forward(self, x):
        if self.stride == 1:
            return x

        def _apply_2d(x2d):
            C = x2d.shape[1]
            weight = self.kernel.to(device=x2d.device, dtype=x2d.dtype).expand(C, 1, 5, 5)
            return F.conv2d(x2d, weight=weight, bias=None, stride=self.stride,
                            padding=2, groups=C)

        if self.dims == 2:
            return _apply_2d(x)
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = _apply_2d(x)
        h2, w2 = x.shape[-2:]
        return rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f, h=h2, w=w2)

    if not getattr(_lu.BlurDownsample, "_h3_device_safe", False):
        _lu.BlurDownsample.forward = _blur_forward
        _lu.BlurDownsample._h3_device_safe = True
        logging.info("[H3-Multishot] LTX rational upsampler: blur kernel follows "
                     "the input device (partial-load safe)")
except Exception as _e:                                   # pragma: no cover
    logging.info("[H3-Multishot] LTX upsampler patch not installed (%s)", _e)
