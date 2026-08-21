"""H3 Retake - regenerate one stretch of a finished H3 clip and keep the rest.

Feed it a rendered clip (frames + audio) and a time window. Everything outside the window is
frozen as raw latents (noise mask 0); only the window is denoised, from a prompt written for
that moment. Video and audio are independent:

  video + audio  - redo the moment completely.
  video only     - keep the performance: voice, timing and room tone untouched.
  audio only     - keep the picture, change the line. Lips are whatever was rendered, so keep
                   the new line about as long as the old one.

Port of the LTX-2.5 retake to H3's AV latent: video [1,24,T,h/16,w/16] on the 17k+5 frame
grid, audio [1,32,2,Ta] at 40 latent fps (time is the LAST axis on the audio side - the one
real difference from the LTX port). H3 samples with a BasicGuider at cfg 1, so there is no
negative branch; wire the same SAMPLER and SIGMAS the clip was rendered with.
"""
import time

import torch

import comfy.model_management
import comfy.nested_tensor
import node_helpers

from comfy_extras import nodes_custom_sampler as ncs

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

MODES = ["video + audio (redo the moment)",
         "video only (keep the performance)",
         "audio only (keep the picture)"]


def _mm_utils():
    """h3_multishot_utils + the comfy H3 module, with the pack's loose-install fallback."""
    try:
        from . import h3_multishot_utils as u
    except ImportError:
        try:
            import h3_multishot_utils as u
        except ImportError:
            import importlib.util as _ilu, os as _os
            _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "h3_multishot_utils.py")
            _s = _ilu.spec_from_file_location("h3_multishot_utils", _p)
            u = _ilu.module_from_spec(_s)
            _s.loader.exec_module(u)
    from comfy_extras import nodes_minimax_h3 as mmh3
    return u, mmh3


class H3Retake:
    """Redo one time window of a finished H3 clip (picture, sound, or both)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "images": ("IMAGE", {"tooltip": "The finished clip's frames, in order."}),
            "audio": ("AUDIO", {"tooltip": "That clip's audio - the same take as the frames."}),
            "prompt": ("STRING", {"multiline": True, "default": "", "tooltip":
                       "What should happen in the window. Write it like a shot prompt; the model "
                       "sees only this text plus the frozen material either side."}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1, "tooltip":
                              "Where the retake starts. Snapped to H3's latent grid (~0.14 s per slot)."}),
            "end_seconds": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 600.0, "step": 0.1}),
            "mode": (MODES, {"default": MODES[0]}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "sampler": ("SAMPLER", {"tooltip": "The sampler the clip was rendered with (euler)."}),
            "sigmas": ("SIGMAS", {"tooltip": "The schedule the clip was rendered with (beta, 10-12 steps; "
                                  "wire the same sigma-shift chain as the render canvas)."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "run"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = "Redo one stretch of a finished H3 clip - picture, sound, or both - and keep the rest."

    @staticmethod
    def _window(n_latent, total_seconds, start_s, end_s):
        per = total_seconds / float(max(1, n_latent))     # seconds per latent slot
        a = int(max(0.0, start_s) / per)
        b = int(round(min(end_s, total_seconds) / per + 0.5))
        a = max(0, min(a, n_latent - 1))
        b = max(a + 1, min(b, n_latent))
        return a, b

    def run(self, model, clip, video_vae, audio_vae, images, audio, prompt,
            start_seconds, end_seconds, mode, seed, sampler, sigmas):
        if end_seconds <= start_seconds:
            raise ValueError("H3 Retake: end_seconds must be after start_seconds.")
        u, mmh3 = _mm_utils()
        t0 = time.time()
        do_video = not mode.startswith("audio only")
        do_audio = not mode.startswith("video only")

        # ---- frames onto H3's 17k+5 grid, canvas to /32 (a finished render already is)
        frames = images[..., :3]
        n_px = frames.shape[0]
        keep = 5 + ((n_px - 5) // 17) * 17 if n_px >= 22 else n_px
        if keep != n_px:
            print("[H3Retake] %d frames -> %d (H3's 17k+5 grid; the tail beyond the grid is "
                  "re-appended untouched after the retake)" % (n_px, keep), flush=True)
        grid_frames = frames[:keep]
        h, w = grid_frames.shape[1], grid_frames.shape[2]
        if h % 32 or w % 32:
            tw, th = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
            grid_frames = mmh3._resize(grid_frames, tw, th, "disabled")
        total_seconds = keep / float(mmh3.FPS)

        # ---- encode both sides to raw latents
        vz = video_vae.encode(grid_frames)                          # [1, 24, T, h/16, w/16]
        wav, _sr = u._wav_for_vae(audio_vae, audio, "retake audio")
        az = audio_vae.encode(wav.movedim(1, -1))                   # [1, 32, 2, Ta] - time LAST
        # audio no longer than the video window
        ta_want = round(total_seconds * mmh3.AUDIO_LATENT_FPS)
        if az.shape[-1] > ta_want:
            az = az[..., :ta_want]

        mv, ma = torch.zeros_like(vz), torch.zeros_like(az)
        vw = aw = None
        if do_video:
            i, j = self._window(vz.shape[2], total_seconds, start_seconds, end_seconds)
            mv[:, :, i:j] = 1.0
            vw = (i, j, vz.shape[2])
        if do_audio:
            i, j = self._window(az.shape[-1], total_seconds, start_seconds, end_seconds)
            ma[..., i:j] = 1.0
            aw = (i, j, az.shape[-1])

        latent = {"samples": comfy.nested_tensor.NestedTensor((vz, az)),
                  "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma))}

        # ---- H3 conditioning: cfg 1, BasicGuider, no negative branch
        tokens = clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)
        guider = ncs.BasicGuider().get_guider(model, cond)[0]
        noise = ncs.RandomNoise().get_noise(seed)[0]
        out, _denoised = ncs.SamplerCustomAdvanced().sample(noise, guider, sampler, sigmas, latent)

        # ---- decode
        lat = out["samples"]
        if getattr(lat, "is_nested", False):
            lat = lat.unbind()[0]
        imgs = video_vae.decode(lat)
        if imgs.ndim == 5:
            imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
        from comfy_extras.nodes_audio import vae_decode_audio
        aud = vae_decode_audio(audio_vae, out)
        if keep != n_px:                                            # re-append the off-grid tail untouched
            imgs = torch.cat([imgs.cpu(), frames[keep:].cpu()], 0)

        info = ("retake %.1f-%.1f s of a %.1f s clip | %s | video slots %s | audio slots %s | %.0f s"
                % (start_seconds, end_seconds, n_px / float(mmh3.FPS), mode,
                   ("%d-%d of %d" % vw) if vw else "frozen",
                   ("%d-%d of %d" % aw) if aw else "frozen", time.time() - t0))
        print("[H3Retake] " + info, flush=True)
        comfy.model_management.soft_empty_cache()
        return (imgs, aud, info)


NODE_CLASS_MAPPINGS["H3Retake"] = H3Retake
NODE_DISPLAY_NAME_MAPPINGS["H3Retake"] = "H3 Retake (redo part of a clip)"
