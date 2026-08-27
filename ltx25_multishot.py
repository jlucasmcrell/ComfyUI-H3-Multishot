"""LTX-2.5 multishot sampler (Joy-LTX 2.5): N shots from the writer, joined by AV-extend.

One node runs the whole two-pass LTX-2.5 pipeline once per shot and joins the shots:

  continue  - the previous shot's last `overlap` latent frames (video AND audio) are pinned
              at the head of the next shot (noise mask 0), the model paints the rest: one
              seamless take across generations, same voice, same room. The replayed head is
              trimmed on output.
  cut       - only the AUDIO tail is pinned: the voice carries straight across a picture cut,
              the picture is free (new angle / framing from the shot's prompt, optionally a
              first-frame image per shot).
  fresh     - nothing pinned; independent shots.

Built on the stock ComfyUI LTX nodes (Conditioning, EmptyLatentAudio, Concat/Separate AV,
DualCFG guider, SamplerCustomAdvanced, LatentUpsampler, tiled decode) - no new math, just the
loop and the masks. Prompts come in as the writer's JSON ({"prompts": [...]}) or a --- list.
"""
import json
import math
import re
import time

import torch

import comfy.utils
import comfy.model_management
import comfy.samplers
import comfy.nested_tensor
import node_helpers
import folder_paths

from nodes import VAEDecodeTiled
from comfy_extras.nodes_custom_sampler import (SamplerCustomAdvanced, RandomNoise, ManualSigmas,
                                                KSamplerSelect)
from comfy_extras.nodes_lt import (LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent,
                                   LTXVDualCFGGuider, LTXVImgToVideo)
from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio, LTXVAudioVAEDecode
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler

_BLOCK_SPLIT = re.compile(r"(?m)^---\s*$")
DIST_SIGMAS_1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DIST_SIGMAS_2 = "0.85, 0.7250, 0.4219, 0.0"


def _first(out):
    """V3 nodes return io.NodeOutput; old nodes return tuples."""
    a = getattr(out, "args", out)
    return a[0] if isinstance(a, (tuple, list)) else a


def _parse_prompts(text):
    text = (text or "").strip()
    if not text:
        return []
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("prompts") or data.get("shots") or []
            if isinstance(data, list):
                out = []
                for item in data:
                    if isinstance(item, dict):
                        item = item.get("prompt") or item.get("text") or ""
                    if str(item).strip():
                        out.append(str(item).strip())
                if out:
                    return out
        except Exception:
            pass
    parts = [p.strip() for p in _BLOCK_SPLIT.split(text) if p.strip()]
    return parts if parts else [text]


class LTX25MultishotSampler:
    """LTX-2.5 Multishot Sampler (Joy-LTX 2.5)."""

    JOINS = ["continue (AV extend: seamless take)", "cut (voice extends, new picture)", "fresh (independent shots)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "prompts": ("STRING", {"multiline": True, "default": "", "tooltip":
                        "The writer's shot prompts: {\"prompts\": [...]} JSON or blocks separated by --- "
                        "(wire the writer here, or paste your own)."}),
            "negative": ("STRING", {"multiline": True, "default":
                         "pc game, console game, video game, cartoon, childish, ugly"}),
            "width": ("INT", {"default": 960, "min": 256, "max": 1920, "step": 32}),
            "height": ("INT", {"default": 544, "min": 256, "max": 1920, "step": 32}),
            "frames_per_shot": ("INT", {"default": 193, "min": 25, "max": 1441, "step": 8, "tooltip":
                                "8n+1 frames per shot at 24 fps (193 = 8 s)."}),
            "shot_count": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip":
                           "0 = every prompt the writer produced; N = the first N."}),
            "join": (cls.JOINS, {"default": cls.JOINS[0]}),
            "overlap": ("INT", {"default": 3, "min": 1, "max": 12, "tooltip":
                        "Latent frames of the previous shot pinned at the head of the next one "
                        "(3 = 17 pixel frames = 0.7 s). More = smoother join, less new content per shot."}),
            "seed": ("INT", {"default": 553010, "min": 0, "max": 0xffffffffffffffff}),
            "seed_per_shot": ("BOOLEAN", {"default": True}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
            "sigmas_pass1": ("STRING", {"default": DIST_SIGMAS_1, "tooltip": "distilled 8-step schedule"}),
            "two_pass": ("BOOLEAN", {"default": True, "tooltip":
                         "Upscale each shot with the latent upsampler and refine (needs upscale_model)."}),
            "sigmas_pass2": ("STRING", {"default": DIST_SIGMAS_2}),
            "video_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
            "audio_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
            "frame_rate": ("FLOAT", {"default": 24.0, "min": 12.0, "max": 60.0, "step": 1.0}),
            "save_every_shot": ("BOOLEAN", {"default": False, "tooltip":
                                "Also write each shot (untrimmed) as output/video/LTX_SHOTS/shot_*.mp4"}),
        }, "optional": {
            "upscale_model": ("LATENT_UPSCALE_MODEL",),
            "start_image": ("IMAGE", {"tooltip": "First frame of shot 1 (image-to-video)."}),
            "shot_images": ("IMAGE", {"tooltip":
                            "One image per shot (batch); used as the first frame of each shot in "
                            "cut/fresh mode (identity carry from your reference plates)."}),
            "image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "run"
    CATEGORY = "video/ltx"
    DESCRIPTION = ("Runs the LTX-2.5 two-pass pipeline once per shot and joins the shots with an "
                   "AV-extend (previous tail pinned as raw latents), so a take can be as long as "
                   "you like and a cut keeps the voice.")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _encode(clip, text):
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)

    @staticmethod
    def _sample(model, positive, negative, latent, sigmas, sampler, seed, video_cfg, audio_cfg):
        guider = _first(LTXVDualCFGGuider.execute(model, positive, negative, video_cfg, audio_cfg))
        noise = _first(RandomNoise.execute(seed))
        out = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)
        return _first(out)

    @staticmethod
    def _tail(av_latent, k_video):
        """Last k video latent frames + the matching audio tail (both raw)."""
        v, a = av_latent["samples"].unbind()
        v = v[:, :, -k_video:].clone().cpu()
        # audio latent (b, c, T, f): take the tail proportional to the pinned pixel span
        n_v = av_latent["samples"].unbind()[0].shape[2]
        k_a = max(1, int(round(a.shape[2] * (k_video / float(n_v)))))
        a = a[:, :, -k_a:].clone().cpu()
        return v, a

    @staticmethod
    def _pin(av, tail, mode):
        """Pin the previous shot's tail (video+audio raw latents) at the head of `av` (noise mask 0)."""
        v, a = av["samples"].unbind()
        v = v.clone()
        a = a.clone()
        if "noise_mask" in av:
            mv, ma = av["noise_mask"].unbind()
            mv = mv.clone()
            ma = ma.clone()
        else:
            mv, ma = torch.ones_like(v), torch.ones_like(a)
        pv, pa = tail
        if mode == "continue" and pv.shape[-2:] == v.shape[-2:]:
            kv = min(pv.shape[2], v.shape[2] - 1)
            v[:, :, :kv] = pv[:, :, -kv:].to(v.device, v.dtype)
            mv[:, :, :kv] = 0.0
        ka = min(pa.shape[2], a.shape[2] - 1)
        a[:, :, :ka] = pa[:, :, -ka:].to(a.device, a.dtype)
        ma[:, :, :ka] = 0.0
        out = dict(av)
        out["samples"] = comfy.nested_tensor.NestedTensor((v, a))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mv, ma))
        return out

    # ------------------------------------------------------------------ main
    def run(self, model, clip, video_vae, audio_vae, prompts, negative, width, height, frames_per_shot,
            shot_count, join, overlap, seed, seed_per_shot, sampler_name, sigmas_pass1, two_pass,
            sigmas_pass2, video_cfg, audio_cfg, frame_rate, save_every_shot,
            upscale_model=None, start_image=None, shot_images=None, image_strength=1.0):
        t0 = time.time()
        shots = _parse_prompts(prompts)
        if not shots:
            raise ValueError("LTX25 Multishot: no shot prompts (the writer output is empty).")
        if shot_count > 0:
            shots = shots[:shot_count]
        n = len(shots)
        frames = ((frames_per_shot - 1) // 8) * 8 + 1
        w, h = (width // 32) * 32, (height // 32) * 32
        mode = "continue" if join.startswith("continue") else ("cut" if join.startswith("cut") else "fresh")
        two_pass = bool(two_pass and upscale_model is not None)
        sampler = _first(KSamplerSelect.execute(sampler_name))
        sig1 = _first(ManualSigmas.execute(sigmas_pass1))
        sig2 = _first(ManualSigmas.execute(sigmas_pass2))
        neg = self._encode(clip, negative)
        head_px = 1 + 8 * (overlap - 1)          # pixel frames the pinned latents decode to
        print(f"[LTX25 Multishot] {n} shot(s) x {frames}f @ {w}x{h} | join={mode} overlap={overlap} "
              f"latent frames (trim {head_px}px on shots 2+) | two_pass={two_pass}", flush=True)

        images_out, audio_out, sr = [], [], None
        prev_tail, prev_tail2 = None, None
        info = []
        for i, text in enumerate(shots):
            s_seed = seed + i if seed_per_shot else seed
            pos = self._encode(clip, text)
            pos, negc = LTXVConditioning.execute(pos, neg, frame_rate).args
            # ---- latents for this shot
            first_img = None
            if i == 0 and start_image is not None:
                first_img = start_image[:1]
            elif shot_images is not None and mode != "continue":
                first_img = shot_images[min(i, shot_images.shape[0] - 1):min(i, shot_images.shape[0] - 1) + 1]
            if first_img is not None:
                pos, negc, vlat = LTXVImgToVideo.execute(pos, negc, first_img, video_vae, w, h, frames, 1,
                                                         image_strength).args
            else:
                vlat = {"samples": torch.zeros([1, 128, (frames - 1) // 8 + 1, h // 32, w // 32],
                                               device=comfy.model_management.intermediate_device())}
            alat = _first(LTXVEmptyLatentAudio.execute(frames, frame_rate, 1, audio_vae))
            av = _first(LTXVConcatAVLatent.execute(vlat, alat))
            # ---- pin the previous tail (AV extend), pass-1 grid
            if prev_tail is not None and mode != "fresh":
                av = self._pin(av, prev_tail, mode)
            # ---- pass 1
            ts = time.time()
            out1 = self._sample(model, pos, negc, av, sig1, sampler, s_seed, video_cfg, audio_cfg)
            prev_tail = self._tail(out1, overlap)      # pin from PASS-1 latents (same grid as the next shot)
            final = out1
            # ---- pass 2 (upscale + refine)
            if two_pass:
                v1, a1 = LTXVSeparateAVLatent.execute(out1).args
                up = _first(LTXVLatentUpsampler.execute(v1, upscale_model, video_vae))
                av2 = _first(LTXVConcatAVLatent.execute(up, a1))
                # pin the previous shot's REFINED tail too, so pass 2 does not re-draw the join
                if prev_tail2 is not None and mode != "fresh":
                    av2 = self._pin(av2, prev_tail2, mode)
                final = self._sample(model, pos, negc, av2, sig2, sampler, s_seed, video_cfg, audio_cfg)
                prev_tail2 = self._tail(final, overlap)
            # ---- decode
            vfin, afin = LTXVSeparateAVLatent.execute(final).args
            dec = VAEDecodeTiled().decode(video_vae, vfin, 512, 64, 64, 16)
            imgs = dec[0] if isinstance(dec, (tuple, list)) else _first(dec)
            aud = _first(LTXVAudioVAEDecode.execute(afin, audio_vae))
            sr = aud["sample_rate"]
            wav = aud["waveform"]
            if save_every_shot:
                self._save_shot(imgs, aud, frame_rate, i)
            # ---- trim the replayed head on shots 2+
            if i > 0 and mode != "fresh":
                imgs = imgs[head_px:]
                cut = int(round(head_px / frame_rate * sr))
                wav = wav[..., cut:]
            # keep the audio exactly as long as the video
            want = int(round(imgs.shape[0] / frame_rate * sr))
            if wav.shape[-1] > want:
                wav = wav[..., :want]
            elif wav.shape[-1] < want:
                wav = torch.nn.functional.pad(wav, (0, want - wav.shape[-1]))
            images_out.append(imgs.cpu())
            audio_out.append(wav.cpu())
            info.append(f"shot {i+1}/{n}: {imgs.shape[0]}f in {time.time()-ts:.0f}s")
            print(f"[LTX25 Multishot] {info[-1]}", flush=True)
            comfy.model_management.soft_empty_cache()

        images = torch.cat(images_out, dim=0)
        audio = {"waveform": torch.cat(audio_out, dim=-1), "sample_rate": sr}
        total = f"{n} shots -> {images.shape[0]} frames (~{images.shape[0]/frame_rate:.1f}s) in {time.time()-t0:.0f}s"
        print(f"[LTX25 Multishot] done: {total}", flush=True)
        return (images, audio, total + "\n" + "\n".join(info))

    @staticmethod
    def _save_shot(images, audio, fps, idx):
        try:
            import os
            from comfy_extras.nodes_video import CreateVideo
            vid = _first(CreateVideo.execute(images, float(fps), audio))
            folder = os.path.join(folder_paths.get_output_directory(), "video", "LTX_SHOTS")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "shot_%02d_%d.mp4" % (idx + 1, int(time.time())))
            vid.save_to(path)
            print(f"[LTX25 Multishot] shot {idx+1} saved -> {path}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[LTX25 Multishot] per-shot save skipped ({e})", flush=True)


NODE_CLASS_MAPPINGS = {"LTX25MultishotSampler": LTX25MultishotSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"LTX25MultishotSampler": "LTX-2.5 Multishot Sampler (Joy-LTX 2.5)"}
