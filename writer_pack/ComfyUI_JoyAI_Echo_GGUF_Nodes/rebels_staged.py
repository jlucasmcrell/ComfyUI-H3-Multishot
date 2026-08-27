"""Rebels JE - Staged single-node pipeline.

Why this exists
---------------
The discrete loader graph (Config -> DiTLoader + TextEncoder + VAELoader ->
Assemble -> TextEncode -> SingleShotGenerate) loads EVERY component and holds
it for the whole ComfyUI run. On a 16GB-RAM box that means the fp8 Gemma
(~12GB) and the DiT (~9GB quantized) sit in RAM at the same time -> >16GB ->
pagefile thrash -> numpy ArrayMemoryError during DiT dequant.

This node fixes that by STAGING inside a single function so we control the
load/free order:

    Stage 1  build Gemma + connector, encode the prompt -> small cond, FREE Gemma
    Stage 2  load DiT (GGUF) + VAEs   (Gemma is already gone)
    Stage 3  denoise + decode using the precomputed cond (no text encoder needed)

Peak system RAM becomes the largest SINGLE component (~12GB for Gemma during
encode, ~9-12GB for the DiT during denoise) instead of the sum (~25GB).

It reuses the existing, working loader classes from rebels_loaders.py and the
real BidirectionalAVInferencePipeline denoise/decode path from nodes.py
(generate_shot, Phase A + Phase B) -- only the in-line encode block is dropped,
because we already have the conditioning.

NOTE: single-shot, no incoming memory bank (fresh, empty -> base pipeline path,
so VAE encoders are not needed -> with_encoders=False saves more RAM).
"""

from __future__ import annotations

import gc
import torch

# Reuse the working loaders + helpers already in the package.
from .rebels_loaders import (
    RebelsJE_Config,
    RebelsJE_TextEncoder,
    RebelsJE_DiTLoader,
    RebelsJE_VAELoader,
    _dev,
)
from .rebels_loaders import gpu_dequant_supported
from gguf import GGMLQuantizationType as QT
from .nodes import SequentialOffloader, DENOISING_SIGMAS, _empty_cache, _move

try:
    from .rebels_loaders import CAT as _CAT
except Exception:
    _CAT = "JoyAI-Echo/Rebels"

import os

# ComfyUI's model-folder registry. Present at runtime; absent in a bare sandbox.
try:
    import folder_paths
except Exception:
    folder_paths = None


def _model_dirs(keys):
    """Resolve ComfyUI folder keys to actual directories. Several keys may map to
    the same physical dir (e.g. 'diffusion_models' includes models/unet), so we
    de-dupe. Unknown keys are ignored."""
    dirs = []
    if folder_paths is None:
        return dirs
    for key in keys:
        try:
            paths = folder_paths.get_folder_paths(key)
        except Exception:
            paths = []
        for d in paths:
            if d not in dirs and os.path.isdir(d):
                dirs.append(d)
    # Hard fallback: derive from models_dir if the registry gave us nothing.
    if not dirs and folder_paths is not None:
        base = getattr(folder_paths, "models_dir", None)
        if base:
            for key in keys:
                d = os.path.join(base, key)
                if os.path.isdir(d) and d not in dirs:
                    dirs.append(d)
    return dirs


def _scan(keys, exts):
    """Return {relative_name: full_path} for files under any directory the given
    folder keys map to, matched by extension. We walk the directories ourselves
    instead of using folder_paths.get_filename_list, because that method filters
    by each folder's *registered* extension set -- and .gguf is NOT registered for
    'unet'/'diffusion_models', so GGUF files would be silently hidden."""
    out = {}
    for d in _model_dirs(keys):
        for root, _, files in os.walk(d):
            for fn in files:
                if exts and not fn.lower().endswith(exts):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, d).replace("\\", "/")
                out.setdefault(rel, full)
    return out


def _combo(mapping, hint):
    """A non-empty, sorted name list so the COMBO always renders; if nothing is
    found yet, show a hint telling the user which folder to populate."""
    names = sorted(mapping.keys())
    return names if names else [hint]


def _resolve(keys, exts, name):
    """Map a dropdown selection back to a full path by re-scanning. Absolute paths
    and hint placeholders pass straight through so the loader raises a clear
    'file not found' rather than something cryptic."""
    if not name:
        return name
    if os.path.isabs(name):
        return name
    mapping = _scan(keys, exts)
    if name in mapping:
        return mapping[name]
    # tolerate basename-only selections
    for rel, full in mapping.items():
        if os.path.basename(rel) == name:
            return full
    return name


# Folder keys + extensions per field. DiT covers all the places a .gguf might live.
_DIT_KEYS = ["diffusion_models", "unet", "unet_gguf"]
_TE_KEYS = ["text_encoders"]
_VAE_KEYS = ["vae"]
_VOC_KEYS = ["audio_encoders"]
_GGUF = (".gguf",)
_ST = (".safetensors",)

# ---------------------------------------------------------------- run caches
# ComfyUI keeps regular models loaded between runs; this node was rebuilding
# EVERYTHING every queue (15-18 min Gemma encode + 8-12 min DiT build before a
# single denoise step). These module-level caches survive between runs:
#   _COND_CACHE: prompt embeddings keyed by (gemma, connector, prompt)
#   _MODEL_CACHE: built generator + VAEs keyed by the model files + resolution
# Everything cached lives on CPU between runs; the packed DiT weights are
# memory-mapped so the cached generator costs almost no resident RAM.
_COND_CACHE = {}
_MODEL_CACHE = {}


# Default config: the JSON shipped inside the node pack's configs/ folder (created
# once via dump_joyai_config.py). Falls back to the raw checkpoint path if absent,
# so it works on the dev box before the JSON is dumped, and on every viewer's box
# after, with no 46GB dependency.
# The architecture config ships inside the node pack at configs/joyai_echo_config.json.
# This path is derived from THIS file's location, so it is correct on any drive, any
# OS, for any user who installed the pack -- no machine-specific paths.
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_CFG = os.path.join(_NODE_DIR, "configs", "joyai_echo_config.json")


def _default_config():
    # Always point at the bundled config. If a user wants to read the arch from the
    # full checkpoint instead, they can paste that path into the field manually.
    return _BUNDLED_CFG


def _tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, max(1, stride)))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return sorted(set(starts))


class _SpatialTiledDecoder(torch.nn.Module):
    """Wraps the LTX video VAE decoder with spatially TILED decoding.

    Why: at 480x480+ the decoder's intermediate feature maps blow past 8GB
    VRAM and the Windows driver spills to system RAM at ~100x slowdown (the
    40-80 minute "decode hangs"). Spatial tiles with feathered overlap cut the
    intermediates by the tile ratio with no visible seams. Tiling is spatial
    ONLY: this VAE is temporally causal, so the time axis stays whole.
    decode_benchmark_sample calls video_vae.decoder(latent) internally, so
    swapping the decoder for this proxy makes JD's own decode path tiled
    without touching any of their code."""

    def __init__(self, decoder, tile=12, overlap=3):
        super().__init__()
        self._dec = decoder
        self._tile = max(4, int(tile))
        self._ov = max(1, min(int(overlap), self._tile // 2))

    def _feather(self, h, w, top, bottom, left, right, ov, device):
        ry = torch.ones(h, device=device)
        rx = torch.ones(w, device=device)
        ramp = torch.linspace(0.0, 1.0, ov, device=device)
        if top:    ry[:ov] = torch.minimum(ry[:ov], ramp)
        if bottom: ry[-ov:] = torch.minimum(ry[-ov:], ramp.flip(0))
        if left:   rx[:ov] = torch.minimum(rx[:ov], ramp)
        if right:  rx[-ov:] = torch.minimum(rx[-ov:], ramp.flip(0))
        return (ry[:, None] * rx[None, :]).view(1, 1, 1, h, w)

    def forward(self, latent, *a, **k):
        if not (torch.is_tensor(latent) and latent.dim() == 5):
            return self._dec(latent, *a, **k)
        B, C, T, H, W = latent.shape
        t, ov = self._tile, self._ov
        if H <= t and W <= t:
            return self._dec(latent, *a, **k)
        ys = _tile_starts(H, t, t - ov)
        xs = _tile_starts(W, t, t - ov)
        out = wmap = None
        scale = None
        for y0 in ys:
            for x0 in xs:
                th = min(t, H - y0)
                tw = min(t, W - x0)
                o = self._dec(latent[..., y0:y0 + th, x0:x0 + tw], *a, **k)
                if not torch.is_tensor(o):           # unexpected API: bail to full decode
                    return self._dec(latent, *a, **k)
                if scale is None:
                    scale = o.shape[-1] // tw
                    out = torch.zeros((o.shape[0], o.shape[1], o.shape[2],
                                       H * scale, W * scale),
                                      dtype=torch.float32, device=o.device)
                    wmap = torch.zeros((1, 1, 1, H * scale, W * scale),
                                       dtype=torch.float32, device=o.device)
                wt = self._feather(o.shape[-2], o.shape[-1],
                                   top=y0 > 0, bottom=y0 + th < H,
                                   left=x0 > 0, right=x0 + tw < W,
                                   ov=ov * scale, device=o.device)
                out[..., y0 * scale:y0 * scale + o.shape[-2],
                         x0 * scale:x0 * scale + o.shape[-1]] += o.float() * wt
                wmap[..., y0 * scale:y0 * scale + o.shape[-2],
                          x0 * scale:x0 * scale + o.shape[-1]] += wt
                del o, wt
                _empty_cache()
        return (out / wmap.clamp_min(1e-6)).to(latent.dtype)


class RebelsJE_StagedPipeline:
    """One node: encode -> free Gemma -> load DiT/VAE -> denoise -> decode."""

    CATEGORY = _CAT
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        dit = _combo(_scan(_DIT_KEYS, _GGUF), "<put DiT .gguf in models/unet>")
        te = _combo(_scan(_TE_KEYS, _ST + _GGUF), "<put encoders in models/text_encoders>")
        vae = _combo(_scan(_VAE_KEYS, _ST), "<put VAEs in models/vae>")
        voc = _combo(_scan(_VOC_KEYS, _ST), "<put vocoder in models/audio_encoders>")
        return {
            "required": {
                # Arch config is bundled with the pack (configs/joyai_echo_config.json)
                # and loaded automatically -- no user input needed.
                "dit_gguf": (dit,),
                "gemma_path": (te,),
                "gemma_format": (["our_fp8", "bf16"], {"default": "our_fp8"}),
                "connector_path": (te,),
                "video_vae_path": (vae,),
                "audio_vae_path": (vae,),
                "vocoder_path": (voc,),
                "prompt": ("STRING", {"default": "a woman walks down a busy street at sunset", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "subtitles, captions, on-screen text, watermark", "multiline": True,
                    "tooltip": "EXPERIMENTAL. The distilled (DMD) pipeline has no CFG, so this works by "
                               "extrapolating the prompt embeddings away from the negative embeddings. "
                               "Set negative_scale to 0 to disable entirely."}),
                "negative_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "0 = off (negative prompt ignored). Try 0.3-0.8. Higher pushes harder "
                               "away from the negative but can degrade the image."}),
                "seed": ("INT", {"default": 12345}),
                "num_frames": ("INT", {"default": 25, "min": 9, "max": 257}),
                "video_height": ("INT", {"default": 512, "min": 64, "max": 1280, "step": 32}),
                "video_width": ("INT", {"default": 768, "min": 64, "max": 1280, "step": 32}),
                "video_fps": ("INT", {"default": 24}),
                "audio_sample_rate": ("INT", {"default": 24000}),
                # OFF = GGUF encoders run on the GPU (streamed, ~10x faster).
                # fp8/bf16 encoders auto-force CPU regardless (12GB won't fit 8GB VRAM).
                "encode_on_cpu": ("BOOLEAN", {"default": False}),
                # Stream one DiT block to GPU at a time so denoise fits 8GB VRAM.
                "sequential_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Spatially tiled VAE decode. At 480px+ the decoder's feature
                # maps overflow 8GB VRAM and Windows spills to system RAM at
                # ~100x slowdown (40-80 min "decode hangs"). Tiling fixes it.
                "tiled_decode": ("BOOLEAN", {"default": True}),
                "decode_tile_size": ("INT", {"default": 12, "min": 6, "max": 40,
                    "tooltip": "Tile size in LATENT units (x32 = pixels). 12 = 384px "
                               "tiles. Lower if decode still overflows VRAM."}),
            }
        }

    # ------------------------------------------------------------------ run
    def run(self, dit_gguf, gemma_path, gemma_format, connector_path,
            video_vae_path, audio_vae_path, vocoder_path, prompt,
            negative_prompt, negative_scale, seed, num_frames,
            video_height, video_width, video_fps, audio_sample_rate,
            encode_on_cpu, sequential_offload,
            tiled_decode=True, decode_tile_size=12):
        config_source = _default_config()  # bundled with the pack; no UI input

        device = _dev()
        dtype = torch.bfloat16
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt is empty.")

        # Dropdown widgets hand back names relative to a model folder; turn them
        # into full paths for the loaders. (config_source is already a full path.)
        dit_gguf       = _resolve(_DIT_KEYS, _GGUF, dit_gguf)
        gemma_path     = _resolve(_TE_KEYS, _ST + _GGUF, gemma_path)
        connector_path = _resolve(_TE_KEYS, _ST, connector_path)
        video_vae_path = _resolve(_VAE_KEYS, _ST, video_vae_path)
        audio_vae_path = _resolve(_VAE_KEYS, _ST, audio_vae_path)
        vocoder_path   = _resolve(_VOC_KEYS, _ST, vocoder_path)

        cfg = RebelsJE_Config().run(config_source)[0]

        # ============================================================ STAGE 1
        # CONDITIONING CACHE: the CPU Gemma encode is slow and only depends on
        # (gemma weights, connector, text). Cached per TEXT, so positive and
        # negative prompts each get their own entry and seed/setting re-runs
        # skip Stage 1 entirely.
        negative_prompt = (negative_prompt or "").strip()
        use_negative = bool(negative_prompt) and float(negative_scale) > 0.0
        _te_holder = {"te": None}

        def _encode_text(text):
            key = (gemma_path, gemma_format, connector_path, text)
            hit = _COND_CACHE.get(key)
            if hit is not None:
                print(f"[StagedJE] Stage 1/3: cached conditioning hit for {text[:40]!r}...", flush=True)
                return hit
            te = _te_holder["te"]
            if te is None:
                print("[StagedJE] Stage 1/3: building text encoder (Gemma)...", flush=True)
                te = RebelsJE_TextEncoder().run(cfg, gemma_path, gemma_format, connector_path, True)[0]
                # META-TENSOR FIX: the text-only Gemma file has no vision_tower /
                # multi_modal_projector / lm_head, so those stay meta and poison
                # model.device. None are used by text encode -- drop them.
                try:
                    gm = getattr(getattr(te, "text_encoder", None), "model", None)
                    if gm is not None:
                        inner = getattr(gm, "model", None)
                        for parent, attr in ((inner, "vision_tower"),
                                             (inner, "multi_modal_projector"),
                                             (gm, "lm_head")):
                            if parent is not None and getattr(parent, attr, None) is not None:
                                try:
                                    setattr(parent, attr, None)
                                except Exception:
                                    pass
                        try:
                            dev0 = next(gm.parameters()).device
                            print(f"[StagedJE] Stage 1/3: encoder device after meta-strip = {dev0}", flush=True)
                        except StopIteration:
                            pass
                except Exception as e:
                    print(f"[StagedJE] meta-strip skipped: {e}", flush=True)
                _is_gguf_enc = str(gemma_path).lower().endswith(".gguf")
                _gpu_encode = (not encode_on_cpu) and _is_gguf_enc and torch.cuda.is_available()
                if (not encode_on_cpu) and not _is_gguf_enc:
                    print("[StagedJE] fp8/bf16 encoder cannot fit 8GB VRAM -- "
                          "encoding on CPU anyway. Use the GGUF encoder for GPU "
                          "encoding.", flush=True)
                if _gpu_encode:
                    # PRE-FLIGHT: verify city96 has GPU kernels for every quant
                    # type in this encoder. Legacy types (e.g. Q5_1) may have no
                    # kernel -- without this check every layer silently falls to
                    # CPU numpy dequant while activations sit on GPU, and a
                    # 1-minute encode becomes a 30+ minute grind.
                    _qtypes = set()
                    for _m in getattr(te, "text_encoder", torch.nn.Module()).modules():
                        _qv = getattr(_m, "qtype_value", None)
                        if _qv is not None:
                            _qtypes.add(int(_qv))
                    _missing = sorted(q for q in _qtypes if not gpu_dequant_supported(q))
                    if _missing:
                        try:
                            _names = [QT(q).name for q in _missing]
                        except Exception:
                            _names = [str(q) for q in _missing]
                        print(f"[StagedJE] GPU encode DISABLED: no GPU dequant kernels "
                              f"for {_names} in this encoder. Falling back to CPU encode. "
                              f"Fix: requant the Gemma encoder to Q8_0 (universally "
                              f"supported, fastest dequant):\n"
                              f"  make_gemma_gguf.py --quant Q8_0", flush=True)
                        _gpu_encode = False
                if _gpu_encode:
                    # GGUF encoder on GPU: packed linears stay memory-mapped and
                    # stream per-layer (same GGUFLinear machinery the DiT uses).
                    # Only ~2.5GB of embeddings/norms move resident -- fits 8GB,
                    # and encodes roughly an order of magnitude faster than CPU.
                    try:
                        te.device = device
                    except Exception:
                        pass
                    for attr in ("text_encoder", "embeddings_processor"):
                        m = getattr(te, attr, None)
                        if m is not None and hasattr(m, "to"):
                            try:
                                m.to(device)
                            except Exception:
                                pass
                    print("[StagedJE] Stage 1/3: GGUF encoder on GPU "
                          "(streamed per-layer dequant).", flush=True)
                else:
                    if _is_gguf_enc and encode_on_cpu:
                        print("[StagedJE] tip: with a GGUF encoder you can set "
                              "encode_on_cpu=False for ~10x faster Stage 1.", flush=True)
                    try:
                        te.device = torch.device("cpu")
                    except Exception:
                        pass
                    for attr in ("text_encoder", "embeddings_processor"):
                        m = getattr(te, attr, None)
                        if m is not None and hasattr(m, "to"):
                            try:
                                m.to("cpu")
                            except Exception:
                                pass
                _te_holder["te"] = te
            print(f"[StagedJE] Stage 1/3: encoding {text[:40]!r}... (slow part on CPU)", flush=True)
            cond = te([text])
            out = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                   for k, v in cond.items()}
            del cond
            _COND_CACHE[key] = out
            while len(_COND_CACHE) > 6:  # bounded (positive + negative pairs)
                _COND_CACHE.pop(next(iter(_COND_CACHE)))
            return out

        cond_cpu = _encode_text(prompt)
        neg_cpu = _encode_text(negative_prompt) if use_negative else None

        if _te_holder["te"] is not None:
            _te_holder["te"] = None
            gc.collect()
            _empty_cache()
            print("[StagedJE] Stage 1/3: Gemma released from RAM.", flush=True)

        # NEGATIVE PROMPT (experimental): the DMD-distilled pipeline has no CFG
        # input, so true classifier-free guidance is impossible here. Instead we
        # extrapolate the conditioning away from the negative in embedding space:
        # cond' = cond + scale * (cond - neg). Masks / integer tensors are left
        # untouched. scale=0 disables this completely (identical to before).
        if use_negative and neg_cpu is not None:
            def _looks_binary(t):
                try:
                    return (not t.is_floating_point()) or bool(((t == 0) | (t == 1)).all())
                except Exception:
                    return True
            combined, n_steered = {}, 0
            for k, v in cond_cpu.items():
                nv = neg_cpu.get(k)
                if (isinstance(v, torch.Tensor) and isinstance(nv, torch.Tensor)
                        and v.shape == nv.shape and v.is_floating_point()
                        and not _looks_binary(v)):
                    combined[k] = v + float(negative_scale) * (v - nv)
                    n_steered += 1
                else:
                    combined[k] = v
            cond_cpu = combined
            print(f"[StagedJE] negative prompt applied (scale={negative_scale}, "
                  f"{n_steered} tensors steered).", flush=True)

        # ============================================================ STAGE 2
        # MODEL CACHE: keep the built generator + VAEs between runs (on CPU,
        # packed weights memory-mapped). Re-queues skip the 8-12 min rebuild.
        model_key = (dit_gguf, video_height, video_width,
                     video_vae_path, audio_vae_path, vocoder_path)
        cached = _MODEL_CACHE.get(model_key)
        if cached is not None:
            print("[StagedJE] Stage 2/3: cached generator + VAEs hit -- skipping load.", flush=True)
            generator, video_vae, audio_vae, sr = cached
        else:
            print("[StagedJE] Stage 2/3: loading DiT (GGUF) + VAEs...", flush=True)
            generator = RebelsJE_DiTLoader().run(cfg, dit_gguf, video_height, video_width)[0]
            video_vae, audio_vae, sr = RebelsJE_VAELoader().run(
                cfg, video_vae_path, audio_vae_path, vocoder_path, False  # with_encoders=False
            )
            _MODEL_CACHE.clear()  # only ever hold one model set on 16GB
            _MODEL_CACHE[model_key] = (generator, video_vae, audio_vae, sr)
        # The vocoder's NATIVE rate must label the waveform -- JD's own pipeline
        # does `audio_vae.get_output_sample_rate() or 24000`. Letting the widget
        # value override it mislabels the audio: a 44.1/48k waveform stamped
        # 24000 plays at ~half speed, an octave deep ("destroyed" audio).
        if sr and audio_sample_rate != sr:
            print(f"[StagedJE] audio_sample_rate {audio_sample_rate} != vocoder native "
                  f"{sr}; using {sr} (widget value ignored).", flush=True)
        audio_sample_rate = sr or audio_sample_rate

        # ============================================================ STAGE 3
        # Denoise + decode using the precomputed conditioning. Lifted from
        # nodes.py JoyEcho_SingleShotGenerate.generate_shot Phase A + Phase B,
        # with the in-line text-encode block removed.
        print("[StagedJE] Stage 3/3: denoise + decode...", flush=True)
        from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
        from ltx_distillation.inference.memory_multishot import normalize_audio_waveform_for_media
        from ltx_distillation.utils import add_noise, compute_latent_shapes, decode_benchmark_sample

        # validate num_frames -> 1 + multiple of 8
        if (num_frames - 1) % 8 != 0:
            num_frames = 1 + ((num_frames - 1) // 8) * 8

        # generator resolution wiring (matches generate_shot)
        generator.video_height = video_height
        generator.video_width = video_width
        generator.latent_height = video_height // 32
        generator.latent_width = video_width // 32
        generator.video_frame_seqlen = generator.latent_height * generator.latent_width

        video_shape, audio_shape = compute_latent_shapes(
            num_frames=num_frames,
            video_height=video_height,
            video_width=video_width,
            batch_size=1,
            video_fps=float(video_fps),
        )

        conditional_dict = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in cond_cpu.items()
        }
        del cond_cpu

        denoising_sigmas = torch.tensor(DENOISING_SIGMAS, device=device, dtype=torch.float32)
        base_pipeline = BidirectionalAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
        )

        # pin_memory=False is REQUIRED here: the default pins every block's
        # weights into page-locked host RAM, which tries to pin the whole packed
        # DiT (~GGUF size) at once and fails with a CUDA OOM on cudaHostAlloc --
        # even though VRAM is nearly empty. Pinning is only a transfer-speed
        # optimization; per-block GPU streaming (what makes this fit in 8GB) is
        # unaffected by turning it off.
        # RUN-2 HANG FIX: the generator is cached across runs, but a NEW
        # offloader was being created and installed each run -- stacking a
        # second set of hooks on every block, which deadlocks the second
        # generation. Keep exactly ONE offloader attached to the generator;
        # its install() is internally one-shot.
        offloader = None
        if sequential_offload:
            offloader = getattr(generator, "_rebels_offloader", None)
            if offloader is None or getattr(offloader, "_device", None) != device:
                offloader = SequentialOffloader(generator, device, pin_memory=False)
                generator._rebels_offloader = offloader

        # ---- Phase A: denoise (generator on GPU, VAEs off) ----
        _move(video_vae.decoder, "cpu")
        _move(audio_vae.decoder, "cpu")
        _move(audio_vae.vocoder, "cpu")
        if getattr(video_vae, "encoder", None) is not None:
            _move(video_vae.encoder, "cpu")
        if getattr(audio_vae, "encoder", None) is not None:
            _move(audio_vae.encoder, "cpu")
        if sequential_offload:
            # Phase B calls offloader.remove() at the end of every run, so the
            # hooks are gone on a cached re-run. Force a clean re-install (the
            # _installed latch would otherwise early-return and leave the
            # generator hookless and parked on CPU -> the run-2 hang).
            try:
                offloader.remove()
            except Exception:
                pass
            offloader._installed = False
            offloader.install()
            # install() re-stages devices, but belt-and-braces for cached runs:
            # but Phase B moves the whole generator to CPU after decoding. On a
            # cache-hit run we therefore re-stage manually: non-block params
            # back to GPU, blocks pinned to CPU for streaming.
            for _n, _p in generator.named_parameters():
                if "transformer_blocks" not in _n:
                    _p.data = _p.data.to(device)
            for _n, _b in generator.named_buffers():
                if "transformer_blocks" not in _n:
                    _b.data = _b.data.to(device)
            try:
                for _blk in generator.model.velocity_model.transformer_blocks:
                    _blk.to("cpu")
            except Exception:
                pass
        else:
            _move(generator, device)
        _empty_cache()

        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            video_latent, audio_latent = base_pipeline.generate(
                video_shape=tuple(video_shape),
                audio_shape=tuple(audio_shape),
                conditional_dict=conditional_dict,
                seed=seed,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        del conditional_dict
        _empty_cache()

        # ---- Phase B: decode (generator off, VAE decoders on GPU) ----
        if sequential_offload:
            offloader.remove()
        _move(generator, "cpu")
        _empty_cache()
        _move(video_vae.decoder, device)
        _move(audio_vae.decoder, device)
        _move(audio_vae.vocoder, device)

        # TILED DECODE: see _SpatialTiledDecoder. Swap the decoder for the
        # tiling proxy so JD's decode path becomes tiled transparently.
        _orig_decoder = None
        if tiled_decode and isinstance(getattr(video_vae, "decoder", None), torch.nn.Module):
            lat_h, lat_w = int(video_latent.shape[-2]), int(video_latent.shape[-1])
            if lat_h > decode_tile_size or lat_w > decode_tile_size:
                _orig_decoder = video_vae.decoder
                video_vae.decoder = _SpatialTiledDecoder(_orig_decoder,
                                                         tile=decode_tile_size, overlap=3)
                print(f"[StagedJE] tiled VAE decode: latent {lat_h}x{lat_w} in "
                      f"{decode_tile_size}-latent tiles (~{decode_tile_size*32}px), "
                      f"overlap 3.", flush=True)
        try:
            video_uint8, audio_waveform = decode_benchmark_sample(
                video_vae, audio_vae, video_latent, audio_latent
            )
        finally:
            if _orig_decoder is not None:
                video_vae.decoder = _orig_decoder

        if device.type == "cuda":
            torch.cuda.synchronize()
        _move(video_vae.decoder, "cpu")
        _move(audio_vae.decoder, "cpu")
        _move(audio_vae.vocoder, "cpu")
        _empty_cache()

        images = video_uint8.float() / 255.0  # [F, H, W, 3]

        audio_out = None
        if audio_waveform is not None:
            audio_norm = normalize_audio_waveform_for_media(audio_waveform)
            # The vocoder overshoots full scale (measured peaks 1.35-1.41).
            # Hard-clamping that much signal bakes in clipping distortion --
            # peak-normalize instead so the waveform shape is preserved.
            peak = audio_norm.abs().max()
            if peak > 1.0:
                audio_norm = audio_norm / peak
            audio_out = {"waveform": audio_norm.unsqueeze(0), "sample_rate": audio_sample_rate}

        del video_latent, audio_latent, video_uint8, audio_waveform
        gc.collect()
        _empty_cache()
        print(f"[StagedJE] done. {images.shape[0]} frames.", flush=True)
        return (images, audio_out)


NODE_CLASS_MAPPINGS = {"RebelsJE_StagedPipeline": RebelsJE_StagedPipeline}
NODE_DISPLAY_NAME_MAPPINGS = {"RebelsJE_StagedPipeline": "Rebels JE - Staged Pipeline (16GB)"}
