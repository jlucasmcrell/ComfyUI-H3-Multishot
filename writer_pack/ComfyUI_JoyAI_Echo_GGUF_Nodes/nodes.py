"""JoyAI-Echo ComfyUI node implementations.

Six nodes faithful to the official inference.py:
1. JoyEcho_ModelLoader          — load text encoder + DiT + VAEs (bf16)
2. JoyEcho_TextEncode           — encode prompts, auto-release text encoder
3. JoyEcho_Generate             — multi-shot denoise + decode with memory bank
4. JoyEcho_SingleShotGenerate   — single-shot with per-shot text box and memory chaining
5. JoyEcho_PromptFormat         — get system prompt for LLM-based prompt enhancement
6. JoyEcho_LLMEnhance           — call LLM API to generate shot prompts from a story idea
"""

from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import Any

import torch


DENOISING_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move(module, device):
    if module is not None:
        module.to(device)


class SequentialOffloader:
    """Layer-by-layer GPU offloading for the DiT transformer blocks.

    Hooks into each transformer block so that only the currently-executing block
    resides on GPU. All other blocks stay on CPU/pinned memory.
    Peak VRAM for the generator drops from ~30GB to ~2-3GB (1 block + activations).
    """

    def __init__(self, generator, device: torch.device, pin_memory: bool = True,
                 resident_blocks: int = 0):
        self._generator = generator
        self._device = device
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._pin_memory = pin_memory
        self._installed = False
        # First N transformer blocks stay permanently on GPU (no hooks, no
        # streaming). Each streamed block costs a PCIe round-trip per denoise
        # step; pinning K of 48 cuts that traffic by K/48 at K x per-block
        # VRAM (bf16 ~0.9GB, fp8-resident ~0.45GB per block).
        self._resident_blocks = max(0, int(resident_blocks))

    def install(self):
        """Install forward hooks on transformer blocks and move them to CPU."""
        if self._installed:
            return
        self._installed = True

        velocity_model = self._generator.model.velocity_model
        blocks = velocity_model.transformer_blocks

        # Keep pre/post processing layers on GPU (small footprint)
        for name, param in velocity_model.named_parameters():
            if "transformer_blocks" not in name:
                param.data = param.data.to(self._device)
        for name, buf in velocity_model.named_buffers():
            if "transformer_blocks" not in name:
                buf.data = buf.data.to(self._device)

        n_res = min(self._resident_blocks, len(blocks))
        resident = list(blocks)[:n_res]
        streamed = list(blocks)[n_res:]

        # Resident blocks live on the GPU permanently.
        for block in resident:
            block.to(self._device)

        # Move streamed blocks to CPU (optionally pinned). Pinning is a
        # transfer-speed optimization (enables async H2D copies), NOT a
        # correctness requirement — so it must never be fatal. cudaHostAlloc
        # exhaustion surfaces as "CUDA error: out of memory" even though it
        # is HOST page-locked memory that ran out (hit on a 24 GB box 2026-07-19:
        # the refine's resident_blocks=0 tried to pin all 48 blocks after
        # the shot passes had pinned only 36 — the last ~11GB of pinning
        # pushed past what the host could lock). On the first failure we
        # stop pinning entirely (the pool is exhausted; per-param retries
        # just burn time) and stream the rest unpinned — the non_blocking
        # copies silently become synchronous, slower but correct.
        # Already-pinned tensors are a no-op for pin_memory(), so re-installs
        # keep whatever pinning already succeeded.
        _pin = self._pin_memory and torch.cuda.is_available()
        for block in streamed:
            block.to("cpu")
            if _pin:
                try:
                    for param in block.parameters():
                        param.data = param.data.pin_memory()
                    for buf in block.buffers():
                        buf.data = buf.data.pin_memory()
                except Exception as e:
                    _pin = False
                    print(f"[JoyEcho] WARNING: pinned-memory allocation failed "
                          f"({e}); streaming remaining blocks UNPINNED (slower "
                          f"PCIe transfers, otherwise identical).", flush=True)

        # Also keep the wrapper's patchifiers and X0Model's non-block params on GPU
        for name, param in self._generator.named_parameters():
            if "velocity_model.transformer_blocks" not in name and "velocity_model" not in name:
                param.data = param.data.to(self._device)
        for name, buf in self._generator.named_buffers():
            if "velocity_model.transformer_blocks" not in name and "velocity_model" not in name:
                buf.data = buf.data.to(self._device)

        def make_pre_hook(block_module):
            def hook(module, args):
                block_module.to(self._device, non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()
            return hook

        def make_post_hook(block_module):
            def hook(module, args, output):
                block_module.to("cpu", non_blocking=True)
            return hook

        for block in streamed:
            h1 = block.register_forward_pre_hook(make_pre_hook(block))
            h2 = block.register_forward_hook(make_post_hook(block))
            self._hooks.extend([h1, h2])

        if n_res:
            print(f"[JoyEcho] Sequential offloading installed: {len(blocks)} blocks "
                  f"({n_res} resident on GPU, {len(streamed)} streamed)", flush=True)
        else:
            print(f"[JoyEcho] Sequential offloading installed: {len(blocks)} blocks", flush=True)

    def remove(self):
        """Remove all hooks and move entire generator back to CPU."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._installed = False
        self._generator.to("cpu")


_MODEL_FILE_MANUAL = "(use checkpoint_path)"
_MODEL_FILE_CATS = ("checkpoints", "diffusion_models", "unet")
_LORA_FILE_MANUAL = "(use lora_path / none)"
_LORA_FILE_CATS = ("loras",)
_GEMMA_FILE_MANUAL = "(use gemma_path field)"
_GEMMA_FILE_CATS = ("text_encoders", "clip")


def _list_cat_files(cats, sentinel, exts=("*.safetensors", "*.gguf")) -> list:
    """Every matching file under the given ComfyUI model-dir categories, as
    'category: relative/path' combo entries. Dirs shared between categories
    (unet is an alias of diffusion_models on newer ComfyUI) are deduped."""
    try:
        import folder_paths
    except ImportError:
        return [sentinel]
    out, seen_dirs, seen = [], set(), set()
    for cat in cats:
        try:
            roots = folder_paths.get_folder_paths(cat)
        except Exception:
            continue
        for root in roots:
            try:
                rp = Path(root).resolve()
            except OSError:
                continue
            if not rp.is_dir() or rp in seen_dirs:
                continue
            seen_dirs.add(rp)
            for ext in exts:
                for f in rp.rglob(ext):
                    label = f"{cat}: {f.relative_to(rp).as_posix()}"
                    if label not in seen:
                        seen.add(label)
                        out.append(label)
    return [sentinel] + sorted(out)


def _resolve_cat_file(choice: str, cats, widget: str, sentinel: str) -> str:
    import folder_paths
    cat, _, rel = choice.partition(": ")
    if cat in cats and rel:
        for root in folder_paths.get_folder_paths(cat):
            p = Path(root) / rel
            if p.is_file():
                return str(p)
    raise FileNotFoundError(
        f"{widget} {choice!r} no longer exists on disk. Refresh the node "
        f"list (R) and re-pick, or use {sentinel}.")


def _list_model_files() -> list:
    return _list_cat_files(_MODEL_FILE_CATS, _MODEL_FILE_MANUAL)


def _resolve_model_file(choice: str) -> str:
    return _resolve_cat_file(choice, _MODEL_FILE_CATS, "model_file", _MODEL_FILE_MANUAL)


def _list_lora_files() -> list:
    return _list_cat_files(_LORA_FILE_CATS, _LORA_FILE_MANUAL, exts=("*.safetensors",))


def _resolve_lora_file(choice: str) -> str:
    return _resolve_cat_file(choice, _LORA_FILE_CATS, "lora_file", _LORA_FILE_MANUAL)


def _parse_lora_entries(raw: str, default_strength: float) -> list:
    """Multi-LoRA (v1.4): lora_path accepts SEVERAL entries separated by commas
    or newlines. Each entry is "path", "path@strength" or "path:strength" - the
    suffix counts as a strength only when it parses as a float, so Windows
    drive letters (F:\\x.safetensors) can never be mistaken for one. Entries
    without a suffix use the lora_strength widget. The fusion engine
    (apply_loras) has always summed a LIST of LoRA deltas; only this plumbing
    was single."""
    import re as _re
    out = []
    for part in _re.split(r"[,\n]+", raw or ""):
        part = part.strip().strip('"').strip("'")
        if not part:
            continue
        strength = float(default_strength)
        for sep in ("@", ":"):
            head, s, tail = part.rpartition(sep)
            if s and head:
                try:
                    strength = float(tail)
                    part = head.strip()
                    break
                except ValueError:
                    continue
        out.append((part, strength))
    return out


def _list_gemma_files() -> list:
    # Gemma can be a single-file .safetensors OR a .gguf, and lives in either
    # models/text_encoders or models/clip depending on how the user filed it.
    return _list_cat_files(_GEMMA_FILE_CATS, _GEMMA_FILE_MANUAL)


def _resolve_gemma_file(choice: str) -> str:
    return _resolve_cat_file(choice, _GEMMA_FILE_CATS, "gemma_file", _GEMMA_FILE_MANUAL)


class JoyEcho_LoraStacker:
    """Chainable LoRA stack - the familiar dropdown+strength UI. Each node adds
    up to three LoRAs; wire lora_stack to another stacker to chain more, and
    the final one into the Model Loader's lora_stack input. Every entry is
    applied on BOTH DiT paths: fused into the weights at load on safetensors,
    applied at runtime via forward hooks on a GGUF DiT (2.2)."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["(none)"] + [x for x in _list_lora_files()
                                  if x != _LORA_FILE_MANUAL]
        opt = {"lora_stack": ("JOYECHO_LORA_STACK", {
            "tooltip": "Chain from another LoRA Stack node to add more slots."})}
        for i in (1, 2, 3):
            opt[f"lora_{i}"] = (lora_list, {
                "default": "(none)",
                "tooltip": "LoRA from models/loras. (none) = slot unused."})
            opt[f"strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})
        return {"required": {}, "optional": opt}

    RETURN_TYPES = ("JOYECHO_LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"
    CATEGORY = "JoyAI-Echo"

    def stack(self, lora_stack=None, **kw):
        out = list(lora_stack) if lora_stack else []
        for i in (1, 2, 3):
            choice = kw.get(f"lora_{i}") or "(none)"
            if choice != "(none)" and ": " in str(choice):
                out.append((_resolve_lora_file(choice),
                            float(kw.get(f"strength_{i}", 1.0))))
        return (out,)


class JoyEcho_ModelLoader:
    """Load JoyAI-Echo model components: text encoder, DiT generator, and VAEs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            # All inputs are optional: pick from the dropdowns for the common
            # case, or fall back to the manual *_path fields for a GGUF's VAE
            # source, an HF gemma DIRECTORY, or a file outside the model tree.
            "required": {},
            "optional": {
                # --- DiT: pick a full/GGUF model, or type a full checkpoint ---
                "model_file": (_list_model_files(), {
                    "default": _MODEL_FILE_MANUAL,
                    "tooltip": "Pick the model instead of typing checkpoint_path. "
                               "A .safetensors = FULL checkpoint (replaces checkpoint_path "
                               "entirely: DiT + VAEs + vocoder + text connectors from that "
                               "file). A .gguf = DiT ONLY - checkpoint_path must still point "
                               "at a full safetensors (e.g. the JoyAI release) to supply the "
                               "VAEs/vocoder/connectors. Refresh the node list (R) after "
                               "adding files.",
                }),
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Manual fallback / GGUF VAE source. A full safetensors "
                               "checkpoint supplying the VAEs, vocoder and text connectors. "
                               "REQUIRED when model_file is a .gguf (DiT only); leave empty "
                               "when model_file is a full .safetensors.",
                }),
                # --- text encoder: pick a single file, or type a path/dir ---
                "gemma_file": (_list_gemma_files(), {
                    "default": _GEMMA_FILE_MANUAL,
                    "tooltip": "Pick the Gemma text encoder from models/text_encoders or "
                               "models/clip instead of typing gemma_path. Single-file "
                               ".safetensors or .gguf only - for an HF gemma-3-12b-it "
                               "DIRECTORY, leave this on the sentinel and type the folder in "
                               "gemma_path. Refresh the node list (R) after adding files.",
                }),
                "gemma_path": ("STRING", {
                    "default": "",
                    "tooltip": "Manual fallback for the text encoder. Use for an HF "
                               "gemma-3-12b-it DIRECTORY (dropdowns list files, not folders), "
                               "or an encoder outside models/text_encoders and models/clip. "
                               "Leave empty when gemma_file is set.",
                }),
                # --- LoRA: pick from the loras tree, or type a path ---
                "lora_file": (_list_lora_files(), {
                    "default": _LORA_FILE_MANUAL,
                    "tooltip": "Pick a LoRA from the models/loras tree instead of typing "
                               "lora_path. Applied at lora_strength on BOTH DiT paths: "
                               "fused into the weights at load on safetensors, applied at "
                               "runtime via forward hooks on a GGUF DiT (2.2). Refresh the "
                               "node list (R) after adding files.",
                }),
                "lora_stack": ("JOYECHO_LORA_STACK", {
                    "tooltip": "Wire a JoyEcho LoRA Stack node here for the "
                               "dropdown+strength multi-LoRA UI. Stacks with "
                               "lora_file and lora_path entries.",
                }),
                "lora_path": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "One or MORE LoRAs, separated by commas or newlines. Each "
                               "entry is a path, optionally with its own strength: "
                               "'a.safetensors@0.7, b.safetensors@0.5'. Entries without a "
                               "strength use lora_strength. Stacks WITH the lora_file pick. "
                               "Works on both DiT paths (fused at load on safetensors, "
                               "runtime hooks on GGUF).",
                }),
                "lora_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                # --- quantization toggles (DiT, then encoder) ---
                "fp8_transformer": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Quantize the DiT's attention/FF linear weights to "
                               "float8_e4m3fn at load (upcast per-layer during inference). "
                               "Roughly halves transformer weight memory - works from the "
                               "normal bf16 checkpoint, keeping JoyAI's memory training and "
                               "projection tensors intact. Slight quality cost; VAEs, text "
                               "encoder and non-linear layers stay bf16. Ignored when a "
                               "GGUF is picked in model_file (already quantized).",
                }),
                "fp8_scaled_mm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store DiT linears as float8_e4m3fn AND compute the matmuls "
                               "natively in fp8 via torch._scaled_mm (RTX 40/50-series). "
                               "Unlike fp8_transformer there is NO per-layer upcast tax - "
                               "and at ~22GB resident the DiT can run with "
                               "sequential_offload OFF at moderate resolutions. Overrides "
                               "fp8_transformer. Ignored for GGUF DiTs.",
                }),
                "encoder_fp8": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store the Gemma text encoder's linear weights as "
                               "float8_e4m3fn (upcast per-layer at encode). Roughly halves "
                               "the encoder's ~24GB footprint - it fits the GPU for the "
                               "encode pass (seconds per shot instead of ~12s on CPU) and "
                               "frees ~11GB system RAM. Encode runs once per queue item, so "
                               "the upcast tax is irrelevant here. Slight embedding shift - "
                               "voice is the canary; A/B before adopting.",
                }),
                # --- misc ---
                "low_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load text encoder on CPU for 24GB GPUs. "
                               "Encoding will be slower but uses no GPU memory.",
                }),
            },
        }

    RETURN_TYPES = ("JOYECHO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "JoyAI-Echo"

    def load_model(self, checkpoint_path: str = "", gemma_path: str = "",
                   lora_path: str = "", lora_strength: float = 1.0,
                   lora_stack=None,
                   low_vram: bool = False, fp8_transformer: bool = False,
                   model_file: str = _MODEL_FILE_MANUAL,
                   lora_file: str = _LORA_FILE_MANUAL,
                   fp8_scaled_mm: bool = False,
                   encoder_fp8: bool = False,
                   gemma_file: str = _GEMMA_FILE_MANUAL):
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_core.quantization import QuantizationPolicy
        from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
        from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
        from ltx_distillation.models.vae_wrapper import create_vae_wrappers

        dropdown_lora = None
        if lora_file and lora_file != _LORA_FILE_MANUAL:
            dropdown_lora = _resolve_lora_file(lora_file)

        gguf_dit_path = None
        if model_file and model_file != _MODEL_FILE_MANUAL:
            _resolved = _resolve_model_file(model_file)
            if _resolved.lower().endswith(".gguf"):
                gguf_dit_path = _resolved
                print(f"[JoyEcho] model_file: DiT from GGUF {_resolved}; VAEs/vocoder/"
                      f"connectors from checkpoint_path.", flush=True)
            else:
                checkpoint_path = _resolved
                print(f"[JoyEcho] model_file: full checkpoint {_resolved}.", flush=True)

        # A real pick is always "category: relative/path"; the sentinel (any
        # "(use ...)" placeholder) has no ": ", so this guard is robust to the
        # sentinel wording and to stale saved values from an older node version.
        if gemma_file and ": " in gemma_file:
            gemma_path = _resolve_gemma_file(gemma_file)
            print(f"[JoyEcho] gemma_file: {gemma_path}", flush=True)
        if not str(gemma_path).strip():
            raise ValueError(
                "No text encoder selected. Pick a Gemma in gemma_file, or type its "
                "path/directory in gemma_path.")

        if not str(checkpoint_path).strip():
            raise ValueError(
                "checkpoint_path is empty. It must point at a FULL safetensors checkpoint"
                + (" - with a GGUF picked in model_file it still supplies the VAEs, "
                   "vocoder and text connectors (e.g. echo-longvideo-release.safetensors)."
                   if gguf_dit_path else
                   " (or pick a .safetensors in model_file)."))

        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        gemma_path = str(Path(gemma_path).expanduser().resolve())

        # ComfyUI-quantized checkpoints ("fp8mixed learned" builds, marked by
        # .comfy_quant tensors) are packaged for the standard ComfyUI loader.
        # This ledger path never applies their weight_scale at runtime (the
        # scaled-mm consumer needs tensorrt_llm) and LoRA fusion assumes the
        # LTX transposed fp8 convention - the model would load MIS-SCALED and
        # LoRA fusion crashes with shape errors. Refuse early and clearly.
        if checkpoint_path.lower().endswith(".safetensors"):
            try:
                import json as _json
                import struct as _struct
                with open(checkpoint_path, "rb") as _f:
                    _n = _struct.unpack("<Q", _f.read(8))[0]
                    _hdr = _json.loads(_f.read(_n))
                _has_comfy_quant = any(k.endswith(".comfy_quant") for k in _hdr)
                _src_is_fp8 = any(isinstance(v, dict) and v.get("dtype") == "F8_E4M3"
                                  for k, v in _hdr.items() if k.startswith("model."))
            except Exception:
                _has_comfy_quant = False  # unreadable header: let the loader error surface
                _src_is_fp8 = False
            # fp8-compute toggles skip the loader's global bf16 cast, so an
            # already-fp8 FILE would load the ENTIRE transformer as fp8 -
            # norms, tables and adalns included. Those break immediately
            # (torch.randn: "normal_kernel_cuda not implemented for
            # Float8_e4m3fn") or silently misbehave. The toggles quantize the
            # right subset themselves FROM bf16 - so require the bf16 file.
            if _src_is_fp8 and (fp8_scaled_mm or fp8_transformer):
                raise ValueError(
                    f"{Path(checkpoint_path).name} is an fp8 checkpoint, but "
                    f"{'fp8_scaled_mm' if fp8_scaled_mm else 'fp8_transformer'} "
                    "needs the bf16 checkpoint as its source (it downcasts just "
                    "the attention/FF linears itself; an fp8 FILE loads every "
                    "tensor as fp8 with the cast skipped, which crashes the "
                    "denoise pipeline). Pick the matching bf16 file in "
                    "model_file - or turn the fp8 toggle off to run this fp8 "
                    "file the normal way (it upcasts to bf16 at load).")
            _cq_int8 = False
            if _has_comfy_quant:
                # INT8 ConvRot builds (int8_tensorwise markers) are SUPPORTED:
                # sft_loader reconstructs bf16 weights at load (dequant +
                # Hadamard un-rotation), so the file behaves exactly like the
                # bf16 checkpoint from here on - LoRA fusion, fp8 toggles and
                # the memory bank all work. Other comfy_quant formats
                # ("fp8mixed learned" etc.) are still refused: their scales
                # use a different contract this loader does not implement.
                try:
                    from safetensors import safe_open as _so
                    with _so(checkpoint_path, framework="pt") as _f:
                        _mk = next(k for k in _f.keys()
                                   if k.endswith(".comfy_quant"))
                        _fmt = _json.loads(bytes(
                            _f.get_tensor(_mk).numpy().tobytes()).decode("utf-8")
                            ).get("format")
                except Exception:
                    _fmt = None
                if _fmt == "int8_tensorwise":
                    _cq_int8 = True
                    print("[JoyEcho] INT8 ConvRot checkpoint detected: weights "
                          "will be reconstructed to bf16 at load (adds a minute "
                          "or two). NOTE: the INT8 file saves DOWNLOAD size, not "
                          "memory - loading needs the same system RAM as the "
                          "bf16 build.", flush=True)
                else:
                    _n = Path(checkpoint_path).name
                    raise ValueError(
                        f"{_n} is a ComfyUI-quantized checkpoint with "
                        f"comfy_quant format '{_fmt}', which this loader does "
                        "not support (INT8 ConvRot / int8_tensorwise builds ARE "
                        "supported and load automatically).\n"
                        "\nWhat to do instead:\n"
                        "  * Use a bf16 .safetensors checkpoint here, or a GGUF "
                        "in model_file (a GGUF is DiT-only, so checkpoint_path "
                        "still needs a full bf16 checkpoint for the VAEs and "
                        "vocoder).\n"
                        "  * Or load this file with ComfyUI's native loader "
                        "nodes - you lose the multishot memory bank that way.")

        # gemma_path: either the HF gemma-3-12b-it DIRECTORY (model*.safetensors +
        # tokenizer.model) or a SINGLE .safetensors/.gguf gemma file (e.g. an
        # fp8mixed export). A single file routes through the Rebels TextEncoder
        # machinery, which builds a .gemma_virtual_folder with the HF sidecars
        # and applies the scale-aware fp8/GGUF weight swap + session cache.
        # (Rebels local patch: gemma_single_file routing)
        _gp = Path(gemma_path)
        gemma_single_file = _gp.is_file() and _gp.suffix.lower() in (".safetensors", ".gguf")
        if _gp.is_file() and not gemma_single_file:
            raise ValueError(
                f"gemma_path points at a file ({_gp.name}) that is neither "
                f".safetensors nor .gguf. Point it at a gemma-3-12b-it folder "
                f"or a single-file gemma export.")
        if not gemma_single_file and not (_gp / "tokenizer.model").is_file():
            raise ValueError(
                f"gemma_path {_gp} is not a valid Gemma root: no tokenizer.model "
                f"inside. Point it at a full gemma-3-12b-it folder, or at a "
                f"single .safetensors/.gguf gemma file.")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.bfloat16

        # Load generator FIRST: the DiT quantize/LoRA-fuse pass is the peak
        # system-RAM moment of the whole load (bf16 checkpoint paged in + fp8
        # copies + fuse transients). Loading the 24GB text encoder before it
        # (the old order) stacked that on top of the peak and produced an
        # access-violation during the subsequent VAE mmap read on a 96GB box.
        # The encoder now loads LAST, after transients are collected.
        # --- host-RAM preflight -------------------------------------------
        # Three users reported Windows hard-crashing (RAM to 99%, then reboot)
        # during this load. Root cause is not a bug but an undisclosed
        # requirement: sft_loader forces an OWNED copy of every tensor off the
        # mmap (copy=True, to dodge a Windows access-violation), so the whole
        # checkpoint materializes in host RAM - then the quantize/LoRA-fuse
        # pass stacks fp8 copies and transients on top. That peak broke a 96GB
        # box once (see the ordering comment below). It fits on a machine with
        # a big pagefile and kills a 64GB machine with the Windows default.
        # Warn LOUDLY before the allocation rather than let the OS die: a
        # warning the user can act on beats an unattended reboot. Deliberately
        # advisory, not fatal - the estimate is coarse and a marginal setup
        # that would have completed should still be allowed to try.
        try:
            import psutil as _ps
            _ckpt_gb = os.path.getsize(checkpoint_path) / 2**30
            _vm, _sw = _ps.virtual_memory(), _ps.swap_memory()
            _avail = (_vm.available + _sw.free) / 2**30
            # Owned copies of the checkpoint + roughly half again for the
            # quantize/fuse transients. The GGUF DiT keeps its own weights
            # packed and memory-mapped, but checkpoint_path is STILL a full
            # checkpoint (it supplies the VAEs/vocoder) and how much of it is
            # copied has not been measured - a GGUF user reported the same
            # crash, so do NOT assume that path is cheap. Estimate coarsely and
            # err toward warning: a false warning costs a line of text, a
            # missed one costs an unattended reboot.
            # An fp8 FILE loaded with the fp8 toggles off is UPCAST TO BF16 at
            # load, so it costs ~2x its on-disk size in host RAM - picking fp8
            # saves disk, not RAM. Without this the estimate silently missed
            # the exact configuration that was reported crashing (25GB fp8
            # file on a 64GB box).
            _mult = 1.4 if gguf_dit_path is not None else 1.6
            if _src_is_fp8 and not (fp8_scaled_mm or fp8_transformer):
                _mult *= 2.0
            # INT8 ConvRot: weights are reconstructed to bf16 at load, so the
            # peak matches the bf16 build (~1.7x its size) even though the
            # file on disk is ~60% of it.
            if '_cq_int8' in dir() and _cq_int8:
                _mult = 2.7
            _need = _ckpt_gb * _mult
            if _avail < _need:
                print(
                    "\n" + "=" * 72 +
                    "\n[JoyEcho] WARNING - LOW SYSTEM MEMORY FOR THIS LOAD"
                    f"\n  checkpoint       : {_ckpt_gb:.1f} GiB"
                    f"\n  estimated peak   : ~{_need:.0f} GiB of host RAM + pagefile"
                    f"\n  currently free   : ~{_avail:.0f} GiB "
                    f"(RAM {_vm.available/2**30:.0f} + pagefile {_sw.free/2**30:.0f})"
                    "\n"
                    "\n  This load may exhaust system memory. On Windows that can"
                    "\n  hang or reboot the machine rather than raise an error."
                    "\n"
                    "\n  Fixes, cheapest first:"
                    "\n   1. RAISE YOUR PAGEFILE. This is the usual fix and costs"
                    "\n      only disk. Windows: System > About > Advanced system"
                    "\n      settings > Performance Settings > Advanced > Virtual"
                    "\n      memory > Change. Set a custom size of 64-128 GB on an"
                    "\n      SSD. The peak is brief; it spills instead of dying."
                    "\n   2. Use a GGUF in model_file instead of a bf16/fp8"
                    "\n      checkpoint - GGUF weights stay packed and memory-"
                    "\n      mapped rather than copied into RAM."
                    "\n   3. Close other applications, and load no LoRAs (fusion"
                    "\n      allocates additional transients at the peak)."
                    "\n" + "=" * 72 + "\n", flush=True)
        except Exception as _e_pf:
            print(f"[JoyEcho] RAM preflight skipped ({type(_e_pf).__name__}).",
                  flush=True)

        print("[JoyEcho] Loading DiT generator...", flush=True)
        _lora_entries = []
        if dropdown_lora:
            _lora_entries.append((dropdown_lora, float(lora_strength)))
        _lora_entries.extend(_parse_lora_entries(lora_path, lora_strength))
        if lora_stack:
            _lora_entries.extend((str(p), float(st)) for p, st in lora_stack)
        # dedupe identical paths (dropdown pick repeated in lora_path)
        _seen = set()
        _lora_entries = [e for e in _lora_entries
                         if not (str(Path(e[0]).expanduser()) in _seen
                                 or _seen.add(str(Path(e[0]).expanduser())))]
        loras = tuple(
            LoraPathStrengthAndSDOps(str(Path(p).expanduser()), float(st),
                                     LTXV_LORA_COMFY_RENAMING_MAP)
            for p, st in _lora_entries)
        if _lora_entries:
            print("[JoyEcho] LoRAs: " + ", ".join(
                f"{Path(p).name}@{st}" for p, st in _lora_entries), flush=True)

        if gguf_dit_path is not None:
            # DiT from GGUF via the Rebels loader machinery; the wrapper class
            # is identical to create_ltx2_wrapper's, so Generate can't tell.
            from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as _Builder
            from ltx_core.model.transformer import LTXModelConfigurator, X0Model
            from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper

            from .rebels_loaders import (
                _LOADER_CFG,
                _SWAP_MAP,
                _dit_module_ops,
                _full_config,
                _GGUFDiTLoader,
                _gguf_entries,
                _materialize_meta,
                _rebind_swapped,
            )

            if fp8_transformer:
                print("[JoyEcho] fp8_transformer ignored: GGUF DiT is already quantized.",
                      flush=True)
            # LoRAs used to be dropped here: fusion bakes W' = W + s*B*A into
            # the weight tensors, which needs real bf16/fp8 tensors, and a GGUF
            # keeps them packed and dequantizes per layer at compute time. They
            # are now applied at RUNTIME instead (see the attach below), so the
            # old "ignored on the GGUF DiT path" warning no longer applies.
            try:
                _cfg = _full_config(checkpoint_path)
            except Exception:
                _cfg = _full_config(_LOADER_CFG)

            _SWAP_MAP.clear()
            _entries = _gguf_entries(gguf_dit_path)
            _consumed = set()
            _builder = _Builder(
                model_class_configurator=LTXModelConfigurator,
                model_path=gguf_dit_path,
                model_sd_ops=None,
                module_ops=_dit_module_ops(_entries, _consumed, dtype),
                model_loader=_GGUFDiTLoader(_cfg, _entries, _consumed, dtype),
            )
            _transformer = _builder.build(device=torch.device("cpu"), dtype=dtype)
            generator = LTX2DiffusionWrapper(
                model=X0Model(_transformer), video_height=736, video_width=1280)
            generator.eval()
            _materialize_meta(generator, _entries, _consumed, dtype)
            _rebind_swapped(generator)
            _SWAP_MAP.clear()

            # Runtime LoRA for the GGUF path. Must come AFTER _materialize_meta
            # and _rebind_swapped so the hooks land on the final module objects
            # rather than ones that are about to be replaced. Never fatal: a bad
            # LoRA falls back to the old drop-behaviour with a loud warning.
            if _lora_entries:
                try:
                    try:
                        from .joyecho_gguf_runtime_lora import attach_runtime_loras
                    except ImportError:
                        from joyecho_gguf_runtime_lora import attach_runtime_loras
                    attach_runtime_loras(generator, _lora_entries, dtype)
                except Exception as _e_lora:
                    print(f"[JoyEcho GGUF-LoRA] WARNING: runtime attach failed "
                          f"({type(_e_lora).__name__}: {_e_lora}) - continuing "
                          f"WITHOUT LoRAs (old behaviour).", flush=True)
        else:
            quantization = None
            if fp8_scaled_mm:
                quantization = QuantizationPolicy.fp8_scaled_mm_torch()
                print("[JoyEcho] fp8_scaled_mm ON: DiT linears stored float8_e4m3fn and "
                      "COMPUTED in fp8 via torch._scaled_mm (no upcast tax; ~22GB resident).",
                      flush=True)
            elif fp8_transformer:
                quantization = QuantizationPolicy.fp8_cast()
                print("[JoyEcho] fp8_transformer ON: quantizing DiT linear weights to "
                      "float8_e4m3fn (upcast per-layer at inference).", flush=True)

            generator = create_ltx2_wrapper(
                checkpoint_path=checkpoint_path,
                # A single-file gemma has no model*.safetensors folder for the
                # ledger's eager text-encoder builder; that builder is never
                # used here (the encoder is built separately below), so skip it.
                gemma_path=None if gemma_single_file else gemma_path,
                device=torch.device("cpu"),
                dtype=dtype,
                video_height=736,
                video_width=1280,
                loras=loras,
                quantization=quantization,
            )
            generator.eval()

        # Free quantize/fuse transients before the next mmap-heavy stage.
        import gc
        gc.collect()

        # Load VAEs to CPU
        print("[JoyEcho] Loading VAEs...", flush=True)
        video_vae, audio_vae = create_vae_wrappers(
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),
            dtype=dtype,
            with_video_encoder=True,
            with_audio_encoder=True,
            decoder_device=torch.device("cpu"),
        )
        video_vae.eval()
        audio_vae.eval()
        gc.collect()

        # Text encoder: built LAZILY via this closure. On conditioning-cache-HIT
        # runs the encoder is never used, and eager loading cost 21-33GB of host
        # RAM plus up to a minute of load for nothing - on a 64GB box that alone
        # pushed the whole run into pagefile thrash. TextEncode resolves the
        # builder only after a cache MISS.
        text_encoder_device = torch.device("cpu") if low_vram else device

        def _build_text_encoder():
            if gemma_single_file:
                # Single-file gemma routes through the Rebels TextEncoder node:
                # .gemma_virtual_folder + sidecars, scale-aware fp8/GGUF weight
                # swap, session model cache. Returns the same
                # GemmaTextEncoderWrapper class as the folder path, so
                # TextEncode's GPU hot-swap and release work unchanged.
                # (Rebels local patch: gemma_single_file routing)
                from .rebels_loaders import RebelsJE_TextEncoder, _full_config, _LOADER_CFG
                if encoder_fp8:
                    print("[JoyEcho] encoder_fp8 ignored for a single-file gemma: "
                          "fp8 files already carry their own quantization; bf16 "
                          "files stay bf16.", flush=True)
                print(f"[JoyEcho] Loading text encoder (single file) on "
                      f"{text_encoder_device} via Rebels routing...", flush=True)
                try:
                    _cfg = _full_config(checkpoint_path)
                except Exception:
                    _cfg = _full_config(_LOADER_CFG)
                wrapper = RebelsJE_TextEncoder().run(
                    _cfg, gemma_path, "our_fp8", checkpoint_path, low_vram)[0]
                wrapper.eval()
                return wrapper
            print(f"[JoyEcho] Loading text encoder (bf16) on {text_encoder_device}...", flush=True)
            text_encoder = create_text_encoder_wrapper(
                checkpoint_path=checkpoint_path,
                gemma_path=gemma_path,
                device=text_encoder_device,
                dtype=dtype,
            )
            text_encoder.eval()

            if encoder_fp8:
                # The vision tower + projector are NEVER touched by text encoding
                # (the conditioning is a trained mix over the LANGUAGE model's
                # hidden states only - feature_extractor stacks hidden_states from
                # the text stack). Drop them entirely: ~1GB weights + buffers,
                # which is exactly the margin that decides whether a 24GB card can
                # host the encode pass on GPU.
                _stripped = 0
                for _mname, _mod in text_encoder.named_modules():
                    for _attr in ("vision_tower", "multi_modal_projector"):
                        _sub = getattr(_mod, _attr, None)
                        if isinstance(_sub, torch.nn.Module):
                            _stripped += sum(p.nbytes for p in _sub.parameters())
                            _stripped += sum(b.nbytes for b in _sub.buffers())
                            setattr(_mod, _attr, None)
                if _stripped:
                    gc.collect()
                    print(f"[JoyEcho] encoder_fp8: dropped the unused vision tower/projector "
                          f"({_stripped/1e9:.1f}GB).", flush=True)

                # Halve the Gemma footprint: store the remaining linear weights as
                # fp8, upcasting per layer at encode. Encode runs ONCE per queue
                # item, so the upcast tax that makes fp8_transformer slow on the
                # DiT is irrelevant here. JD's embeddings processor / connector
                # projections stay bf16.
                from ltx_core.quantization.fp8_cast import _replace_fwd_with_upcast
                _n = 0
                for _name, _m in text_encoder.named_modules():
                    if (isinstance(_m, torch.nn.Linear)
                            and ("language_model" in _name or "vision_tower" in _name)
                            and _m.weight.dtype in (torch.bfloat16, torch.float16)):
                        _m.weight.data = _m.weight.data.to(torch.float8_e4m3fn)
                        if _m.bias is not None:
                            _m.bias.data = _m.bias.data.to(torch.float8_e4m3fn)
                        _replace_fwd_with_upcast(_m)
                        _n += 1
                _gb = (sum(p.nbytes for p in text_encoder.parameters())
                       + sum(b.nbytes for b in text_encoder.buffers())) / 1e9
                print(f"[JoyEcho] encoder_fp8 ON: {_n} Gemma linears stored float8_e4m3fn; "
                      f"wrapper now {_gb:.1f}GB (upcast per-layer at encode).", flush=True)
            return text_encoder

        audio_sample_rate = audio_vae.get_output_sample_rate() or 24000

        model = {
            "text_encoder": None,  # resolved lazily from text_encoder_builder
            "text_encoder_builder": _build_text_encoder,
            "generator": generator,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "audio_sample_rate": audio_sample_rate,
            "device": device,
            "dtype": dtype,
            "checkpoint_path": checkpoint_path,
            "gemma_path": gemma_path,
            "encoder_fp8": bool(encoder_fp8),
        }

        print(f"[JoyEcho] Model loaded. Audio sample rate: {audio_sample_rate}", flush=True)
        return (model,)


# Default negative for the DMD (no-CFG) pipeline: steers each shot's conditioning
# away from these in embedding space. Covers BOTH failure modes seen on the
# multishot path: burned-in captions/subtitles (video context) and invented
# sung/musical audio from the Hat Man etc. (audio context). Kept as the FUNCTION
# default too, so it still fires when a stale graph node lacks the new widget.
_DEFAULT_JOYECHO_NEGATIVE = (
    "subtitles, captions, closed captions, on-screen text, text characters, glyphs, "
    "letters, words, writing, garbled text, lyrics, signatures, watermark, timestamp, "
    "logo, music, singing, song, humming, melody, chanting, vocalizing, score, "
    "soundtrack, musical, instrumental"
)

# Split per-domain defaults. The encoder emits SEPARATE video_context /
# audio_context tensors, so each domain gets its own negative text + scale:
#   - video: burned-in captions live here -> can be pushed hard
#   - audio: music lives here, but SPEECH does too ("subtitles" also correlates
#     with speech in training data) -> push gently, music tokens ONLY, no
#     voice-adjacent words (humming/chanting/vocalizing strangle whispers).
_DEFAULT_JOYECHO_NEGATIVE_VIDEO = (
    "subtitles, captions, closed captions, on-screen text, text characters, glyphs, "
    "letters, words, writing, garbled text, lyrics, signatures, watermark, timestamp, logo"
)
_DEFAULT_JOYECHO_NEGATIVE_AUDIO = (
    "music, singing, song, melody, score, soundtrack, musical, instrumental, "
    "background music"
)


class JoyEcho_TextEncode:
    """Encode text prompts using Gemma-3-12b.

    Supports:
    - One prompt per line (multi-line text, each line = one shot)
    - JSON format: {"prompts": ["shot1", "shot2", ...]} (official format)
    - JSON file path (*.json)

    After encoding, the text encoder is released from GPU to free ~24GB VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JOYECHO_MODEL",),
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One prompt per line, JSON object, or path to .json file",
                }),
            },
            "optional": {
                "negative_prompt_video": ("STRING", {
                    "multiline": True,
                    "default": _DEFAULT_JOYECHO_NEGATIVE_VIDEO,
                    "tooltip": "Steered away from in VIDEO context only (burned-in captions/subtitles/text). Safe to push hard - does not touch the audio lane. Empty or scale 0 disables.",
                }),
                "negative_scale_video": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Video-context steering strength. Renormalized, so higher values no longer degrade the image the way the old shared lever did.",
                }),
                "negative_prompt_audio": ("STRING", {
                    "multiline": True,
                    "default": _DEFAULT_JOYECHO_NEGATIVE_AUDIO,
                    "tooltip": "Steered away from in AUDIO context only. Music tokens ONLY - do NOT add caption words (captions correlate with speech; steering audio away from them kills dialogue). Empty or scale 0 disables.",
                }),
                "negative_scale_audio": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Audio-context steering strength. Keep LOW (~0.2-0.4) or dialogue suffers.",
                }),
                "release_text_encoder": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("JOYECHO_MODEL", "JOYECHO_COND",)
    RETURN_NAMES = ("model", "conditioning",)
    FUNCTION = "encode"
    CATEGORY = "JoyAI-Echo"

    @staticmethod
    def _parse_prompts(prompts: str) -> list[str]:
        """Parse prompts from text, JSON string, or JSON file path."""
        text = prompts.strip()

        # Check if it's a file path to a .json
        if text.endswith(".json") and not text.startswith("{"):
            p = Path(text).expanduser()
            if not p.is_absolute():
                p = Path(__file__).resolve().parent / p
            p = p.resolve()
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return JoyEcho_TextEncode._extract_from_json(data)

        # Check if it's a JSON object
        if text.startswith("{"):
            try:
                data = json.loads(text)
                return JoyEcho_TextEncode._extract_from_json(data)
            except json.JSONDecodeError:
                pass

        # Fall back to one-prompt-per-line
        return [line.strip() for line in text.split("\n") if line.strip()]

    @staticmethod
    def _extract_from_json(data: dict) -> list[str]:
        """Extract prompt list from JSON (supports 'prompts' or 'shots' key)."""
        if isinstance(data.get("prompts"), list):
            return [str(p).strip() for p in data["prompts"] if str(p).strip()]
        if isinstance(data.get("shots"), list):
            return [str(p).strip() for p in data["shots"] if str(p).strip()]
        raise ValueError("JSON must contain a 'prompts' or 'shots' array.")

    def encode(self, model: dict, prompts: str, negative_prompt: str = _DEFAULT_JOYECHO_NEGATIVE,
               negative_scale: float = 0.5, release_text_encoder: bool = True,
               negative_prompt_video: str = None, negative_scale_video: float = None,
               negative_prompt_audio: str = None, negative_scale_audio: float = None):
        text_encoder = model.get("text_encoder")
        if text_encoder is None and not callable(model.get("text_encoder_builder")):
            raise RuntimeError(
                "Text encoder not available. It may have been released already. "
                "Reload the model to encode new prompts."
            )

        prompt_list = self._parse_prompts(prompts)
        if not prompt_list:
            raise ValueError("No prompts provided. Enter text, JSON, or a .json file path.")

        # Speakers + voice anchors ride INSIDE the conditioning from here on.
        # They are properties of THIS text, so they belong on the data path, not
        # in module globals: the old stash pattern leaked a previous script's
        # speakers/anchors into any graph without a picker node (audit finding
        # 2026-07-29). Derived here, attached to conds[0] as "joyecho_meta",
        # cached WITH the conditioning (correct: same text = same speakers),
        # consumed by Generate, and stripped before the transformer sees it.
        _je_meta = None
        try:
            import json as _json_meta
            _md = _json_meta.loads(prompts) if str(prompts).lstrip().startswith("{") else None
            if isinstance(_md, dict):
                try:
                    from .joyecho_script_picker import derive_speakers as _derive_spk
                except ImportError:
                    from joyecho_script_picker import derive_speakers as _derive_spk
                _arr = _md.get("prompts") or _md.get("shots") or []
                _je_meta = {
                    "speakers": _derive_spk(_md, _arr),
                    "voice_refs": dict(_md.get("voice_refs") or {}),
                }
        except Exception:
            _je_meta = None

        device = model["device"]

        # --- Conditioning disk cache: the same script + negatives + encoder
        # always produces the same conditioning, so encode it ONCE per machine.
        # Re-renders (the A/B loop) skip the encoder entirely - on a box whose
        # encoder can only run on CPU (24GB cards), this turns a many-minutes
        # encode into a sub-second load.
        import hashlib as _hashlib
        import json as _json_cc
        _cc_key = _hashlib.sha256(_json_cc.dumps([
            prompt_list, str(negative_prompt), float(negative_scale or 0),
            str(negative_prompt_video), str(negative_scale_video),
            str(negative_prompt_audio), str(negative_scale_audio),
            str(model.get("gemma_path")), bool(model.get("encoder_fp8")),
            # checkpoint identity: the text-embedding connectors come from the
            # checkpoint and co-shape conditioning, so a model swap with the
            # same text must MISS, not serve the previous model's tensors
            # (audit finding 2026-07-29; invalidates all earlier cache files).
            str(model.get("checkpoint_path")),
        ]).encode()).hexdigest()[:16]
        _cc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cond_cache")
        _cc_path = os.path.join(_cc_dir, f"conds_{_cc_key}.pt")
        if os.path.isfile(_cc_path):
            try:
                cached_conds = torch.load(_cc_path, map_location="cpu", weights_only=True)
                print(f"[JoyEcho] Conditioning cache HIT ({os.path.basename(_cc_path)}) - "
                      f"{len(cached_conds)} shot(s), encode skipped entirely.", flush=True)
                if release_text_encoder and text_encoder is not None:
                    print("[JoyEcho] Releasing text encoder to free VRAM...", flush=True)
                    del text_encoder
                    model["text_encoder"] = None
                    gc.collect()
                    _empty_cache()
                return (model, cached_conds,)
            except Exception as _e:
                print(f"[JoyEcho] Conditioning cache unreadable ({_e}); re-encoding.", flush=True)

        print(f"[JoyEcho] Encoding {len(prompt_list)} prompt(s)...", flush=True)

        # Cache MISS: now (and only now) the encoder is actually needed.
        if text_encoder is None:
            text_encoder = model["text_encoder_builder"]()
            model["text_encoder"] = text_encoder

        # Hot-swap: a low_vram (CPU-resident) encoder costs ~10s+ per shot to run
        # the 12B gemma on CPU, while the GPU sits idle during this phase. Borrow
        # the GPU for the encode pass when the encoder fits in free VRAM, and hand
        # it back before the denoise phase - denoise-time VRAM is untouched.
        moved_to_gpu = False
        if getattr(device, "type", "") == "cuda" and torch.cuda.is_available():
            try:
                enc_dev = next(text_encoder.parameters()).device
            except (StopIteration, AttributeError):
                enc_dev = None
            if enc_dev is not None and enc_dev.type == "cpu":
                need = sum(p.nbytes for p in text_encoder.parameters())
                need += sum(b.nbytes for b in text_encoder.buffers())
                free = torch.cuda.mem_get_info(device)[0]
                # Absolute headroom, not a multiplier: encode activations for a
                # <=1k-token prompt are well under 1GB, and a 1.15x margin on a
                # ~21GB encoder demanded ~3GB of phantom headroom - exactly what
                # kept 24GB cards on CPU encode. The OOM fallback in _enc()
                # backstops a miss.
                if free > need + 1.2e9:
                    print(f"[JoyEcho] Text encoder -> GPU for the encode pass "
                          f"({need/1e9:.1f}GB weights, {free/1e9:.1f}GB free)...", flush=True)
                    try:
                        _move(text_encoder, device)
                        moved_to_gpu = True
                    except Exception as _e:
                        _move(text_encoder, torch.device("cpu"))
                        _empty_cache()
                        print(f"[JoyEcho] GPU encode swap failed ({_e}); encoding on CPU.", flush=True)
                else:
                    print(f"[JoyEcho] Encoder stays on CPU for encode "
                          f"({need/1e9:.1f}GB weights vs {free/1e9:.1f}GB free VRAM).", flush=True)
                    if any(p.dtype == torch.float8_e4m3fn for p in text_encoder.parameters()):
                        print("[JoyEcho] WARNING: encoder is fp8-stored but must encode on CPU - "
                              "the per-layer CPU upcast makes this MUCH slower than plain bf16. "
                              "On GPUs where the encoder cannot fit (24GB cards), set "
                              "encoder_fp8=False.", flush=True)

        def _enc(texts):
            nonlocal moved_to_gpu
            try:
                return text_encoder(texts)
            except RuntimeError as _e:
                # A hard allocator failure surfaces as a GENERIC RuntimeError
                # ("CUDA error: out of memory"), not torch.cuda.OutOfMemoryError
                # - on 2026-07-19 that escaped this handler and killed the whole
                # queue item instead of degrading to CPU encode. Treat any
                # out-of-memory RuntimeError as the fallback trigger.
                # (Rebels local patch: broad OOM fallback)
                _is_oom = (isinstance(_e, torch.cuda.OutOfMemoryError)
                           or "out of memory" in str(_e).lower())
                if not _is_oom or not moved_to_gpu:
                    raise
                print(f"[JoyEcho] GPU encode OOM ({type(_e).__name__}) - "
                      "falling back to CPU for the rest.", flush=True)
                _move(text_encoder, torch.device("cpu"))
                _empty_cache()
                moved_to_gpu = False
                return text_encoder(texts)

        # PER-DOMAIN NEGATIVE STEERING: the DMD-distilled pipeline has no CFG, so
        # we extrapolate conditioning away from a negative in embedding space:
        #   cond' = cond + scale * (cond - neg)      (then renormalized, below)
        # The encoder emits SEPARATE video_context / audio_context tensors, so
        # each domain gets its own negative text and scale:
        #   - video_context: burned-in captions live here -> push hard
        #   - audio_context: music lives here, but speech does too -> push gently
        # The old SHARED lever coupled the two: raising it past ~0.5 to kill
        # subtitles also strangled dialogue (captions correlate with speech in
        # training data) and drove the audio context off-manifold (the hum).
        # RENORM: raw extrapolation grows the context norm by ~(1+scale); the DiT
        # never saw conditioning at that magnitude -> audio hum / image drift.
        # Restoring the original per-token norm keeps only the DIRECTION change.
        if negative_prompt_video is None and negative_prompt_audio is None:
            # Stale graph still carrying the old single-lever widgets: preserve
            # the old behavior (same negative, same scale, both domains).
            negative_prompt_video = negative_prompt
            negative_prompt_audio = negative_prompt
            negative_scale_video = negative_scale
            negative_scale_audio = negative_scale

        domains = []  # (context_key, negative_text, scale)
        for key, txt, sc in (("video_context", negative_prompt_video, negative_scale_video),
                             ("audio_context", negative_prompt_audio, negative_scale_audio)):
            try:
                sc = float(sc)
            except (TypeError, ValueError):
                sc = 0.0
            txt = str(txt).strip() if txt is not None else ""
            if sc > 0.0 and txt:
                domains.append((key, txt, sc))

        neg_ctx = {}  # context_key -> (neg_tensor, scale)
        if domains:
            encoded = {}  # one encoder pass per DISTINCT negative text
            for key, txt, sc in domains:
                if txt not in encoded:
                    _nc = _enc([txt])
                    encoded[txt] = {k: (t.detach() if isinstance(t, torch.Tensor) else t)
                                    for k, t in _nc.items()}
                    del _nc
                nv = encoded[txt].get(key)
                if isinstance(nv, torch.Tensor) and nv.is_floating_point():
                    neg_ctx[key] = (nv, sc)
                    print(f"[JoyEcho] Negative for {key}: scale={sc}.", flush=True)

        def _steer(v, nv, scale):
            out = v + scale * (v - nv.to(v.device))
            # Norm-preserving rescale (same idea as RescaleCFG): keep the
            # direction change, restore the original per-token magnitude.
            norm_in = v.norm(dim=-1, keepdim=True)
            norm_out = out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return out * (norm_in / norm_out)

        cached_conds = []
        for i, prompt in enumerate(prompt_list):
            cond = _enc([prompt])
            if neg_ctx:
                cond = dict(cond)
                for key, (nv, sc) in neg_ctx.items():
                    v = cond.get(key)
                    if (isinstance(v, torch.Tensor) and v.is_floating_point()
                            and v.shape == nv.shape):
                        cond[key] = _steer(v, nv, sc)
                    elif i == 0:
                        print(f"[JoyEcho]   WARNING: negative SKIPPED for {key} "
                              f"(shape {getattr(v, 'shape', None)} vs {tuple(nv.shape)}).",
                              flush=True)
                if i == 0:
                    applied = ", ".join(f"{k}@{sc}" for k, (_, sc) in neg_ctx.items())
                    print(f"[JoyEcho]   negative applied per-domain: {applied}.", flush=True)
            cached_conds.append(
                {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                 for k, v in cond.items()}
            )
            del cond
            print(f"[JoyEcho]   Encoded shot {i+1}/{len(prompt_list)}", flush=True)

        if neg_ctx:
            neg_ctx.clear()

        if moved_to_gpu:
            _move(text_encoder, torch.device("cpu"))
            _empty_cache()
            print("[JoyEcho] Text encoder -> CPU (encode pass done).", flush=True)

        # Attach the script-derived speakers/anchors so they persist in the
        # cache and reach Generate on the data path (see comment at derivation).
        if _je_meta and cached_conds:
            cached_conds[0]["joyecho_meta"] = _je_meta

        # Persist the conditioning for instant re-runs of the same script.
        try:
            _cc_bytes = sum(v.nbytes for c in cached_conds for v in c.values()
                            if isinstance(v, torch.Tensor))
            if _cc_bytes < 4_000_000_000:  # sanity cap
                os.makedirs(_cc_dir, exist_ok=True)
                torch.save(cached_conds, _cc_path)
                print(f"[JoyEcho] Conditioning cached ({os.path.basename(_cc_path)}, "
                      f"{_cc_bytes/1e6:.0f}MB) - future runs of this script skip the encode.",
                      flush=True)
        except Exception as _e:
            print(f"[JoyEcho] Conditioning cache save failed ({_e}); continuing.", flush=True)

        if release_text_encoder:
            print("[JoyEcho] Releasing text encoder to free VRAM...", flush=True)
            del text_encoder
            model["text_encoder"] = None
            gc.collect()
            _empty_cache()

        return (model, cached_conds,)


class JoyEcho_Generate:
    """Generate multi-shot video + audio using DMD few-step denoising with memory bank.

    Implements the same hot-swap memory management as official inference.py:
    - Denoise phase: generator on GPU, VAE on CPU
    - Decode phase: generator on CPU, VAE on GPU
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JOYECHO_MODEL",),
                "conditioning": ("JOYECHO_COND",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 2**31 - 1}),
                "num_frames": ("INT", {"default": 241, "min": 9, "max": 1441, "step": 8,
                                       "tooltip": "Must be 1 + 8*k (e.g. 121, 241, 361)"}),
                # Rebels local patch: portrait resolutions. Height was capped at
                # 1088 while width allowed 1920, which silently forbade portrait
                # (e.g. 1088x1920 to match a portrait Z-Image first frame). Both
                # axes now cap at 1920; step 32 keeps the latent packing valid.
                "video_height": ("INT", {"default": 736, "min": 256, "max": 1920, "step": 32}),
                "video_width": ("INT", {"default": 1280, "min": 256, "max": 1920, "step": 32}),
            },
            "optional": {
                "video_fps": ("INT", {"default": 24, "min": 1, "max": 60, "tooltip": "KEEP AT 24. The LTX joint audio-video prior is 24fps-native: any other value (25 included) systematically drifts spoken voices toward Commonwealth accents (British/Australian) and overrides accent wording in the prompt. Verified A/B 2026-07-29."}),
                "v2a_grad_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "memory_max_size": ("INT", {"default": 7, "min": 0, "max": 20}),
                "num_fix_frames": ("INT", {"default": 3, "min": 0, "max": 10}),
                "enable_audio_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Feed previous shots' audio latents as cross-shot conditioning. "
                               "ON keeps the same voice across every shot - the point of "
                               "multishot - at a small lip-sync cost on long dialogue. OFF "
                               "gives the tightest sync but the voice can drift between "
                               "shots; use only for sync-critical single-voice pieces with "
                               "a strong voice description repeated in every shot.",
                }),
                "audio_memory_window_size": ("INT", {"default": 96, "min": 16, "max": 256}),
                "memory_video_anchor": (["last", "loudest"], {
                    "default": "last",
                    "tooltip": "Which part of a shot becomes the VIDEO half of its memory "
                               "slot. last = the final frames, so the next shot continues "
                               "from where this one ended. loudest = the original behaviour: "
                               "the frames around the shot's most articulated speech, which "
                               "is usually mid-shot and makes the next shot look like it "
                               "rewound a second or two. The AUDIO half is chosen by "
                               "max_response either way - that is the right voice exemplar.",
                }),
                "speaker_order": ("STRING", {
                    "default": "",
                    "tooltip": "Who SPEAKS in each shot, comma or space separated "
                               "(e.g. 'A,B'). Tags each memory slot with its speaker so a "
                               "shot only inherits audio memory from the character talking "
                               "in it - the other character's slots are silenced for that "
                               "shot while their face memory is kept. This is the fix for "
                               "two same-gender voices merging across shots. The list CYCLES, "
                               "so 'A,B' covers any number of alternating shots. Leave EMPTY "
                               "for the previous behaviour (every voice in every shot).",
                }),
                "sequential_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable layer-by-layer GPU offloading for DiT. "
                               "Reduces VRAM from ~30GB to ~3GB at the cost of slower inference.",
                }),
                "output_prefix": ("STRING", {
                    "default": "joyecho/shot",
                    "tooltip": "Prefix for per-shot video files saved immediately after each shot completes.",
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional identity reference (e.g. a Z-Image render). Pre-seeds the "
                               "cross-shot memory bank as a permanent anchor slot, so every shot is "
                               "conditioned on this face/look - reference-driven I2V. Uses one of the "
                               "num_fix_frames anchor slots.",
                }),
                "transition": (["cut", "dissolve", "vhs_glitch"], {
                    "default": "cut",
                    "tooltip": "Shot-boundary treatment. cut = hard cuts (original). dissolve = "
                               "overlap cross-dissolve + equal-power audio crossfade (shortens total "
                               "by transition_frames per boundary). vhs_glitch = analog static burst "
                               "at each cut: snow, tearing bands, dropout lines + a tape-noise audio "
                               "hit (length unchanged).",
                }),
                "transition_frames": ("INT", {
                    "default": 8, "min": 1, "max": 48,
                    "tooltip": "Length of the transition in frames. Dissolve: 8-12 is natural. "
                               "VHS glitch: 3-6 reads as a head-switch stutter, 8-12 as a violent burst.",
                }),
                "glitch_intensity": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "vhs_glitch only: how hard the burst hits (snow mix, tear count, "
                               "audio static level).",
                }),
                "head_trim_frames": ("INT", {
                    "default": 14, "min": 0, "max": 24,
                    "tooltip": "Trim this many frames (plus matching audio) from the START of every "
                               "shot. The model's first frames morph out of the reference/memory "
                               "content, and shot-start lip sync is at its worst there. 14 is the "
                               "production-proven value (THE WITNESS ran it). 0 = auto: trims 8 "
                               "when a reference_image is wired, none otherwise.",
                }),
                "decode_tiling": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": "Temporal-chunked VAE decode (64-frame chunks, 24-frame blended "
                               "overlap, no spatial tiles = no spatial seams). Caps decode peak "
                               "memory at ~one chunk instead of the whole shot - fixes the hard "
                               "crash decoding 241f at 1280x736. auto = only when "
                               "height*width*frames exceeds the known-safe budget; small renders "
                               "keep the original single-pass decode.",
                }),
                "head_trim_first_shot": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "Extra-long head trim for SHOT 1 ONLY. Shot 1 has an empty memory "
                               "bank, so its conditioning is 100% the reference clips - it opens "
                               "practically ON the reference and takes far longer than later "
                               "shots to morph out. 0 = auto: double head_trim_frames (min 24) "
                               "when a reference_image is wired. Raise to 32-40 if the first "
                               "frame still shows the reference.",
                }),
                "reference_zoom": ("FLOAT", {
                    "default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Over-zoom applied to reference images before conditioning. A "
                               "reference at EXACTLY the render size is pixel-continuable - the "
                               "model opens the shot ON it (a long literal ghost that no head "
                               "trim covers). Cropping in ~20% keeps identity and wide "
                               "composition but breaks the wholesale-continuation shortcut. "
                               "1.0 = old passthrough behavior.",
                }),
                "reference_shots": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Comma list from RefPicker's ref_shots output: 0-based shot "
                               "index where each reference injects (per-character entry "
                               "shots). Unwired/empty = all references at shot 1.",
                }),
                "resident_blocks": ("INT", {
                    "default": 0, "min": 0, "max": 48,
                    "tooltip": "Sequential offload only: keep the first N of 48 transformer "
                               "blocks permanently on the GPU and stream the rest. Each "
                               "streamed block is a PCIe round-trip per denoise step, so "
                               "N=24 halves the streaming time for N x per-block VRAM "
                               "(bf16 ~0.9GB/block, fp8 ~0.45GB). Raise until VRAM is "
                               "nearly full; 0 = stream everything (old behavior).",
                }),
                "hires_factor": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "ROUTING switch, not a quality slider - it decides who builds "
                               "your master. 1.0 (RECOMMENDED): shots stay base-res and the "
                               "AutoFinish worker upscales them deterministically "
                               "(bicubic+CAS) - zero per-frame detail shimmer. Above 1.0: the "
                               "selected hires_denoise pass runs in-render and the master is "
                               "built from ITS output instead (AutoFinish detects the hires "
                               "shots and skips its own upscale). Only raise this when you "
                               "deliberately want the refine modes' synthesized detail "
                               "(slight texture shimmer) or the spatial mode (even /32 dims "
                               "only, e.g. 768-height). Judge results from the *_MASTER.mp4 "
                               "file, never the in-canvas preview.",
                }),
                "hires_denoise": (["subtle (1 step)", "medium (2 steps)",
                                   "strong (tenstrip 4-step)",
                                   "spatial (LTX latent upsampler, no churn)",
                                   "anneal (latent x1.5 + 7-step ladder)",
                                   "anneal-x2 (latent x2-1.1 + ladder)",
                                   "latent (whole-shot x1.5 + ladder)",
                                   "latent-x2 (whole-shot x2-1.1 + ladder)",
                                   "latent-x2-turbo (x2 + 3-step distilled tail)"], {
                    "default": "subtle (1 step)",
                    "tooltip": "How the hires pass works. The three refine modes re-noise and "
                               "re-denoise at target res - they SYNTHESIZE real detail but "
                               "reshuffle fine texture slightly per frame (subtle = sigma 0.42 "
                               "one step, most faithful; medium = 0.725 two steps; strong = "
                               "tenstrip's 0.92/0.725/0.42 ladder, deepest). spatial = the LTX "
                               "latent upsampler on the shot's own latents: deterministic, "
                               "zero temporal churn, fixed 1.5x (hires_factor just enables "
                               "it) - use heights whose /32 is EVEN (768, not 736) or it "
                               "smears one edge.",
                }),
                "temporal_upscale": (["off", "2x (48 fps master)"], {
                    "default": "off",
                    "tooltip": "LTX temporal latent upsampler (x2) on each shot's OWN latents "
                               "after sampling: per-shot files and the master come out at "
                               "double fps (24 -> ~48) with the audio untouched - same "
                               "duration, twice the motion samples, no optical-flow ghosting. "
                               "The render itself stays at video_fps 24, so the accent/sync "
                               "law is unaffected; the in-canvas preview also stays at base "
                               "fps (only the saved files double). v1 limitation: requires "
                               "hires_factor 1.0 (skipped with a warning otherwise). Keep OFF "
                               "for found-footage looks - interpolated camcorder reads as "
                               "soap opera.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO",)
    RETURN_NAMES = ("images", "audio",)
    FUNCTION = "generate"
    CATEGORY = "JoyAI-Echo"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, hires_factor=None, reference_zoom=None, **kwargs):
        # Turn ComfyUI's cryptic pre-execution validation error into an
        # actionable one for the widget-scramble case. Loading a workflow saved
        # under an OLDER node layout shifts values into the wrong slots (values
        # serialize by position): the tell is hires_factor holding a
        # hires_denoise STRING ("could not convert string to float") and/or
        # reference_zoom holding a big resident_blocks number ("bigger than max
        # of 2.0"). Listing these two here makes ComfyUI defer their validation
        # to us so we can name the real fix instead of a type error. generate()
        # still clamps at runtime as a backstop.
        _scramble = ("JoyEcho Generate: this node's widget values are scrambled - "
                     "you loaded a workflow saved under an OLDER node layout, so "
                     "values shifted into the wrong slots. FIX: right-click this "
                     "node and remove it, add a fresh 'JoyEcho Generate' node, "
                     "reconnect its inputs and re-enter your settings (or load the "
                     "workflow from the current release zip). ")
        if hires_factor is not None:
            try:
                float(hires_factor)
            except (TypeError, ValueError):
                return _scramble + f"(hires_factor got {hires_factor!r})"
        if reference_zoom is not None:
            try:
                if not (1.0 <= float(reference_zoom) <= 2.0):
                    return _scramble + f"(reference_zoom got {reference_zoom})"
            except (TypeError, ValueError):
                return _scramble + f"(reference_zoom got {reference_zoom!r})"
        return True

    def generate(
        self,
        model: dict,
        conditioning: list,
        seed: int = 12345,
        num_frames: int = 241,
        video_height: int = 736,
        video_width: int = 1280,
        video_fps: int = 25,
        v2a_grad_scale: float = 2.0,
        memory_max_size: int = 7,
        num_fix_frames: int = 3,
        enable_audio_memory: bool = True,
        audio_memory_window_size: int = 96,
        memory_video_anchor: str = "last",
        speaker_order: str = "",
        sequential_offload: bool = False,
        output_prefix: str = "joyecho/shot",
        reference_image=None,
        transition: str = "cut",
        transition_frames: int = 8,
        glitch_intensity: float = 0.7,
        head_trim_frames: int = 0,
        decode_tiling: str = "auto",
        head_trim_first_shot: int = 0,
        reference_zoom: float = 1.2,
        reference_shots: str = "",
        resident_blocks: int = 0,
        hires_factor: float = 1.0,
        hires_denoise: str = "subtle (1 step)",
        temporal_upscale: str = "off",
    ):
        from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
        from ltx_distillation.inference.memory_bidirectional_pipeline import BidirectionalMemoryAVInferencePipeline
        from ltx_distillation.inference.memory_multishot import (
            PairedAudioVideoMemoryBank,
            build_paired_audio_memory_kwargs,
            video_uint8_to_pil_frames,
        )
        from ltx_distillation.utils import (
            add_noise,
            compute_latent_shapes,
            decode_benchmark_sample,
            encode_memory_frames_batch,
        )

        # WIDGET-SHIFT SELF-HEAL. ComfyUI serializes widget values by POSITION,
        # so a graph saved under an older node layout maps its tail values into
        # the wrong slots after an upgrade (classic signature: reference_zoom
        # gets an old resident_blocks value like 24, hires_factor gets a STRING
        # from hires_denoise and arrives as NaN). Clamp the two unambiguous
        # cases loudly instead of rendering garbage; the real fix is deleting
        # and re-adding the node (fresh, correctly-ordered widgets).
        import math as _math
        try:
            _hf = float(hires_factor)
        except (TypeError, ValueError):
            _hf = float("nan")
        if _math.isnan(_hf) or _math.isinf(_hf) or not (1.0 <= _hf <= 2.0):
            print(f"[JoyEcho] WARNING: hires_factor={hires_factor!r} is not a "
                  "sane value - your saved graph was made under an OLDER node "
                  "layout and the widget values have shifted slots. Using 1.0 "
                  "(hires off). Fix: delete + re-add the Generate node, then "
                  "re-enter your settings.", flush=True)
            hires_factor = 1.0
        if not (0.3 <= float(reference_zoom) <= 4.0):
            print(f"[JoyEcho] WARNING: reference_zoom={reference_zoom} is not a "
                  "sane value (widget-shift from an older graph layout). Using "
                  "1.2. Fix: delete + re-add the Generate node.", flush=True)
            reference_zoom = 1.2

        generator = model["generator"]
        video_vae = model["video_vae"]
        audio_vae = model["audio_vae"]
        audio_sample_rate = model["audio_sample_rate"]
        device = model["device"]
        dtype = model["dtype"]

        # Validate num_frames
        if (num_frames - 1) % 8 != 0:
            num_frames = 1 + ((num_frames - 1) // 8) * 8
            print(f"[JoyEcho] Adjusted num_frames to {num_frames} (must be 1 + 8*k)", flush=True)

        # Update generator resolution if changed
        generator.video_height = video_height
        generator.video_width = video_width
        generator.latent_height = video_height // 32
        generator.latent_width = video_width // 32
        generator.video_frame_seqlen = generator.latent_height * generator.latent_width
        # RENDER fps must drive the video RoPE clock (wrapper class default is
        # 24.0). At 25fps the hardcoded 24 skewed video rope-time 4% fast vs
        # audio's true-seconds rope -> ~40ms/s growing mouth-ahead-of-audio
        # drift, crossing visibility at ~9.6s into every shot (the "10s lip
        # sync cliff", 2026-07-23).
        generator.VIDEO_FPS = float(video_fps)
        if int(video_fps) != 24:
            print(f"[JoyEcho] WARNING: video_fps={video_fps}. The joint AV prior is "
                  f"24fps-native - non-24 fps drifts voices toward Commonwealth "
                  f"accents and overrides accent wording (verified 2026-07-29). "
                  f"Use 24 unless you specifically want that.", flush=True)

        # Compute latent shapes
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=num_frames,
            video_height=video_height,
            video_width=video_width,
            batch_size=1,
            video_fps=float(video_fps),
        )

        # Build pipelines
        denoising_sigmas = torch.tensor(DENOISING_SIGMAS, device=device, dtype=torch.float32)
        base_pipeline = BidirectionalAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
        )
        memory_pipeline = BidirectionalMemoryAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
            memory_downscale_factor=1,
        )

        # Memory bank
        memory_bank = PairedAudioVideoMemoryBank(
            max_size=memory_max_size,
            save_mode="random_every_shot_frame",
            num_fix_frames=num_fix_frames,
        )

        # REFERENCE CONDITIONING (I2V-as-reference): references are VIDEO-ONLY
        # conditioning clips kept OUT of the paired audio-video memory bank.
        # (The earlier design seeded them into the bank with zero-filled audio
        # latents; with 2+ refs those silent paired slots audibly polluted the
        # audio lane. The bank now holds only real generated shots; refs are
        # prepended at the video-memory encode step, invisible to all audio
        # machinery, and persist for the whole run.)
        _ref_clips = []
        _ref_sched = []  # must exist even with NO reference image (a no-character
                         # script legitimately produces none - first hit 2026-07-18)
        if reference_image is not None and memory_max_size > 0:
            import numpy as np
            from PIL import Image as _PILImage
            try:
                _sched_ints = [int(x) for x in str(reference_shots).split(",") if x.strip() != ""]
            except ValueError:
                _sched_ints = []
            def _sched_of(_j):
                return _sched_ints[_j] if _j < len(_sched_ints) else 0
            # Dedupe on (pixels, scheduled shot): the same picked image delivered
            # twice by a wiring quirk collapses, but the same image scheduled at
            # TWO different shots is a deliberate re-entry injection (character
            # returning after an absence) and every copy must survive.
            _uniq_idx = []
            _seen = []
            for _i in range(int(reference_image.shape[0])):
                _t = reference_image[_i]
                if not any(_sched_of(_i) == _s and _t.shape == _u.shape and torch.equal(_t, _u)
                           for _u, _s in _seen):
                    _seen.append((_t, _sched_of(_i)))
                    _uniq_idx.append(_i)
            if len(_uniq_idx) < int(reference_image.shape[0]):
                print(f"[JoyEcho] Reference batch: {int(reference_image.shape[0])} images, "
                      f"{len(_uniq_idx)} unique after dedupe.", flush=True)
            _tw, _th = int(video_width), int(video_height)
            _zoom = max(1.0, float(reference_zoom))
            _ref_sched = []
            for _ri in _uniq_idx[:6]:
                _arr = reference_image[_ri].detach().cpu().numpy()
                _arr = (np.clip(_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                _ref_pil = _PILImage.fromarray(_arr)
                # The memory encoder requires frames at EXACTLY the render size
                # (frames_to_video_tensor raises on mismatch). ALWAYS cover-fit
                # with a deliberate over-zoom: a reference at exactly the render
                # size is pixel-continuable, and the model opens the shot ON it
                # (a long literal full-frame ghost that no head trim covers).
                # Cropping in keeps identity and wide composition but breaks the
                # wholesale-continuation shortcut. Crop is mildly top-biased so
                # faces survive.
                _scale = max(_tw / _ref_pil.width, _th / _ref_pil.height) * _zoom
                _rw = max(_tw, int(round(_ref_pil.width * _scale)))
                _rh = max(_th, int(round(_ref_pil.height * _scale)))
                if (_rw, _rh) != _ref_pil.size:
                    _ref_pil = _ref_pil.resize((_rw, _rh), _PILImage.LANCZOS)
                if (_rw, _rh) != (_tw, _th):
                    _left = (_rw - _tw) // 2
                    _top = int((_rh - _th) * 0.25)  # bias crop toward the top (faces)
                    _ref_pil = _ref_pil.crop((_left, _top, _left + _tw, _top + _th))
                _ref_clips.append([_ref_pil] * 9)
                _ref_sched.append(_sched_ints[_ri] if _ri < len(_sched_ints) else 0)
            print(f"[JoyEcho] {len(_ref_clips)} reference image(s) prepared as VIDEO-ONLY "
                  f"conditioning clips ({_tw}x{_th}); audio lane untouched.", flush=True)

        all_video_frames = []
        all_audio_waveforms = []
        _hires_audio_lats = []
        _hires_video_lats = []  # spatial hires mode only (~10MB/shot on CPU)
        _hires_spatial = str(hires_denoise).lower().startswith("spatial")
        _hires_latent = str(hires_denoise).lower().startswith("latent")
        _temporal_on = str(temporal_upscale).lower().startswith("2x")
        if _temporal_on and hires_factor > 1.0:
            print("[JoyEcho] temporal_upscale requires hires_factor 1.0 in this "
                  "version - the hires passes re-save shots at base fps and would "
                  "undo it. Temporal upscale SKIPPED.", flush=True)
            _temporal_on = False

        num_shots = len(conditioning)
        offloader = None
        if sequential_offload:
            offloader = SequentialOffloader(generator, device,
                                            resident_blocks=resident_blocks)
        elif device.type == "cuda":
            # PRE-FLIGHT for no-offload runs. A config that cannot fit does not
            # raise a normal OOM here - cudaMallocAsync hard-aborts the entire
            # ComfyUI process mid-denoise (observed dying on the tiny timestep
            # embedding once weights + the AdaLN working set pinned VRAM full).
            # Refuse certain death with a catchable error; warn on tight fits.
            try:
                _free, _total = torch.cuda.mem_get_info()
                _pbytes = sum(p.numel() * p.element_size() for p in generator.parameters())
                _tokens = (video_height // 32) * (video_width // 32) * (num_frames // 8 + 1)
                _act_est = _tokens * 4096 * 2 * 32  # empirical AdaLN/attention working-set floor
            except Exception:
                _pbytes = 0
            if _pbytes:
                if _pbytes + 4 * 2**30 > _total:
                    raise ValueError(
                        f"sequential_offload=False but the DiT weighs {_pbytes/2**30:.1f}GiB and this "
                        f"GPU has {_total/2**30:.1f}GiB total. This cannot fit and would hard-abort "
                        f"the whole ComfyUI process (fatal CUDA error, not a catchable OOM). Enable "
                        f"sequential_offload, or shrink the weights (fp8_transformer / a GGUF DiT).")
                if _pbytes + _act_est + 2 * 2**30 > _free:
                    print(f"[JoyEcho] WARNING: no-offload budget is tight: DiT {_pbytes/2**30:.1f}GiB "
                          f"+ ~{_act_est/2**30:.1f}GiB est. activations ({_tokens} tokens) vs "
                          f"{_free/2**30:.1f}GiB free. A fatal process abort mid-denoise is likely - "
                          f"consider sequential_offload=True or lower resolution/frames.", flush=True)

        # Temporal-chunked VAE decode: 241f at 1280x736 decoded in ONE pass
        # hard-crashes a 32GB card (cuDNN abort mid-conv); 361f at 544x960
        # (~189M pixels*frames) is render-proven safe, so auto kicks in just
        # above that. Temporal-only tiling = no spatial seams; 24-frame
        # blended overlap.
        _decode_tiling_config = None
        if decode_tiling == "on" or (
            decode_tiling == "auto"
            and video_height * video_width * num_frames > 195_000_000
        ):
            from ltx_core.model.video_vae import TemporalTilingConfig, TilingConfig
            _decode_tiling_config = TilingConfig(
                spatial_config=None,
                temporal_config=TemporalTilingConfig(
                    tile_size_in_frames=64, tile_overlap_in_frames=24),
            )
            print("[JoyEcho] Tiled VAE decode ON (temporal 64f chunks, 24f overlap).",
                  flush=True)

        print(f"[JoyEcho] Generating {num_shots} shot(s) at {video_width}x{video_height}, "
              f"{num_frames} frames{' [sequential offload]' if sequential_offload else ''}...",
              flush=True)

        # PER-CHARACTER AUDIO MEMORY. The bank stores one slot per SHOT and is
        # otherwise character-blind, so with two characters alternating shots the
        # audio lane hands BOTH voices to every shot and the model may continue
        # either one - which is what makes two same-gender voices converge.
        # Tagging each slot with its speaker lets a shot inherit audio memory
        # only from the character actually talking in it. Empty = old behaviour.
        # The speaker_order WIDGET is an override. Left empty (the normal case in
        # a queue-driven workflow, where nobody retypes it per script), the order
        # comes from the script itself - either an explicit "speakers" array or
        # the "<ID> is talking" attribution already present in every shot. The
        # widget only exists for the rare case of overriding a script by hand.
        import re as _re_spk
        # Conditioning-carried metadata is the AUTHORITATIVE script-derived
        # source: it was computed from the exact text that produced these
        # embeddings and travels with them through the cache. The module-global
        # stash is retained only as a last-resort legacy fallback (it can be
        # stale when a loader node is ComfyUI-cached - audit finding 2026-07-29).
        _je_meta = {}
        try:
            _m0 = conditioning[0].get("joyecho_meta") if conditioning else None
            if isinstance(_m0, dict):
                _je_meta = _m0
        except (AttributeError, IndexError, TypeError):
            _je_meta = {}
        _speakers = [t for t in _re_spk.split(r"[,\s]+", str(speaker_order).strip()) if t]
        _spk_src = "speaker_order widget"
        if not _speakers and _je_meta.get("speakers"):
            _speakers = [str(x) for x in _je_meta["speakers"]]
            _spk_src = "script (via conditioning)"
        if not _speakers:
            try:
                from .joyecho_script_picker import LAST_SPEAKERS as _auto
            except Exception:
                try:
                    from joyecho_script_picker import LAST_SPEAKERS as _auto
                except Exception:
                    _auto = []
            if _auto:
                _speakers = list(_auto)
                _spk_src = "script (legacy stash - may be stale)"
        if _speakers:
            _speakers = [_speakers[i % len(_speakers)] for i in range(num_shots)]
            print(f"[JoyEcho] per-character audio memory ON (from {_spk_src}): "
                  f"{' '.join(_speakers)}", flush=True)
        else:
            print("[JoyEcho] per-character audio memory OFF - no speakers declared "
                  "and no '<ID> is talking' attribution found in the script.",
                  flush=True)

        # VOICE ANCHORS (2026-07-28). The memory bank guarantees CONSISTENCY, not
        # CORRECTNESS: shot 1 rolls its voice from text conditioning alone (empty
        # bank), and whatever region/timbre that roll lands on, the bank then
        # carries faithfully - measured twice today: two renders on different
        # seeds AND different DiT files produced the same wrong voices at the
        # same shots, because both merges share audio branches and the roll is
        # text-deterministic. No prompt wording pins it.
        #
        # So pin it with DATA: the script may carry
        #     "voice_refs": {"Rae": "path/to/clip.mp4", ...}
        # and each clip's audio is encoded (the pack's audio-VAE encoder, loaded
        # but never previously called) and stored as a bank slot TAGGED with the
        # character BEFORE the shot loop. Shot 1 then CONTINUES an approved voice
        # instead of auditioning a new one, and the same file re-casts the same
        # voice in every future render. Keys must exactly match the script's
        # speaker tags or per-character filtering will zero the anchor once the
        # speaker owns a generated slot.
        _voice_refs = {}
        if isinstance(_je_meta.get("voice_refs"), dict) and _je_meta["voice_refs"]:
            _voice_refs = dict(_je_meta["voice_refs"])    # authoritative: rode
        else:                                             # with the conditioning
            try:
                from .joyecho_script_picker import LAST_VOICE_REFS as _lvr
            except ImportError:
                try:
                    from joyecho_script_picker import LAST_VOICE_REFS as _lvr
                except ImportError:
                    _lvr = {}
            if isinstance(_lvr, dict):
                _voice_refs = dict(_lvr)
        # TIER-1 AUTO-CAST (2026-07-28): a speaker whose tag matches a folder
        # under input/joyecho_voices/ is cast automatically - no voice_refs key
        # needed. Explicit voice_refs entries WIN over the folder scan. The pick
        # is DETERMINISTIC (alphabetically first file): same character, same
        # file, same voice, every render. Dropping a file into the folder IS
        # casting the character; a random per-render pick would reintroduce the
        # per-video voice roulette this whole system exists to kill.
        try:
            import folder_paths as _fp
            _vroot = os.path.join(_fp.get_input_directory(), "joyecho_voices")
        except Exception:
            _fp, _vroot = None, ""
        if _vroot and os.path.isdir(_vroot) and _speakers:
            _vexts = (".mp4", ".wav", ".mp3", ".flac", ".mov", ".m4a", ".webm")
            for _sp in dict.fromkeys(_speakers):
                if _sp in _voice_refs:
                    continue
                _vd = os.path.join(_vroot, str(_sp).lower())
                if not os.path.isdir(_vd):
                    continue
                _vfiles = sorted(f for f in os.listdir(_vd)
                                 if f.lower().endswith(_vexts))
                if _vfiles:
                    _voice_refs[_sp] = os.path.join(_vd, _vfiles[0])
                    print(f"[JoyEcho] voice anchor {_sp!r}: auto-cast from folder "
                          f"({_vfiles[0]}).", flush=True)
        if _voice_refs and not enable_audio_memory:
            print("[JoyEcho] voice anchors present but enable_audio_memory is off - "
                  "anchors SKIPPED (the bank is the injection path).", flush=True)
        elif _voice_refs and memory_max_size > 0:
            from ltx_pipelines.utils.media_io import decode_audio_from_file
            from PIL import Image as _PILImage
            import av as _av
            for _vchar, _vpath in _voice_refs.items():
                try:
                    _aud = decode_audio_from_file(str(_vpath), torch.device("cpu"),
                                                  0.0, 6.0)
                    if _aud is None:
                        print(f"[JoyEcho] voice anchor {_vchar!r}: no audio stream "
                              f"in {_vpath}; skipped.", flush=True)
                        continue
                    # conv_in expects 2 channels; the bank's own normalizer is the
                    # canonical mono->stereo/downmix path.
                    _wav = type(memory_bank)._normalize_waveform_channels(
                        _aud.waveform, target_channels=2)
                    _lat = audio_vae.encode(_wav, int(_aud.sampling_rate))
                    # Generated slots store the pipeline's bf16 latents; a
                    # float32 seed would crash torch.cat at the next shot's
                    # kwargs build (belt: get_memory_audio also harmonizes).
                    _lat = _lat.to(torch.bfloat16).detach().cpu().contiguous()

                    # Video half: 9 frames from the SAME clip, cover-fit to the
                    # render size - the slot pairs the voice with the face that
                    # was producing it, same contract as generated slots.
                    _frames = []
                    _cont = _av.open(str(_vpath))
                    try:
                        _vs = next((s for s in _cont.streams if s.type == "video"),
                                   None)
                        if _vs is not None:
                            _all = [f.to_image() for _i, f in
                                    zip(range(150), _cont.decode(_vs))]
                            if _all:
                                _step = max(1, len(_all) // 9)
                                _frames = _all[::_step][:9]
                    finally:
                        _cont.close()
                    if not _frames and _fp is not None:
                        # Audio-only file (library wav/flac): pair the voice with
                        # the character's REF IMAGE - the same face the video
                        # lane is conditioned with - so the slot keeps its
                        # face+voice contract instead of being skipped.
                        try:
                            _rr = os.path.join(_fp.get_input_directory(),
                                               "joyecho_refs", str(_vchar).lower())
                            _imgs = sorted(
                                f for f in os.listdir(_rr)
                                if f.lower().endswith((".png", ".jpg", ".jpeg",
                                                       ".webp")))
                            if _imgs:
                                _frames = [_PILImage.open(
                                    os.path.join(_rr, _imgs[0])).convert("RGB")]
                                print(f"[JoyEcho] voice anchor {_vchar!r}: audio-"
                                      f"only file; video half from ref image "
                                      f"{_imgs[0]}.", flush=True)
                        except OSError:
                            pass
                    if not _frames:
                        print(f"[JoyEcho] voice anchor {_vchar!r}: no video frames "
                              f"and no ref image under joyecho_refs/"
                              f"{str(_vchar).lower()}/; skipped.", flush=True)
                        continue
                    _tw, _th = int(video_width), int(video_height)
                    _fitted = []
                    # loop var must NOT be _fp - that's the folder_paths alias,
                    # and shadowing it silently killed the SECOND character's
                    # audio-only ref-image fallback (audit finding, 2026-07-29)
                    for _frm in _frames:
                        _scale = max(_tw / _frm.width, _th / _frm.height)
                        _rw = max(_tw, int(round(_frm.width * _scale)))
                        _rh = max(_th, int(round(_frm.height * _scale)))
                        if (_rw, _rh) != _frm.size:
                            _frm = _frm.resize((_rw, _rh), _PILImage.LANCZOS)
                        if (_rw, _rh) != (_tw, _th):
                            _left = (_rw - _tw) // 2
                            _top = int((_rh - _th) * 0.25)
                            _frm = _frm.crop((_left, _top, _left + _tw, _top + _th))
                        _fitted.append(_frm)
                    while len(_fitted) < 9:
                        _fitted.append(_fitted[-1])

                    # Seeded TWICE: memory is cross-attention context, and with
                    # one slot the anchor only CONTESTS the text prior - measured
                    # 2026-07-28 evening: same config, different seeds, shot 1
                    # flipped American then Australian. Doubling the anchor's
                    # share of the context tips the coin toward the cast voice.
                    # (The real fix is AV-extend continuation of the anchor at
                    # each character's first shot - context can be ignored, a
                    # waveform being extended cannot. Until that lands, weight.)
                    for _rep in range(2):
                        memory_bank.save_memory_slot(
                            _fitted, _lat,
                            audio_window_size=audio_memory_window_size,
                            video_clip_num_frames=9,
                            audio_waveform=_aud.waveform,
                            audio_sample_rate=int(_aud.sampling_rate),
                            video_fps=float(video_fps),
                            audio_window_selection_mode="max_response",
                            video_frame_selection_mode="center",
                            character=str(_vchar),
                        )
                        # Tag as anchor: the anchor+latest policy keeps these
                        # slots live and protected from drift-outvoting.
                        memory_bank.memory[-1].metadata["voice_anchor"] = True
                    print(f"[JoyEcho] voice anchor {_vchar!r}: seeded from "
                          f"{os.path.basename(str(_vpath))} "
                          f"({_lat.shape[1]} audio tokens).", flush=True)
                except Exception as _vexc:
                    print(f"[JoyEcho] voice anchor {_vchar!r} FAILED "
                          f"({type(_vexc).__name__}: {_vexc}); continuing without.",
                          flush=True)

        for shot_idx in range(num_shots):
            _shot_speaker = _speakers[shot_idx] if _speakers else None
            prompt_seed = seed + shot_idx
            conditional_dict = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in conditioning[shot_idx].items()
                if k != "joyecho_meta"    # node-level metadata, not model input
            }

            print(f"[JoyEcho] Shot {shot_idx+1}/{num_shots}, seed={prompt_seed}, "
                  f"memory_size={len(memory_bank)}", flush=True)

            # --- Phase A: Denoise (generator on GPU, VAE on CPU) ---
            _move(video_vae.encoder, "cpu")
            _move(video_vae.decoder, "cpu")
            _move(audio_vae.encoder, "cpu")
            _move(audio_vae.decoder, "cpu")
            _move(audio_vae.vocoder, "cpu")
            if sequential_offload:
                offloader.install()
            else:
                _move(generator, device)
            _empty_cache()

            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(prompt_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(prompt_seed)

                # References seed IDENTITY on shot 1 only. From shot 2 the bank
                # holds real rendered frames carrying identity AND staging;
                # re-injecting a wide full-scene reference every shot keeps
                # offering the whole room as alternative staging and makes the
                # subject relocate between shots (observed: the character in a different
                # spot per shot). Bank-only continuity held identity fine in
                # every bank-era run.
                _shot_refs = [c for c, s in zip(_ref_clips, _ref_sched) if s == shot_idx]
                if _shot_refs:
                    print(f"[JoyEcho] injecting {len(_shot_refs)} reference(s) at shot {shot_idx+1}.",
                          flush=True)
                if _shot_refs or len(memory_bank) > 0:
                    # Encode memory frames (briefly bring video encoder to GPU).
                    # References prepend as pure video conditioning; the bank
                    # contributes only real generated shots.
                    _mem_frames = list(_shot_refs) + (memory_bank.get_memory_frames()
                                                      if len(memory_bank) > 0 else [])
                    _move(video_vae.encoder, device)
                    memory_video = encode_memory_frames_batch(
                        video_vae=video_vae,
                        batch_memory_frames=[_mem_frames],
                        target_h=video_height,
                        target_w=video_width,
                        device=device,
                        dtype=dtype,
                    )
                    _move(video_vae.encoder, "cpu")
                    _empty_cache()

                    # Audio memory kwargs come from the BANK ONLY (never refs);
                    # an empty bank means no audio-memory kwargs at all.
                    memory_audio_kwargs = {}
                    if len(memory_bank) > 0:
                        memory_audio_kwargs = build_paired_audio_memory_kwargs(
                            memory_bank,
                            enable_audio_memory=enable_audio_memory,
                            v2a_grad_scale=v2a_grad_scale,
                            memory_position_mode="reference",
                            speaker=_shot_speaker,
                        )
                    # Reference clips are prepended as VIDEO-only slots, but the
                    # audio side comes from the bank alone. The paired cross-mask
                    # zips the two slot lists POSITIONALLY, and
                    # _memory_slot_ranges_from_lengths falls back to an EVEN split
                    # whenever len(segment_lengths) != num_slots - so a single
                    # injected ref re-slices the whole audio memory at the wrong
                    # boundaries and every stored voice is smeared across the
                    # wrong slots. That is the "completely different voice on the
                    # re-entry shot" bug, not drift.
                    #
                    # Fix: give each ref its own silent audio slot at the FRONT so
                    # the counts match and the bank's real segments line up with
                    # their own video slots again. Zero-LENGTH padding cannot work
                    # here - _memory_slot_ranges_from_lengths drops empty ranges
                    # (`if end > start`) and the alignment breaks a second time.
                    if _shot_refs and memory_audio_kwargs:
                        _npad = len(_shot_refs)
                        _ma = memory_audio_kwargs.get("memory_audio")
                        _segs = memory_audio_kwargs.get("memory_audio_segment_lengths")
                        if _ma is not None and _segs:
                            _PAD_T = 8   # tokens per ref slot; small, contributes silence
                            _pad = torch.zeros(
                                (_ma.shape[0], _PAD_T * _npad, _ma.shape[2]),
                                dtype=_ma.dtype, device=_ma.device)
                            memory_audio_kwargs["memory_audio"] = torch.cat([_pad, _ma], dim=1)
                            memory_audio_kwargs["memory_audio_segment_lengths"] = tuple(
                                tuple([_PAD_T] * _npad) + tuple(row) for row in _segs)
                            memory_audio_kwargs["memory_audio_timestep"] = torch.zeros(
                                memory_audio_kwargs["memory_audio"].shape[:2],
                                dtype=torch.float32)
                            print(f"[JoyEcho] paired-memory realign: {_npad} reference slot(s) "
                                  f"padded with silent audio so {len(_mem_frames)} video slots "
                                  f"match {len(_mem_frames)} audio slots.", flush=True)

                    video_latent, audio_latent = memory_pipeline.generate(
                        video_shape=tuple(video_shape),
                        audio_shape=tuple(audio_shape),
                        conditional_dict=conditional_dict,
                        memory_video=memory_video,
                        seed=prompt_seed,
                        **memory_audio_kwargs,
                    )
                    del memory_video
                else:
                    video_latent, audio_latent = base_pipeline.generate(
                        video_shape=tuple(video_shape),
                        audio_shape=tuple(audio_shape),
                        conditional_dict=conditional_dict,
                        seed=prompt_seed,
                    )

            if device.type == "cuda":
                torch.cuda.synchronize()

            del conditional_dict
            _empty_cache()

            # Save audio latent for memory before decode moves things around.
            # NOTE: storage is deliberately NOT gated on enable_audio_memory —
            # the paired bank needs an audio slot to save the VIDEO slot, and
            # skipping the save silently disabled ALL cross-shot identity memory
            # whenever audio memory was off (diagnosed 2026-07-14: memory_size=0
            # every shot). enable_audio_memory still gates the INJECTION path
            # (build_paired_audio_memory_kwargs), which is where the wrong-rate
            # drone bug lived, so storing here reintroduces no audio artifacts.
            audio_memory_latent = (
                audio_latent.detach().cpu().contiguous()
                if audio_latent is not None
                else None
            )

            # --- Phase B: Decode (generator off GPU, VAE on GPU) ---
            if sequential_offload:
                offloader.remove()
            _move(generator, "cpu")
            _empty_cache()
            _move(video_vae.decoder, device)
            _move(audio_vae.decoder, device)
            _move(audio_vae.vocoder, device)

            video_uint8, audio_waveform = decode_benchmark_sample(
                video_vae, audio_vae, video_latent, audio_latent,
                video_tiling_config=_decode_tiling_config,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Temporal x2: second decode from the shot's OWN latents through the
            # LTX temporal upsampler while the decoder is still on GPU. Only the
            # SAVED shot files (and thus the worker-built master) get the doubled
            # fps; the bank, refs and in-graph frames stay base-fps native.
            _tu_uint8 = None
            if _temporal_on:
                try:
                    _tu_uint8 = self._temporal_upsample_decode(
                        video_vae, video_latent, _decode_tiling_config, device)
                    print(f"[JoyEcho] temporal x2: shot {shot_idx+1} "
                          f"{video_uint8.shape[0]}f -> {_tu_uint8.shape[0]}f "
                          f"(saved at {video_fps * 2} fps).", flush=True)
                except Exception as _tu_e:
                    print(f"[JoyEcho] temporal x2 FAILED on shot {shot_idx+1}: "
                          f"{_tu_e} - saving base-fps shot instead.", flush=True)
                    _tu_uint8 = None

            # Move VAE back to CPU
            _move(video_vae.decoder, "cpu")
            _move(audio_vae.decoder, "cpu")
            _move(audio_vae.vocoder, "cpu")
            _empty_cache()

            # Update memory bank
            memory_frames_pil = video_uint8_to_pil_frames(video_uint8)
            if audio_memory_latent is not None:
                memory_bank.save_memory_slot(
                    memory_frames_pil,
                    audio_memory_latent,
                    audio_window_size=audio_memory_window_size,
                    video_clip_num_frames=9,
                    audio_waveform=audio_waveform,
                    audio_sample_rate=16000,
                    video_fps=float(video_fps),
                    audio_window_selection_mode="max_response",
                    video_frame_selection_mode=(
                        "last" if str(memory_video_anchor) == "last" else "center"),
                    audio_memory_mel_bins=128,
                    audio_memory_mel_hop_length=160,
                    audio_memory_n_fft=1024,
                    audio_memory_downsample_factor=4,
                    audio_memory_is_causal=True,
                    character=_shot_speaker,
                )

            # HEAD TRIM: each shot's first frames morph out of the memory /
            # reference content (a split-second flash of the reference image).
            # Trim the DECODED shot ONCE, up front, with matching audio, so the
            # trim reaches EVERYTHING downstream: the final concatenated output,
            # the per-shot preview files, and any external concat of them. (The
            # memory-bank save above uses center-frame selection, so it never
            # saw the head flash and is left on the full shot.)
            _trim = max(0, int(head_trim_frames))
            if _trim == 0 and _ref_clips:
                _trim = 8  # auto when references are wired
            if _shot_refs:
                # A shot that RECEIVES references (its character's entry shot)
                # opens morphing out of the reference and needs the longer trim.
                _t1 = max(0, int(head_trim_first_shot))
                if _t1 == 0:
                    _t1 = max(2 * _trim, 24)  # auto
                _trim = max(_trim, _t1)
            if _trim > 0 and video_uint8.shape[0] > _trim + 16:
                video_uint8 = video_uint8[_trim:]
                if _tu_uint8 is not None:
                    _tu_uint8 = _tu_uint8[2 * _trim:]  # same cut in doubled frames
                if audio_waveform is not None:
                    _cut = int(round(_trim / float(video_fps) * audio_sample_rate))
                    if audio_waveform.shape[-1] > _cut:
                        audio_waveform = audio_waveform[..., _cut:]
                print(f"[JoyEcho] head-trim: dropped {_trim} frames from shot {shot_idx+1} "
                      f"(all outputs + per-shot preview).", flush=True)
            else:
                _trim = 0

            # Collect outputs (trimmed) as uint8 [F,H,W,3]. Float conversion is
            # deferred to final assembly (progressive fill there): storing shots
            # as float32 cost 4x the RAM, and at 10-shot hires the accumulated
            # float frames (~75GB at 1920x1088) plus the offloader's pinned host
            # pages hard-crashed the box at end-of-refine (2026-07-19).
            all_video_frames.append(video_uint8)
            if hires_factor > 1.0:
                # keep the shot's audio latent for the joint hires refine pass
                _hires_audio_lats.append(
                    audio_memory_latent.clone() if audio_memory_latent is not None else None)
                if _hires_spatial or _hires_latent:
                    # spatial AND latent modes work from the NATIVE latents
                    # (no VAE roundtrip) - stash them pre-decode, plus this
                    # shot's head-trim so the upscaled decode can be cut
                    # identically (latents are untrimmed)
                    _hires_video_lats.append(
                        (video_latent.detach().to("cpu", torch.float32).contiguous(),
                         int(_trim)))

            if audio_waveform is not None:
                from ltx_distillation.inference.memory_multishot import normalize_audio_waveform_for_media
                all_audio_waveforms.append(normalize_audio_waveform_for_media(audio_waveform))

            # Save per-shot video immediately for real-time preview (now trimmed,
            # so it matches the final output frame-for-frame).
            if _tu_uint8 is not None:
                # doubled-fps shot file: the AutoFinish worker probes r_frame_rate
                # per shot, so the master assembles at the doubled rate untouched
                self._save_shot_video(
                    _tu_uint8, audio_waveform, shot_idx,
                    video_fps * 2, audio_sample_rate, output_prefix
                )
                del _tu_uint8
            else:
                self._save_shot_video(
                    video_uint8, audio_waveform, shot_idx,
                    video_fps, audio_sample_rate, output_prefix
                )

            del video_latent, audio_latent, audio_memory_latent, video_uint8, audio_waveform
            _empty_cache()

            print(f"[JoyEcho] Shot {shot_idx+1}/{num_shots} done.", flush=True)

        # --- Optional hires-fix second pass: refine every shot at higher res.
        # Runs AFTER the loop so the memory bank and per-shot previews stay at
        # base res, and a failure here falls back to the base frames instead of
        # killing the render.
        if hires_factor > 1.0:
            try:
                if _hires_spatial:
                    self._hires_spatial_pass(
                        video_vae=video_vae, audio_vae=audio_vae,
                        video_lats=_hires_video_lats, frames_list=all_video_frames,
                        device=device, waveforms=all_audio_waveforms,
                        fps=video_fps, audio_sr=audio_sample_rate,
                        out_prefix=str(output_prefix) + "_hires",
                    )
                elif _hires_latent:
                    self._hires_latent_pass(
                        generator=generator, video_vae=video_vae,
                        audio_vae=audio_vae, conditioning=conditioning,
                        video_lats=_hires_video_lats,
                        frames_list=all_video_frames,
                        audio_lats=_hires_audio_lats, device=device,
                        x2="x2" in str(hires_denoise).split()[0].lower(),
                        turbo="turbo" in str(hires_denoise).lower(),
                        waveforms=all_audio_waveforms, fps=video_fps,
                        audio_sr=audio_sample_rate,
                        out_prefix=str(output_prefix) + "_hires",
                        base_seed=int(seed),
                    )
                else:
                    self._hires_refine_pass(
                        generator=generator, video_vae=video_vae, audio_vae=audio_vae,
                        conditioning=conditioning, frames_list=all_video_frames,
                        audio_lats=_hires_audio_lats, device=device,
                        factor=float(hires_factor), mode=str(hires_denoise),
                        sequential_offload=bool(sequential_offload),
                        resident_blocks=int(resident_blocks),
                        waveforms=all_audio_waveforms, fps=video_fps,
                        audio_sr=audio_sample_rate,
                        out_prefix=str(output_prefix) + "_hires",
                        base_seed=int(seed),
                    )
            except Exception:
                import traceback
                traceback.print_exc()
                print("[JoyEcho] WARNING: hires refine pass FAILED - delivering the "
                      "base-resolution frames instead.", flush=True)
            finally:
                _hires_audio_lats.clear()
                _hires_video_lats.clear()
                _empty_cache()

        # Concatenate all shots with the selected boundary treatment.
        xf = max(1, int(transition_frames)) if transition == "dissolve" else 0
        paired_audio = bool(all_audio_waveforms) and len(all_audio_waveforms) == len(all_video_frames)
        if xf > 0 and len(all_video_frames) > 1:
            vids = all_video_frames
            auds = all_audio_waveforms if paired_audio else None
            # Shots are stored uint8; the blend math needs float 0..1 -
            # convert lazily as each shot is consumed.
            out_v = vids[0].float().div_(255.0)
            out_a = auds[0] if auds else None
            for i in range(1, len(vids)):
                b_v = vids[i].float().div_(255.0)
                n = min(xf, out_v.shape[0], b_v.shape[0])
                if n <= 0:
                    out_v = torch.cat([out_v, b_v], dim=0)
                    if auds is not None:
                        out_a = torch.cat([out_a, auds[i]], dim=-1)
                    continue
                w = torch.linspace(0.0, 1.0, n, dtype=out_v.dtype).view(n, 1, 1, 1)
                blend = out_v[-n:] * (1.0 - w) + b_v[:n] * w
                out_v = torch.cat([out_v[:-n], blend, b_v[n:]], dim=0)
                if auds is not None:
                    b_a = auds[i]
                    n_s = min(int(round(n / float(video_fps) * audio_sample_rate)),
                              out_a.shape[-1], b_a.shape[-1])
                    if n_s > 0:
                        t = torch.linspace(0.0, 1.0, n_s, dtype=out_a.dtype)
                        fade_out = torch.cos(t * torch.pi / 2.0)
                        fade_in = torch.sin(t * torch.pi / 2.0)
                        a_blend = out_a[..., -n_s:] * fade_out + b_a[..., :n_s] * fade_in
                        out_a = torch.cat([out_a[..., :-n_s], a_blend, b_a[..., n_s:]], dim=-1)
                    else:
                        out_a = torch.cat([out_a, b_a], dim=-1)
            images = out_v
            print(f"[JoyEcho] Crossfaded {len(vids)-1} shot boundaries ({xf} frames each).", flush=True)
            audio_out = None
            if paired_audio:
                audio_out = {"waveform": out_a.unsqueeze(0), "sample_rate": audio_sample_rate}
            elif all_audio_waveforms:
                combined_waveform = torch.cat(all_audio_waveforms, dim=-1)
                audio_out = {"waveform": combined_waveform.unsqueeze(0), "sample_rate": audio_sample_rate}
        else:
            # Progressive uint8 -> float fill. A plain torch.cat(...).float()
            # holds the uint8 total AND the float total simultaneously; this
            # allocates the final float tensor once and frees each uint8 shot
            # as it lands, so peak = float total + one shot.
            _shot_lengths = [int(v.shape[0]) for v in all_video_frames]
            _total_f = sum(_shot_lengths)
            _H0 = int(all_video_frames[0].shape[1])
            _W0 = int(all_video_frames[0].shape[2])
            images = torch.empty((_total_f, _H0, _W0, 3), dtype=torch.float32)
            _pos = 0
            for _i in range(len(all_video_frames)):
                _v = all_video_frames[_i]
                images[_pos:_pos + _v.shape[0]] = _v.float().div_(255.0)
                _pos += _v.shape[0]
                all_video_frames[_i] = _v[:0]  # drop the uint8 payload
            audio_out = None
            if all_audio_waveforms:
                combined_waveform = torch.cat(all_audio_waveforms, dim=-1)  # [2, total_samples]
                audio_out = {
                    "waveform": combined_waveform.unsqueeze(0),  # [1, 2, samples]
                    "sample_rate": audio_sample_rate,
                }

            # VHS GLITCH transition: corrupt the frames AROUND each boundary in
            # place (snow, tearing bands, dropout lines) + a tape-static audio
            # hit. Total length unchanged; deterministic per seed+boundary.
            if transition == "vhs_glitch" and len(all_video_frames) > 1:
                n = max(1, int(transition_frames))
                amt_base = float(max(0.1, min(1.0, glitch_intensity)))
                boundaries = []
                acc = 0
                # _shot_lengths, not the list shapes - the progressive fill
                # above empties each shot tensor after copying it out.
                for _len in _shot_lengths[:-1]:
                    acc += _len
                    boundaries.append(acc)  # first frame index of the NEXT shot
                total_f = images.shape[0]
                H, W = images.shape[1], images.shape[2]
                for bi, b in enumerate(boundaries):
                    g = torch.Generator().manual_seed(int(seed) * 1009 + bi)
                    start = max(0, b - n // 2)
                    end = min(total_f, start + n)
                    span = max(1, end - start - 1)
                    for k, fidx in enumerate(range(start, end)):
                        env = 1.0 - abs((k - span / 2.0) / (span / 2.0 or 1.0))
                        amt = amt_base * (0.35 + 0.65 * max(0.0, env))
                        f = images[fidx]
                        # snow (monochrome noise mix)
                        snow = torch.rand((H, W, 1), generator=g).expand(H, W, 3)
                        f = f * (1.0 - amt * 0.8) + snow * (amt * 0.8)
                        # horizontal tearing bands
                        for _ in range(int(1 + amt * 6)):
                            y0 = int(torch.randint(0, max(1, H - 8), (1,), generator=g))
                            bh = int(torch.randint(2, max(3, H // 20), (1,), generator=g))
                            dx = int(torch.randint(-W // 6, W // 6 + 1, (1,), generator=g))
                            f[y0:y0 + bh] = torch.roll(f[y0:y0 + bh], shifts=dx, dims=1)
                        # dropout scanlines
                        for _ in range(int(amt * 4)):
                            y = int(torch.randint(0, H, (1,), generator=g))
                            f[y:y + 1] = float(torch.rand((1,), generator=g))
                        images[fidx] = f.clamp(0.0, 1.0)
                    # audio: tape-static bed over a WIDER window than the video
                    # burst - JoyAI's per-shot room tone fades out at shot edges,
                    # so the static must SPAN that dead seam (>=1.2s), not just
                    # the few glitched frames, or the cut reads as burst->dead
                    # air->tone. Quieter hit with a smooth raised-cosine envelope
                    # (the old short triangular hit at 0.22 was sharp and loud).
                    if audio_out is not None:
                        wav = audio_out["waveform"][0]  # [2, samples]
                        c = int(round(b / float(video_fps) * audio_sample_rate))
                        n_s = max(int(round(n / float(video_fps) * audio_sample_rate)),
                                  int(round(1.2 * audio_sample_rate)))
                        s0 = max(0, c - n_s // 2)
                        s1 = min(wav.shape[-1], s0 + n_s)
                        if s1 > s0:
                            ln = s1 - s0
                            t = torch.linspace(0.0, 1.0, ln)
                            env_a = 0.5 - 0.5 * torch.cos(t * 2.0 * torch.pi)  # raised cosine
                            noise = (torch.rand((wav.shape[0], ln), generator=g) * 2.0 - 1.0)
                            wav[..., s0:s1] = (wav[..., s0:s1] * (1.0 - 0.35 * amt_base * env_a)
                                               + noise * (0.10 * amt_base) * env_a).clamp(-1.0, 1.0)
                print(f"[JoyEcho] VHS glitch applied at {len(boundaries)} boundaries "
                      f"({n} frames, intensity {amt_base}).", flush=True)

        # Sidecar for the AutoFinish worker: the worker rebuilds the master from
        # the PRE-glitch per-shot files and cannot see these widgets, so record
        # them next to the shots. Written for every transition type so the
        # worker honors "cut"/"dissolve" too (its own defaults only apply when
        # no same-run sidecar exists).
        try:
            import json as _json
            import folder_paths as _fpaths
            _parts = str(output_prefix).rsplit("/", 1)
            _sdir = os.path.join(_fpaths.get_output_directory(),
                                 _parts[0] if len(_parts) == 2 else "")
            os.makedirs(_sdir, exist_ok=True)
            with open(os.path.join(_sdir, "_transition.json"), "w", encoding="utf-8") as _fh:
                _json.dump({"transition": str(transition),
                            "frames": int(transition_frames),
                            "intensity": float(glitch_intensity),
                            "seed": int(seed)}, _fh)
        except Exception as _e:  # noqa: BLE001
            print(f"[JoyEcho] WARNING: transition sidecar not written ({_e}).", flush=True)

        print(f"[JoyEcho] Generation complete. {images.shape[0]} frames, "
              f"{num_shots} shot(s).", flush=True)

        return (images, audio_out,)

    def _hires_spatial_pass(self, *, video_vae, audio_vae, video_lats, frames_list,
                            device, waveforms=None, fps=25, audio_sr=48000,
                            out_prefix=""):
        """Deterministic hires: run each shot's NATIVE latents through the LTX
        x1.5 spatial upscaler (one forward pass, no noise, no re-generation),
        then decode at the higher resolution. The standard-workflow upscale
        path, minus the VAE roundtrip the post-pass version pays. Temporal
        reflect-pad mirrors the RIFT padded upsampler (the x1.5-1.0 model
        corrupts sequence ends; TPAD=8 latents puts the damage in disposable
        mirrors). Factor is FIXED at 1.5 by the model - hires_factor is
        ignored beyond enabling the pass. x1.5 needs EVEN spatial latent dims
        (heights like 704/768, not 736) or it fabricates an edge band.
        """
        import time as _time
        import json as _json
        import folder_paths
        import comfy.utils as _cutils
        from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
        from ltx_core.model.video_vae import TemporalTilingConfig, TilingConfig
        from ltx_distillation.utils import decode_benchmark_sample

        _t0 = _time.time()
        _dt = torch.bfloat16
        UPSCALER_FILE = "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"
        TPAD = 8

        path = folder_paths.get_full_path_or_raise("latent_upscale_models", UPSCALER_FILE)
        sd, metadata = _cutils.load_torch_file(path, safe_load=True, return_metadata=True)
        up_model = LatentUpsampler.from_config(_json.loads(metadata["config"])).to(dtype=_dt)
        up_model.load_state_dict(sd)
        up_model.eval().to(device)
        del sd

        stats = video_vae.decoder.per_channel_statistics
        _move(video_vae.decoder, device)
        tiling = TilingConfig(
            spatial_config=None,
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64,
                                                 tile_overlap_in_frames=24))
        try:
            for si, (lat, trim) in enumerate(video_lats):
                # [1,F,C,h,w] normalized (pipeline layout) -> [1,C,F,h,w] raw
                lat = lat.permute(0, 2, 1, 3, 4)
                h_lat, w_lat = int(lat.shape[3]), int(lat.shape[4])
                if h_lat % 2 or w_lat % 2:
                    print(f"[JoyEcho] Hires-spatial WARNING: odd latent dim "
                          f"(h={h_lat}, w={w_lat}) - the x1.5 model will smear one "
                          f"edge. Use heights/widths whose /32 is EVEN (768, not 736).",
                          flush=True)
                # stats is a decoder submodule - it moved to device with it
                raw = stats.un_normalize(lat.to(device))
                # temporal reflect pad (no edge repeat), crop after
                f = raw.shape[2]
                p = min(TPAD, f - 1)
                if p > 0:
                    raw = torch.cat([raw[:, :, 1:p + 1].flip(2), raw,
                                     raw[:, :, -(p + 1):-1].flip(2)], dim=2)
                with torch.no_grad():
                    up = up_model(raw.to(device=device, dtype=_dt))
                del raw
                if p > 0:
                    up = up[:, :, p:-p]
                up = stats.normalize(up.to(torch.float32)).to(_dt)
                up = up.permute(0, 2, 1, 3, 4).contiguous()  # back to [1,F,C,H,W]
                _empty_cache()

                u8, _ = decode_benchmark_sample(video_vae, audio_vae, up.to(device), None,
                                                video_tiling_config=tiling)
                del up
                _empty_cache()
                if trim > 0 and u8.shape[0] > trim:
                    u8 = u8[trim:]
                frames_list[si] = u8
                if out_prefix:
                    try:
                        _wav = (waveforms[si]
                                if waveforms is not None and si < len(waveforms) else None)
                        self._save_shot_video(u8, _wav, si, fps, audio_sr, out_prefix)
                    except Exception as _e:
                        print(f"[JoyEcho] Hires-spatial: per-shot save failed for "
                              f"shot {si+1}: {_e}", flush=True)
                print(f"[JoyEcho] Hires-spatial: shot {si+1}/{len(video_lats)} "
                      f"upscaled ({u8.shape[2]}x{u8.shape[1]}).", flush=True)
        finally:
            up_model.cpu()
            _move(video_vae.decoder, "cpu")
            _empty_cache()
        print(f"[JoyEcho] Hires-spatial pass done in {_time.time()-_t0:.0f}s.", flush=True)

    def _hires_latent_pass(self, *, generator, video_vae, audio_vae, conditioning,
                           video_lats, frames_list, audio_lats, device, x2,
                           turbo=False, waveforms=None, fps=25, audio_sr=48000,
                           out_prefix="", base_seed=0):
        """Whole-shot latent-space anneal - the full USE THIS ONE architecture.

        The windowed refine modes commit a texture identity PER 33f PIXEL
        WINDOW: decoded pixels are re-encoded through the VAE, denoised
        independently, and blended in pixel space, so fine detail rotates at
        the window cadence. Measured on the 2026-08-03 master: cuff weave
        changes once per second; aligned-crop texture correlation 0.38-0.60
        within a window vs 0.22-0.35 across windows. It reads as "each
        second is a new set of fine detail".

        This pass never leaves latent space: the shot's OWN generation latents
        (stashed pre-decode, like spatial mode) -> learned latent upsampler ->
        the gentle ladder over LARGE temporal chunks with a latent-domain
        crossfade -> ONE tiled decode. No VAE re-encode exists anywhere, and
        texture identity holds for the chunk length (~4-11 s) instead of 1 s.

        Sequential offload is FORCED here regardless of the widget: chunk size
        is set by the activation budget, and streamed weights buy ~3x bigger
        chunks for a few seconds of PCIe per ladder step. resident_blocks=0
        for the same reason as the windowed pass.
        """
        import time as _time
        import json as _json
        import folder_paths
        import comfy.utils as _cutils
        from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
        from ltx_core.model.video_vae import TemporalTilingConfig, TilingConfig
        from ltx_distillation.utils import add_noise, decode_benchmark_sample

        _t0 = _time.time()
        _dt = torch.bfloat16
        # turbo = the stock "RIFT - Z-Image + LTX" refine recipe (read from its
        # graph 2026-08-03): re-noise the upsampled latent all the way to 0.85
        # so the model RE-SYNTHESIZES texture at target res, then the distilled
        # 3-step tail. The gentle ladder enters at 0.40 and only polishes -
        # more steps, softer detail.
        LADDER = ([0.85, 0.725, 0.4219, 0.0] if turbo
                  else [0.400, 0.340, 0.280, 0.220, 0.165, 0.110, 0.055, 0.0])
        TPAD = 8
        OVL = 6                      # latent frames of crossfade (= 2.0 s)
        TOKEN_BUDGET = 90_000        # 98.5k is proven on the 5090; keep margin
        _ufile = ("ltx-2.3-spatial-upscaler-x2-1.1.safetensors" if x2
                  else "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors")
        _uscale = 2.0 if x2 else 1.5

        lat0 = video_lats[0][0]
        h_lat, w_lat = int(lat0.shape[3]), int(lat0.shape[4])
        if not x2 and (h_lat % 2 or w_lat % 2):
            print(f"[JoyEcho] Hires-latent: odd latent dims (h={h_lat}, "
                  f"w={w_lat}) smear an edge through the x1.5 model - use "
                  f"768-class dims, or latent-x2 (no such limit).", flush=True)
        H_lat, W_lat = int(round(h_lat * _uscale)), int(round(w_lat * _uscale))
        th, tw = H_lat * 32, W_lat * 32
        tok_lf = H_lat * W_lat
        chunk = max(8, min(40, TOKEN_BUDGET // tok_lf))
        print(f"[JoyEcho] Hires-latent{' TURBO' if turbo else ''}: "
              f"{w_lat*32}x{h_lat*32} -> {tw}x{th} "
              f"(x{_uscale}), {len(LADDER)-1}-step ladder, chunks of {chunk} "
              f"latent frames (~{chunk*8//24}s) + {OVL}f latent blend, "
              f"{tok_lf*chunk:,} tokens/chunk.", flush=True)

        path = folder_paths.get_full_path_or_raise("latent_upscale_models", _ufile)
        sd, md = _cutils.load_torch_file(path, safe_load=True, return_metadata=True)
        up_model = LatentUpsampler.from_config(_json.loads(md["config"])).to(dtype=_dt)
        up_model.load_state_dict(sd)
        up_model.eval().to(device)
        del sd

        tiling = TilingConfig(
            spatial_config=None,
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64,
                                                 tile_overlap_in_frames=24))
        _res_saved = (generator.video_height, generator.video_width,
                      generator.latent_height, generator.latent_width,
                      generator.video_frame_seqlen)
        generator.video_height = th
        generator.video_width = tw
        generator.latent_height = H_lat
        generator.latent_width = W_lat
        generator.video_frame_seqlen = H_lat * W_lat
        generator.VIDEO_FPS = float(fps)

        _n_res = 0
        if turbo:
            try:
                _tot_vram = torch.cuda.get_device_properties(device).total_memory
                _n_res = 6 if _tot_vram > 28e9 else 0
            except Exception:
                _n_res = 0
        offl = SequentialOffloader(generator, device, resident_blocks=_n_res)
        offl.install()
        _move(video_vae.decoder, device)
        stats = video_vae.decoder.per_channel_statistics

        try:
            for si, (lat, trim) in enumerate(video_lats):
                if audio_lats[si] is None:
                    print(f"[JoyEcho] Hires-latent: shot {si+1} has no audio "
                          f"latent; skipped.", flush=True)
                    continue
                # 1. upsample the WHOLE shot's latents, reflect-padded
                raw = stats.un_normalize(
                    lat.permute(0, 2, 1, 3, 4).to(device, torch.float32))
                f = int(raw.shape[2])
                p = min(TPAD, f - 1)
                if p > 0:
                    raw = torch.cat([raw[:, :, 1:p + 1].flip(2), raw,
                                     raw[:, :, -(p + 1):-1].flip(2)], dim=2)
                with torch.no_grad():
                    up = up_model(raw.to(dtype=_dt))
                del raw
                if p > 0:
                    up = up[:, :, p:-p]
                up = stats.normalize(up.to(torch.float32)).to(_dt)
                full = up.permute(0, 2, 1, 3, 4).contiguous().cpu()  # [1,F,C,H,W]
                del up
                _empty_cache()
                F_up = int(full.shape[1])

                cond = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in conditioning[si].items()
                        if k != "joyecho_meta"}
                alat_full = audio_lats[si]
                Fa = int(alat_full.shape[1])

                # per-shot noise fields, one per noising event, sliced by
                # GLOBAL latent index (the HIRES V2 rule: chunks share one
                # field, never draw fresh noise)
                gs = torch.Generator(device="cpu").manual_seed(
                    int(base_seed) + 5678 + si * 100)
                n_events = len(LADDER) - 1
                fields_v = [torch.randn((1, F_up) + tuple(full.shape[2:]),
                                        generator=gs) for _ in range(n_events)]
                fields_a = [torch.randn((1, Fa) + tuple(alat_full.shape[2:]),
                                        generator=gs) for _ in range(n_events)]

                starts = list(range(0, max(F_up - chunk, 0) + 1,
                                    max(chunk - OVL, 1)))
                if not starts:
                    starts = [0]
                if starts[-1] + chunk < F_up:
                    starts.append(F_up - chunk)
                out_lat = torch.zeros(tuple(full.shape), dtype=torch.float32)
                wsum = torch.zeros((1, F_up, 1, 1, 1), dtype=torch.float32)

                for ci, s in enumerate(starts):
                    e = min(s + chunk, F_up)
                    Fl = e - s
                    v = full[:, s:e].to(device)
                    a0 = int(round(Fa * s / float(F_up)))
                    a1 = max(a0 + 2, int(round(Fa * e / float(F_up))))
                    alat = alat_full[:, a0:min(a1, Fa)].to(device=device, dtype=_dt)
                    Fa_w = int(alat.shape[1])

                    nv = fields_v[0][:, s:e].to(device=device, dtype=_dt)
                    na = fields_a[0][:, a0:a0 + Fa_w].to(device=device, dtype=_dt)
                    s0 = torch.full((1, Fl), LADDER[0])
                    sa0 = torch.full((1, Fa_w), LADDER[0])
                    v = add_noise(v.flatten(0, 1), nv.flatten(0, 1),
                                  s0.flatten(0, 1)).unflatten(0, (1, Fl))
                    a = add_noise(alat, na, sa0)
                    del nv, na

                    for i_s, sig in enumerate(LADDER[:-1]):
                        vs = sig * torch.ones((1, Fl), device=device)
                        as_ = sig * torch.ones((1, Fa_w), device=device)
                        pred_v, pred_a = generator(
                            noisy_image_or_video=v, conditional_dict=cond,
                            timestep=vs, noisy_audio=a, audio_timestep=as_)
                        nxt = LADDER[i_s + 1]
                        if nxt > 0:
                            fnv = fields_v[i_s + 1][:, s:e].to(device=device,
                                                               dtype=_dt)
                            fna = fields_a[i_s + 1][:, a0:a0 + Fa_w].to(
                                device=device, dtype=_dt)
                            nvs = nxt * torch.ones((1, Fl), device=device)
                            nas = nxt * torch.ones((1, Fa_w), device=device)
                            v = add_noise(pred_v.flatten(0, 1), fnv.flatten(0, 1),
                                          nvs.flatten(0, 1)).unflatten(0, (1, Fl))
                            a = add_noise(pred_a, fna, nas)
                            del fnv, fna
                        else:
                            v = pred_v
                            a = pred_a
                    del a, alat

                    w = torch.ones((1, Fl, 1, 1, 1), dtype=torch.float32)
                    if s > 0:
                        r = min(OVL, Fl)
                        w[0, :r, 0, 0, 0] = torch.linspace(0.0, 1.0, r)
                    out_lat[:, s:e] += v.detach().float().cpu() * w
                    wsum[:, s:e] += w
                    del v
                    _empty_cache()
                    print(f"[JoyEcho] Hires-latent: shot {si+1} chunk "
                          f"{ci+1}/{len(starts)} done.", flush=True)

                refined = (out_lat / wsum.clamp_min(1e-6)).to(_dt)
                del out_lat, wsum, fields_v, fields_a
                u8, _ = decode_benchmark_sample(video_vae, audio_vae,
                                                refined.to(device), None,
                                                video_tiling_config=tiling)
                del refined
                _empty_cache()
                if trim > 0 and u8.shape[0] > trim:
                    u8 = u8[trim:]
                frames_list[si] = u8
                if out_prefix:
                    try:
                        _wav = (waveforms[si]
                                if waveforms is not None and si < len(waveforms)
                                else None)
                        self._save_shot_video(u8, _wav, si, fps, audio_sr,
                                              out_prefix)
                    except Exception as _e:
                        print(f"[JoyEcho] Hires-latent: per-shot save failed "
                              f"for shot {si+1}: {_e}", flush=True)
                print(f"[JoyEcho] Hires-latent: shot {si+1}/{len(video_lats)} "
                      f"refined + decoded ({u8.shape[2]}x{u8.shape[1]}).",
                      flush=True)
        finally:
            (generator.video_height, generator.video_width,
             generator.latent_height, generator.latent_width,
             generator.video_frame_seqlen) = _res_saved
            offl.remove()
            up_model.cpu()
            _move(generator, "cpu")
            _move(video_vae.decoder, "cpu")
            _empty_cache()
        print(f"[JoyEcho] Hires-latent pass done in {_time.time()-_t0:.0f}s.",
              flush=True)

    def _hires_refine_pass(self, *, generator, video_vae, audio_vae, conditioning,
                           frames_list, audio_lats, device, factor, mode,
                           sequential_offload, resident_blocks,
                           waveforms=None, fps=25, audio_sr=48000, out_prefix="",
                           base_seed=0):
        """Second-pass hires fix: per shot, upscale (bicubic) -> VAE re-encode ->
        re-noise at a tail sigma -> re-denoise through the DMD ladder tail at the
        TARGET resolution -> decode. The model synthesizes genuine detail the
        base render never had (RTX-class upscalers only sharpen what exists).

        Video-only output: audio latents ride along for the joint AV model's
        cross-attention, then the refined audio is discarded (the original
        track is untouched). Runs in 65-frame windows with an 8-frame linear
        cross-fade so a 24GB card survives 1920x1088 refines.
        """
        import time as _time
        from PIL import Image as _PILImage
        import numpy as _np
        from ltx_distillation.utils import add_noise, decode_benchmark_sample, frames_to_video_tensor
        from ltx_core.model.video_vae import TemporalTilingConfig, TilingConfig

        _t0 = _time.time()
        _dt = torch.bfloat16  # fixed: never derive dtype from parameters (fp8 trap)
        SIGMA_TAILS = {"subtle": [0.421875, 0.0], "medium": [0.725, 0.421875, 0.0],
                       # tenstrip's published UPSCALE ladder (HF, ~Jul 1) -
                       # purpose-built for a refine pass, unlike the generation
                       # tails above. (Shallower/frozen/reused-noise variants
                       # were tested 2026-07-23 and removed: none reduced the
                       # per-frame detail churn - see project notes.)
                       "strong": [0.92, 0.725, 0.421875, 0.0],
                       # USE THIS ONE's refine ladder (live graph, verified
                       # 2026-08-02) - and that graph does NOT churn. Measured
                       # on audition_hires_000 vs its base: the base's static-
                       # region detail correlation is flat ~0.85 out to lag 24,
                       # the refined clip decays 0.68 -> 0.37. The churn is the
                       # single big jump committing one x0 guess per frame;
                       # many small steps let each step SEE the previous step's
                       # detail and reinforce it, annealing one texture
                       # identity instead of re-rolling it. Frozen noise could
                       # never fix this (and measurably didn't): the variance
                       # is in the model's per-frame x0 uncertainty, not in the
                       # noise draw. Pairs with the latent-upsampler input
                       # below - the second half of the same recipe.
                       "anneal": [0.400, 0.340, 0.280, 0.220, 0.165, 0.110,
                                  0.055, 0.0],
                       # same ladder through the x2-1.1 upsampler - the FIXED
                       # upstream checkpoint (the sequence-end corruption that
                       # forced the padded wrapper was fixed in 1.1 for x2;
                       # x1.5 never got a 1.1). Built for low-VRAM natives:
                       # a 3090's 576x1024 -> 1152x2048 clears vertical 1080p.
                       "anneal-x2": [0.400, 0.340, 0.280, 0.220, 0.165, 0.110,
                                     0.055, 0.0]}
        sigmas = SIGMA_TAILS.get(str(mode).split()[0].lower(), SIGMA_TAILS["subtle"])
        base_h, base_w = int(frames_list[0].shape[1]), int(frames_list[0].shape[2])
        # ANNEAL mode = the USE THIS ONE architecture ported here: LATENT
        # upsample (learned x1.5) feeding the gentle ladder, instead of bicubic
        # pixels VAE-re-encoded. Bicubic-then-encode hands the model a soft,
        # off-manifold latent whose ENTIRE detail layer must be invented -
        # invention is where the per-frame variance lives. The upsampled latent
        # already carries coherent detail; the ladder only polishes it.
        _mode0 = str(mode).split()[0].lower()
        _anneal = _mode0.startswith("anneal")
        _x2 = _mode0 == "anneal-x2"
        _up_model = None
        if _anneal:
            # the upsampler model fixes the factor; hires_factor only enabled
            # the pass. x2 uses the FIXED 1.1 checkpoint; x1.5 only exists as
            # the 1.0 (reflect-pad covers its end-corruption either way).
            _uscale = 2.0 if _x2 else 1.5
            _ufile = ("ltx-2.3-spatial-upscaler-x2-1.1.safetensors" if _x2
                      else "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors")
            th = int(base_h * _uscale)
            tw = int(base_w * _uscale)
            if float(factor) not in (1.0, _uscale):
                print(f"[JoyEcho] Hires-{_mode0}: factor {factor} ignored - "
                      f"the latent upsampler is fixed x{_uscale} "
                      f"({base_w}x{base_h} -> {tw}x{th}).", flush=True)
            # x1.5 only: its rational resampler ends in a stride-2 blur-
            # downsample, which smears an edge on odd /32 latent dims and can
            # break the target's /32 alignment. x2 is a pure PixelShuffle
            # (den=1, no downsample) and doubles any /32 canvas cleanly.
            if not _x2 and ((base_h // 32) % 2 or (base_w // 32) % 2
                            or th % 32 or tw % 32):
                # fall back to the bicubic input path but KEEP the anneal
                # ladder (half the recipe).
                print(f"[JoyEcho] Hires-anneal: base {base_w}x{base_h} has odd "
                      f"/32 latent dims - x1.5 latent upsampler skipped (use "
                      f"768-class heights, not 736/864 - or anneal-x2, which "
                      f"has no such limit). Running the anneal ladder on the "
                      f"bicubic path instead.", flush=True)
                _anneal = False
                th = max(32, int(round(base_h * 1.5 / 32.0)) * 32)
                tw = max(32, int(round(base_w * 1.5 / 32.0)) * 32)
        else:
            th = max(32, int(round(base_h * factor / 32.0)) * 32)
            tw = max(32, int(round(base_w * factor / 32.0)) * 32)
        if (th, tw) == (base_h, base_w):
            print("[JoyEcho] Hires: factor rounds to the base resolution; skipping.", flush=True)
            return
        if _anneal:
            import json as _json
            import folder_paths
            import comfy.utils as _cutils
            from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
            _upath = folder_paths.get_full_path_or_raise(
                "latent_upscale_models", _ufile)
            _usd, _umd = _cutils.load_torch_file(_upath, safe_load=True,
                                                 return_metadata=True)
            _up_model = LatentUpsampler.from_config(
                _json.loads(_umd["config"])).to(dtype=_dt)
            _up_model.load_state_dict(_usd)
            _up_model.eval().to(device)
            del _usd
        # 33-frame windows (8n+1): a 65f window at 1920x1088 hard-aborts the VAE
        # ENCODER (uncatchable cuDNN abort - the encode sibling of the decode
        # crash that tiled decode fixed). 33f keeps encode peaks ~half the
        # proven-safe decode chunk size.
        WIN, OVL = 33, 9
        # ANNEAL runs 7 ladder steps per window, so window COUNT dominates its
        # wall time (a 25s shot at 33f/24-stride is ~24 windows -> ~168 DiT
        # forwards, 3-4x the base render). The 33f cap existed because the
        # bicubic path VAE-ENCODES at TARGET res and a 65f target-res encode
        # hard-aborts - but anneal encodes at BASE res, so that ceiling does
        # not apply to it. 65f at a <=1.05MP base costs the same encoder
        # activations as the proven-safe 33f at a 2.1MP target (0.5x pixels x
        # 2x frames), and the DiT window stays small (~27k tokens at x2 on
        # 1344x768). Stride 56 = 7 latent frames, so the global noise-field
        # indexing stays exact. Result: ~24 windows -> ~10, roughly 2.3x
        # faster, identical math per window.
        if _anneal and (base_h * base_w) <= 1344 * 768:
            WIN, OVL = 65, 9
            print(f"[JoyEcho] Hires-{_mode0}: base-res encode allows 65f "
                  f"windows ({(base_h * base_w) / 1e6:.2f}MP base).", flush=True)
        print(f"[JoyEcho] Hires refine: {base_w}x{base_h} -> {tw}x{th}, "
              f"start sigma {sigmas[0]} ({len(sigmas)-1} step(s)), "
              f"{WIN}f windows / {OVL}f blend.", flush=True)

        tiling = TilingConfig(
            spatial_config=None,
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64,
                                                 tile_overlap_in_frames=24))

        # Device setup once for the whole pass. resident_blocks is deliberately
        # 0 here: pinned DiT blocks (~12GB at 30 resident) compete with the VAE
        # encoder's conv activations at target res - exactly what pushed the
        # 65f encode into a fatal abort. The refine is 1-2 steps per window, so
        # streaming all blocks costs seconds while freeing the VRAM the VAE needs.
        # The wrapper caches the RENDER resolution (video_frame_seqlen drives the
        # per-token timestep expansion; latent_height/width drive unpatchify).
        # generate() sets them once from the base res, so a refine at target res
        # feeds 2040-token frames against a 1008-token modulation vector ->
        # "size of tensor a (10200) must match tensor b (5040)". Retarget for the
        # pass, restore in the finally so the memory bank / next queue item are
        # unaffected.
        _res_saved = (generator.video_height, generator.video_width,
                      generator.latent_height, generator.latent_width,
                      generator.video_frame_seqlen)
        generator.video_height = th
        generator.video_width = tw
        generator.latent_height = th // 32
        generator.latent_width = tw // 32
        generator.video_frame_seqlen = generator.latent_height * generator.latent_width
        generator.VIDEO_FPS = float(fps)  # refine RoPE clock = render fps too

        offl = None
        if sequential_offload:
            offl = SequentialOffloader(generator, device, resident_blocks=0)
            offl.install()
        else:
            _move(generator, device)
        _move(video_vae.encoder, device)
        _move(video_vae.decoder, device)

        try:
            for si in range(len(frames_list)):
                if audio_lats[si] is None:
                    print(f"[JoyEcho] Hires: shot {si+1} has no audio latent; skipped.", flush=True)
                    continue
                vf = frames_list[si]                       # [T,H,W,3] uint8
                T = int(vf.shape[0])
                if T < WIN:
                    print(f"[JoyEcho] Hires: shot {si+1} shorter than a window ({T}f); skipped.", flush=True)
                    continue
                cond = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in conditioning[si].items()
                        if k != "joyecho_meta"}
                alat_full = audio_lats[si]                 # [1,Fa,C] cpu
                Fa = int(alat_full.shape[1])

                starts = list(range(0, T - WIN + 1, WIN - OVL))
                if starts[-1] + WIN < T:
                    starts.append(T - WIN)
                out = torch.zeros((T, th, tw, 3), dtype=torch.float32)
                wsum = torch.zeros((T, 1, 1, 1), dtype=torch.float32)

                # HIRES V2 (2026-07-23): every window of a shot slices from ONE
                # per-shot noise field per noising event, indexed by global
                # latent position, instead of drawing independent per-window
                # noise. Independent noise was the structural failure of v1:
                # each window synthesized a different texture identity and the
                # cross-fade blended two textures instead of reconciling one
                # (measured detail-correlation collapse at exactly the window
                # stride). Fields are created lazily at the first window (latent
                # shapes are known only after the first VAE encode), seeded
                # per-shot, kept on CPU (~45MB each) and sliced to GPU per window.
                shot_fields = None

                for wi, s in enumerate(starts):
                    if _anneal:
                        # encode the window at BASE res (sharp, on-manifold
                        # pixels - and a smaller encode than the target-res one
                        # the bicubic path pays), then upsample in LATENT space.
                        pil = [_PILImage.fromarray(f.numpy())
                               for f in vf[s:s + WIN]]
                        fv = frames_to_video_tensor(pil, base_h, base_w).unsqueeze(0).to(
                            device=device, dtype=_dt)
                        del pil
                        lat = video_vae.encode(fv).to(_dt)   # [1,C,F,h,w] normalized
                        del fv
                        _stats = video_vae.decoder.per_channel_statistics
                        raw = _stats.un_normalize(lat.to(torch.float32))
                        del lat
                        # temporal reflect pad: the x1.5-1.0 model corrupts
                        # sequence ends; mirrors absorb it (same trick as the
                        # spatial pass and the padded upsampler)
                        _f = int(raw.shape[2])
                        _p = min(8, _f - 1)
                        if _p > 0:
                            raw = torch.cat([raw[:, :, 1:_p + 1].flip(2), raw,
                                             raw[:, :, -(_p + 1):-1].flip(2)], dim=2)
                        with torch.no_grad():
                            up = _up_model(raw.to(dtype=_dt))
                        del raw
                        if _p > 0:
                            up = up[:, :, _p:-_p]
                        lat = _stats.normalize(up.to(torch.float32)).to(_dt)
                        del up
                        lat = lat.permute(0, 2, 1, 3, 4).contiguous()  # [1,Fl,C,H,W]
                        _empty_cache()
                    else:
                        win = vf[s:s + WIN].float().div_(255.0)  # uint8 -> float 0..1
                        x = win.permute(0, 3, 1, 2)            # [33,3,H,W]
                        x = torch.nn.functional.interpolate(
                            x, size=(th, tw), mode="bicubic", antialias=True).clamp(0, 1)
                        pil = [_PILImage.fromarray(
                            (f.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy())
                            for f in x]
                        del x
                        fv = frames_to_video_tensor(pil, th, tw).unsqueeze(0).to(
                            device=device, dtype=_dt)          # [1,3,65,th,tw] in [-1,1]
                        del pil
                        lat = video_vae.encode(fv).permute(0, 2, 1, 3, 4).to(_dt)  # [1,Fl,C,h,w]
                        del fv
                        _empty_cache()

                    Fl = int(lat.shape[1])
                    a0 = int(round(Fa * s / float(T)))
                    a1 = max(a0 + 2, int(round(Fa * (s + WIN) / float(T))))
                    alat = alat_full[:, a0:min(a1, Fa)].to(device=device, dtype=_dt)
                    Fa_w = int(alat.shape[1])

                    if shot_fields is None:
                        n_events = len(sigmas) - 1  # initial noising + each re-corruption
                        NG = T // 8 + Fl + 2        # global latent-frame budget with margin
                        # The refine noise was seeded from a CONSTANT (1234) for
                        # months - every render's detail layer shared identical
                        # noise regardless of the seed widget (audit, 2026-07-29).
                        # base_seed routes the user's seed in; the +1234 offset
                        # keeps refine noise decorrelated from the base pass.
                        gs = torch.Generator(device="cpu").manual_seed(
                            int(base_seed) + 1234 + si * 100)
                        shot_fields = {
                            "v": [torch.randn((1, NG) + tuple(lat.shape[2:]), generator=gs)
                                  for _ in range(n_events)],
                            "a": [torch.randn((1, Fa) + tuple(alat.shape[2:]), generator=gs)
                                  for _ in range(n_events)],
                        }
                    # window latent j -> global latent ls+j; window starts are
                    # multiples of (WIN-OVL)=24 pixel frames = 3 latent frames,
                    # so overlapping windows land on the same global indices.
                    ls = min(int(round(s / 8.0)), shot_fields["v"][0].shape[1] - Fl)
                    nv = shot_fields["v"][0][:, ls:ls + Fl].to(device=device, dtype=_dt)
                    na = shot_fields["a"][0][:, a0:a0 + Fa_w].to(device=device, dtype=_dt)
                    s0 = torch.full((1, Fl), sigmas[0])
                    sa0 = torch.full((1, Fa_w), sigmas[0])
                    v = add_noise(lat.flatten(0, 1), nv.flatten(0, 1),
                                  s0.flatten(0, 1)).unflatten(0, (1, Fl))
                    a = add_noise(alat, na, sa0)
                    del lat, nv, na

                    for i_s, sig in enumerate(sigmas[:-1]):
                        vs = sig * torch.ones((1, Fl), device=device)
                        as_ = sig * torch.ones((1, Fa_w), device=device)
                        pred_v, pred_a = generator(
                            noisy_image_or_video=v, conditional_dict=cond,
                            timestep=vs, noisy_audio=a, audio_timestep=as_)
                        nxt = sigmas[i_s + 1]
                        if nxt > 0:
                            # re-corruption noise from the step's shared field,
                            # same global slice - never randn_like (that was
                            # fresh unseeded noise per window per step in v1)
                            fnv = shot_fields["v"][i_s + 1][:, ls:ls + Fl].to(
                                device=device, dtype=_dt)
                            fna = shot_fields["a"][i_s + 1][:, a0:a0 + Fa_w].to(
                                device=device, dtype=_dt)
                            nvs = nxt * torch.ones((1, Fl), device=device)
                            nas = nxt * torch.ones((1, Fa_w), device=device)
                            v = add_noise(pred_v.flatten(0, 1), fnv.flatten(0, 1),
                                          nvs.flatten(0, 1)).unflatten(0, (1, Fl))
                            a = add_noise(pred_a, fna, nas)
                        else:
                            v = pred_v
                            a = pred_a
                    del a

                    u8w, _ = decode_benchmark_sample(video_vae, audio_vae, v, None,
                                                     video_tiling_config=tiling)
                    del v
                    _empty_cache()

                    w = torch.ones((WIN, 1, 1, 1), dtype=torch.float32)
                    if s > 0:
                        w[:OVL, 0, 0, 0] = torch.linspace(0.0, 1.0, OVL)
                    out[s:s + WIN] += (u8w.float() / 255.0) * w
                    wsum[s:s + WIN] += w
                    del u8w

                refined = (out / wsum.clamp_min(1e-6)).mul_(255.0).round_().clamp_(0, 255).to(torch.uint8)
                frames_list[si] = refined
                del out, wsum, cond
                _empty_cache()
                # Persist the refined shot IMMEDIATELY, same crash-safety
                # contract as the base per-shot saves: a crash later in the
                # refine (or in final assembly) costs at most one shot's
                # refine, never the pass. 2026-07-19: the box hard-crashed at
                # end-of-refine and all refined frames (RAM-only) were lost.
                if out_prefix:
                    try:
                        _wav = (waveforms[si]
                                if waveforms is not None and si < len(waveforms) else None)
                        self._save_shot_video(refined, _wav, si, fps, audio_sr, out_prefix)
                    except Exception as _e:
                        print(f"[JoyEcho] Hires: per-shot save failed for shot "
                              f"{si+1}: {_e}", flush=True)
                print(f"[JoyEcho] Hires: shot {si+1}/{len(frames_list)} refined "
                      f"({len(starts)} windows).", flush=True)
        finally:
            (generator.video_height, generator.video_width,
             generator.latent_height, generator.latent_width,
             generator.video_frame_seqlen) = _res_saved
            if offl is not None:
                offl.remove()
            if _up_model is not None:
                _up_model.cpu()
            _move(generator, "cpu")
            _move(video_vae.encoder, "cpu")
            _move(video_vae.decoder, "cpu")
            _empty_cache()
        print(f"[JoyEcho] Hires refine pass done in {_time.time()-_t0:.0f}s.", flush=True)

    def _temporal_upsample_decode(self, video_vae, video_latent, tiling_config, device):
        """Decode a shot's video latents through the LTX temporal x2 upsampler.

        Mirrors the hires-spatial pass: pipeline latents [1,F,C,h,w] ->
        un-normalize -> reflect-pad T (the upsampler family corrupts sequence
        ends; pads absorb it) -> upsample -> crop pad -> re-normalize -> tiled
        decode. Deterministic, video-only - the audio lane is never touched.
        Returns uint8 frames with F_out ~= 2*F-1. Decoder must already be on
        `device` (call inside Phase B).
        """
        import json as _json
        import folder_paths
        import comfy.utils as _cutils
        from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
        from ltx_core.model.video_vae import TemporalTilingConfig, TilingConfig
        from ltx_distillation.utils import decode_benchmark_sample

        # The doubled sequence (~2x frames) MUST decode temporally tiled no
        # matter what the shot's own tiling resolved to - an untiled 449-frame
        # decode is exactly the uncatchable native cuDNN abort the tiled-decode
        # work fixed (killed the whole process on first try, 2026-07-30).
        tiling_config = TilingConfig(
            spatial_config=None,
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64,
                                                 tile_overlap_in_frames=24))

        _dt = torch.bfloat16
        cls = type(self)
        if getattr(cls, "_tu_model", None) is None:
            path = folder_paths.get_full_path_or_raise(
                "latent_upscale_models",
                "ltx-2.3-temporal-upscaler-x2-1.0.safetensors")
            sd, metadata = _cutils.load_torch_file(
                path, safe_load=True, return_metadata=True)
            m = LatentUpsampler.from_config(
                _json.loads(metadata["config"])).to(dtype=_dt)
            m.load_state_dict(sd)
            m.eval()
            cls._tu_model = m
            del sd
        up_model = cls._tu_model.to(device)

        stats = video_vae.decoder.per_channel_statistics  # on device with decoder
        raw = video_latent.permute(0, 2, 1, 3, 4)          # [1,C,F,h,w]
        raw = stats.un_normalize(raw.to(device=device, dtype=torch.float32))
        f_in = raw.shape[2]
        p = min(8, f_in - 1)
        if p > 0:
            raw = torch.cat([raw[:, :, 1:p + 1].flip(2), raw,
                             raw[:, :, -(p + 1):-1].flip(2)], dim=2)
        with torch.no_grad():
            up = up_model(raw.to(dtype=_dt))
        _scale = up.shape[2] / raw.shape[2]
        del raw
        if p > 0:
            _po = int(round(p * _scale))
            up = up[:, :, _po:up.shape[2] - _po]
        up = stats.normalize(up.to(torch.float32)).to(_dt)
        up = up.permute(0, 2, 1, 3, 4).contiguous()        # [1,F2,C,h,w]
        u8, _ = decode_benchmark_sample(video_vae, None, up.to(device), None,
                                        video_tiling_config=tiling_config)
        del up
        up_model.to("cpu")
        return u8

    @staticmethod
    def _save_shot_video(video_uint8, audio_waveform, shot_idx, fps, audio_sr, prefix):
        """Save a single shot as mp4 immediately after generation."""
        import av
        import numpy as np

        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except Exception:
            output_dir = Path("/root/ComfyUI/output")

        # Build output path
        parts = prefix.rsplit("/", 1)
        if len(parts) == 2:
            sub_dir = Path(output_dir) / parts[0]
            name_prefix = parts[1]
        else:
            sub_dir = Path(output_dir)
            name_prefix = prefix

        sub_dir.mkdir(parents=True, exist_ok=True)
        out_path = sub_dir / f"{name_prefix}_{shot_idx:03d}.mp4"

        frames_np = video_uint8.cpu().numpy() if isinstance(video_uint8, torch.Tensor) else video_uint8

        container = av.open(str(out_path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.height = frames_np.shape[1]
        stream.width = frames_np.shape[2]
        stream.pix_fmt = "yuv420p"
        # crf18/fast produced visible per-frame quality pumping on the
        # grain-heavy analog-horror content (2026-07-19, output_00153):
        # preset fast enables b-pyramid, so the GOP alternates
        # well-fed P / reference-B frames with starved outer B-frames -
        # sharp/soft flicker with a period of 2, measured at 14/14
        # sign-flips in per-frame Laplacian variance. bf=0 removes
        # B-frames entirely (the only full cure), tune=grain keeps the
        # noise field stable across frames, crf 16 + medium feeds it.
        # These per-shot files are the MASTERS (finals should be
        # stream-copy concats of them - SaveVideo exposes no quality
        # knobs); the ~40% size increase is the cost of no pumping.
        stream.options = {"crf": "16", "preset": "medium",
                          "tune": "grain", "bf": "0"}

        for frame_data in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # Save audio sidecar. DELIBERATE ORDER: soundfile FIRST, torchaudio never.
        # torchaudio.save on torch 2.8+ dispatches to torchcodec, which dlopens
        # the system FFmpeg shared libs (libavutil.so.60 = FFmpeg 7/8). On a box
        # whose FFmpeg is older (Ubuntu 24.04 ships 6.x = libavutil.so.58) that
        # dlopen does not always raise a catchable Python error - it can trip the
        # glibc loader assertion "ld.so: Assertion `listp != NULL' failed!", which
        # ABORTS the process. A try/except cannot save you from that: ComfyUI just
        # dies, mid-run, after shots have already rendered (reported 2026-07-24).
        # soundfile bundles libsndfile and never touches FFmpeg, so it cannot trip
        # it; the builtin wave module has no deps at all. Neither needs torchaudio,
        # which buys nothing for a plain PCM WAV write. Do not "restore" torchaudio
        # to the front of this chain.
        if audio_waveform is not None:
            wav_path = sub_dir / f"{name_prefix}_{shot_idx:03d}.wav"
            waveform = audio_waveform.cpu()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            _saved = False
            try:
                import soundfile as _sf
                _sf.write(str(wav_path), waveform.transpose(0, 1).numpy(),
                          int(audio_sr))
                _saved = True
            except Exception as _e_sf:
                import wave as _wave
                pcm = (waveform.clamp(-1.0, 1.0) * 32767.0).round().to(
                    torch.int16).transpose(0, 1).contiguous().numpy()
                with _wave.open(str(wav_path), "wb") as _w:
                    _w.setnchannels(int(waveform.shape[0]))
                    _w.setsampwidth(2)
                    _w.setframerate(int(audio_sr))
                    _w.writeframes(pcm.tobytes())
                _saved = True
                print(f"[JoyEcho] wav via builtin wave, PCM16 (soundfile "
                      f"unavailable: {type(_e_sf).__name__}). "
                      f"pip install soundfile for float-wav output.",
                      flush=True)

        print(f"[JoyEcho] Shot {shot_idx} saved → {out_path}", flush=True)


class JoyEcho_SingleShotGenerate:
    """Generate a single shot with memory bank input/output for chaining.

    Each instance has its own editable prompt text box and outputs video frames
    that can be previewed immediately via CreateVideo → SaveVideo.
    Chain multiple instances via the memory output → next shot's memory input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JOYECHO_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Single shot prompt text",
                }),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 2**31 - 1}),
                "num_frames": ("INT", {"default": 241, "min": 9, "max": 1441, "step": 8,
                                       "tooltip": "Must be 1 + 8*k (e.g. 121, 241, 361)"}),
                # Rebels local patch: portrait resolutions. Height was capped at
                # 1088 while width allowed 1920, which silently forbade portrait
                # (e.g. 1088x1920 to match a portrait Z-Image first frame). Both
                # axes now cap at 1920; step 32 keeps the latent packing valid.
                "video_height": ("INT", {"default": 736, "min": 256, "max": 1920, "step": 32}),
                "video_width": ("INT", {"default": 1280, "min": 256, "max": 1920, "step": 32}),
            },
            "optional": {
                "memory": ("JOYECHO_MEMORY",),
                "video_fps": ("INT", {"default": 24, "min": 1, "max": 60, "tooltip": "KEEP AT 24. The LTX joint audio-video prior is 24fps-native: any other value (25 included) systematically drifts spoken voices toward Commonwealth accents (British/Australian) and overrides accent wording in the prompt. Verified A/B 2026-07-29."}),
                "v2a_grad_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "memory_max_size": ("INT", {"default": 7, "min": 0, "max": 20}),
                "num_fix_frames": ("INT", {"default": 3, "min": 0, "max": 10}),
                "enable_audio_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Feed previous shots' audio latents as cross-shot conditioning. "
                               "ON keeps the same voice across every shot - the point of "
                               "multishot - at a small lip-sync cost on long dialogue. OFF "
                               "gives the tightest sync but the voice can drift between "
                               "shots; use only for sync-critical single-voice pieces with "
                               "a strong voice description repeated in every shot.",
                }),
                "audio_memory_window_size": ("INT", {"default": 96, "min": 16, "max": 256}),
                "sequential_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable layer-by-layer GPU offloading for DiT.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "JOYECHO_MEMORY", "JOYECHO_MODEL",)
    RETURN_NAMES = ("images", "audio", "memory", "model",)
    FUNCTION = "generate_shot"
    CATEGORY = "JoyAI-Echo"

    def generate_shot(
        self,
        model: dict,
        prompt: str,
        seed: int = 12345,
        num_frames: int = 241,
        video_height: int = 736,
        video_width: int = 1280,
        memory: dict | None = None,
        video_fps: int = 25,
        v2a_grad_scale: float = 2.0,
        memory_max_size: int = 7,
        num_fix_frames: int = 3,
        enable_audio_memory: bool = True,
        audio_memory_window_size: int = 96,
        sequential_offload: bool = False,
    ):
        from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
        from ltx_distillation.inference.memory_bidirectional_pipeline import BidirectionalMemoryAVInferencePipeline
        from ltx_distillation.inference.memory_multishot import (
            PairedAudioVideoMemoryBank,
            build_paired_audio_memory_kwargs,
            video_uint8_to_pil_frames,
        )
        from ltx_distillation.utils import (
            add_noise,
            compute_latent_shapes,
            decode_benchmark_sample,
            encode_memory_frames_batch,
        )

        if not prompt.strip():
            raise ValueError("Prompt is empty. Enter a shot description.")

        text_encoder = model.get("text_encoder")
        if text_encoder is None and callable(model.get("text_encoder_builder")):
            text_encoder = model["text_encoder_builder"]()
            model["text_encoder"] = text_encoder
        if text_encoder is None:
            raise RuntimeError(
                "Text encoder not available. It may have been released by a previous shot. "
                "Set release_text_encoder=False on earlier shots."
            )

        generator = model["generator"]
        video_vae = model["video_vae"]
        audio_vae = model["audio_vae"]
        audio_sample_rate = model["audio_sample_rate"]
        device = model["device"]
        dtype = model["dtype"]

        # Validate num_frames
        if (num_frames - 1) % 8 != 0:
            num_frames = 1 + ((num_frames - 1) // 8) * 8

        # Update generator resolution
        generator.video_height = video_height
        generator.video_width = video_width
        generator.latent_height = video_height // 32
        generator.latent_width = video_width // 32
        generator.video_frame_seqlen = generator.latent_height * generator.latent_width
        # render fps drives the video RoPE clock (see multishot Generate note)
        generator.VIDEO_FPS = float(video_fps)
        if int(video_fps) != 24:
            print(f"[JoyEcho] WARNING: video_fps={video_fps}. The joint AV prior is "
                  f"24fps-native - non-24 fps drifts voices toward Commonwealth "
                  f"accents and overrides accent wording (verified 2026-07-29). "
                  f"Use 24 unless you specifically want that.", flush=True)

        # Compute latent shapes
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=num_frames,
            video_height=video_height,
            video_width=video_width,
            batch_size=1,
            video_fps=float(video_fps),
        )

        # Get or create memory bank.
        # CLONE the incoming bank, never adopt it (audit CRITICAL, 2026-07-29):
        # ComfyUI caches node outputs, so the upstream node hands this node the
        # SAME live bank object on every queue run. Mutating it in place made
        # each run condition on the PREVIOUS run's rendered frames and audio -
        # a literal identity/voice carryover between unrelated queue presses.
        # A shallow instance copy with a fresh slot list breaks the aliasing;
        # the entries themselves are immutable after save and safe to share.
        if memory is not None:
            import copy as _copy
            _src_bank = memory["bank"]
            memory_bank = _copy.copy(_src_bank)
            memory_bank.memory = list(_src_bank.memory)
        else:
            memory_bank = PairedAudioVideoMemoryBank(
                max_size=memory_max_size,
                save_mode="random_every_shot_frame",
                num_fix_frames=num_fix_frames,
            )

        print(f"[JoyEcho] SingleShot: encoding prompt, seed={seed}, "
              f"memory_size={len(memory_bank)}", flush=True)

        # --- Phase 0: Encode (text encoder on GPU, everything else off) ---
        _move(generator, "cpu")
        _move(video_vae.encoder, "cpu")
        _move(video_vae.decoder, "cpu")
        _move(audio_vae.encoder, "cpu")
        _move(audio_vae.decoder, "cpu")
        _move(audio_vae.vocoder, "cpu")
        _move(text_encoder, device)
        _empty_cache()

        cond = text_encoder([prompt.strip()])
        conditional_dict = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in cond.items()
        }
        del cond

        # Offload text encoder immediately after encoding
        _move(text_encoder, "cpu")
        _empty_cache()

        # Build pipelines
        denoising_sigmas = torch.tensor(DENOISING_SIGMAS, device=device, dtype=torch.float32)
        base_pipeline = BidirectionalAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
        )
        memory_pipeline = BidirectionalMemoryAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
            memory_downscale_factor=1,
        )

        offloader = None
        if sequential_offload:
            offloader = SequentialOffloader(generator, device)

        # --- Phase A: Denoise (generator on GPU, everything else off) ---
        if sequential_offload:
            offloader.install()
        else:
            _move(generator, device)
        _empty_cache()

        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)

            if len(memory_bank) > 0:
                _move(video_vae.encoder, device)
                memory_video = encode_memory_frames_batch(
                    video_vae=video_vae,
                    batch_memory_frames=[memory_bank.get_memory_frames()],
                    target_h=video_height,
                    target_w=video_width,
                    device=device,
                    dtype=dtype,
                )
                _move(video_vae.encoder, "cpu")
                _empty_cache()

                memory_audio_kwargs = build_paired_audio_memory_kwargs(
                    memory_bank,
                    enable_audio_memory=enable_audio_memory,
                    v2a_grad_scale=v2a_grad_scale,
                    memory_position_mode="reference",
                )

                video_latent, audio_latent = memory_pipeline.generate(
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    conditional_dict=conditional_dict,
                    memory_video=memory_video,
                    seed=seed,
                    **memory_audio_kwargs,
                )
                del memory_video
            else:
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

        # Storage not gated on enable_audio_memory (same memory-bank fix as the
        # multishot node above): the flag gates INJECTION only.
        audio_memory_latent = (
            audio_latent.detach().cpu().contiguous()
            if audio_latent is not None
            else None
        )

        # --- Phase B: Decode ---
        if sequential_offload:
            offloader.remove()
        _move(generator, "cpu")
        _empty_cache()
        _move(video_vae.decoder, device)
        _move(audio_vae.decoder, device)
        _move(audio_vae.vocoder, device)

        video_uint8, audio_waveform = decode_benchmark_sample(
            video_vae, audio_vae, video_latent, audio_latent
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        _move(video_vae.decoder, "cpu")
        _move(audio_vae.decoder, "cpu")
        _move(audio_vae.vocoder, "cpu")
        _empty_cache()

        # Update memory bank
        memory_frames_pil = video_uint8_to_pil_frames(video_uint8)
        if audio_memory_latent is not None:
            memory_bank.save_memory_slot(
                memory_frames_pil,
                audio_memory_latent,
                audio_window_size=audio_memory_window_size,
                video_clip_num_frames=9,
                audio_waveform=audio_waveform,
                audio_sample_rate=16000,
                video_fps=float(video_fps),
                audio_window_selection_mode="max_response",
                video_frame_selection_mode="center",
                audio_memory_mel_bins=128,
                audio_memory_mel_hop_length=160,
                audio_memory_n_fft=1024,
                audio_memory_downsample_factor=4,
                audio_memory_is_causal=True,
            )

        # Build outputs
        images = video_uint8.float() / 255.0  # [F, H, W, 3]

        audio_out = None
        if audio_waveform is not None:
            from ltx_distillation.inference.memory_multishot import normalize_audio_waveform_for_media
            audio_norm = normalize_audio_waveform_for_media(audio_waveform)
            audio_out = {
                "waveform": audio_norm.unsqueeze(0),  # [1, C, samples]
                "sample_rate": audio_sample_rate,
            }

        memory_out = {"bank": memory_bank}

        del video_latent, audio_latent, audio_memory_latent, video_uint8, audio_waveform
        _empty_cache()

        print(f"[JoyEcho] SingleShot done. {images.shape[0]} frames.", flush=True)

        return (images, audio_out, memory_out, model,)


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_DEFAULT_LONG_STORY_SYSTEM_PROMPT = ""
_long_sp_path = _PROMPTS_DIR / "long_story_writer_system_prompt.md"
if _long_sp_path.exists():
    _DEFAULT_LONG_STORY_SYSTEM_PROMPT = _long_sp_path.read_text(encoding="utf-8").strip()


def _load_system_prompt(mode: str, engine: str = "auto") -> str:
    """Load the full system prompt from the bundled markdown file.

    ENGINE matters because the two video models disagree about one thing that
    silently ruins a take: MiniMax-H3 reads double quotation marks as text
    PRINTED IN THE SHOT and takes spoken words only inside <d> tags, while the
    LTX/AutoTTS path extracts dialogue FROM double quotes. One shared prompt
    cannot be right for both - written for LTX it makes H3 paint the line on a
    wall and invent mumbling for the audio it was never given.

    An engine-specific sibling is used when present: engine="h3" prefers
    long_story_writer_system_prompt.h3.md. "auto" keeps the previous
    behaviour exactly, so existing graphs are unaffected.
    """
    stem = ("long_story_writer_system_prompt" if "long" in mode
            else "short_story_writer_system_prompt")
    if engine and str(engine).lower() == "h3":
        h3 = _PROMPTS_DIR / f"{stem}.h3.md"
        if h3.exists():
            return h3.read_text(encoding="utf-8").strip()
        logging.warning("[JoyEcho] engine=h3 but %s is missing; falling back "
                        "to the shared prompt, which marks speech with quotes "
                        "- H3 will render those as on-screen text", h3.name)
    fp = _PROMPTS_DIR / f"{stem}.md"
    if fp.exists():
        return fp.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"System prompt not found: {fp}")


# Boundary discipline for chained (context_pin / Motion-Context) rendering.
# ONE authoritative copy per machine: prompts/h3_boundary_rules_continuous.md
# in this pack. The doctrine validator on G: reads the SAME file - do not
# fork the text into other tools (that is how doctrine copies drift).
_BOUNDARY_RULES_PATH = _PROMPTS_DIR / "h3_boundary_rules_continuous.md"
# Extend take: one continuous speech across pinned windows - the OPPOSITE
# discipline (no airlock, no settle, speech runs through the boundary).
_EXTEND_RULES_PATH = _PROMPTS_DIR / "h3_extend_rules.md"
_REFS_RULES_PATH = _PROMPTS_DIR / "h3_refs_attached_rules.md"

# Only for the strict "continuous take" style: no cuts exist at all.
_TAKE_ONLY_RULE = (
    "6. CONTINUOUS TAKE: there are no camera cuts anywhere in the scene. "
    "One [Shot 1] per block, no [Shot 2], no cut verbs, no <scenetrans>, "
    "and the same camera position and framing stated identically in every "
    "block.")


def _is_extend(join_style) -> bool:
    return bool(join_style) and str(join_style).startswith("extend")


def _load_boundary_rules(join_style: str) -> str:
    """The system-prompt addendum for a chained join style ('' for cuts)."""
    if not join_style or join_style.startswith("cuts"):
        return ""
    if _is_extend(join_style):
        if not _EXTEND_RULES_PATH.exists():
            raise FileNotFoundError(
                f"extend rules file missing: {_EXTEND_RULES_PATH} - "
                "reinstall the pack.")
        return _EXTEND_RULES_PATH.read_text(encoding="utf-8").strip()
    if not _BOUNDARY_RULES_PATH.exists():
        raise FileNotFoundError(
            f"boundary rules file missing: {_BOUNDARY_RULES_PATH} - "
            "reinstall the pack; the rules are not embedded anywhere else "
            "on purpose (single source of truth).")
    rules = _BOUNDARY_RULES_PATH.read_text(encoding="utf-8").strip()
    if "take" in join_style:
        rules += "\n" + _TAKE_ONLY_RULE
    return rules


def _load_refs_rules() -> str:
    """System-prompt addendum used when reference photographs ride along.

    Render-verified 2026-08-17 (same seed): the writer's identity sentence
    beats attached reference photographs - the model rendered the prose's
    person, not the photographs'. Pointing the identity sentence at the
    photographs instead rendered the referenced person. Single source of
    truth in prompts/h3_refs_attached_rules.md.
    """
    if not _REFS_RULES_PATH.exists():
        raise FileNotFoundError(
            f"refs rules file missing: {_REFS_RULES_PATH} - reinstall the pack.")
    return _REFS_RULES_PATH.read_text(encoding="utf-8").strip()


_ID_SENTENCE = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9']*(?:_[A-Z])?)"
    r"|(?:(?:[Tt]he|[Aa])\s+(?:[a-z][a-z0-9-]*\s+){0,2}"
    r"(?:woman|man|girl|boy|person|figure)))"
    r" is (?:an? |the )?"
    r"[^.\n]*?\b(?:year|years|-old|hair|face|build|eyes|eyed|skin|"
    r"woman|man|girl|boy|teen\w*|adult|elderly|complexion)"
    r"[^.\n]*\.")


# --- local-Ollama context window -------------------------------------------------
# Ollama ignores a model's TRAINED context and uses num_ctx from the Modelfile, which is unset
# on most pulls - so it silently falls back to 4096. This system prompt alone is ~5.5k tokens,
# so on a local endpoint the RULES WERE BEING TRUNCATED AWAY before the model ever reached the
# premise. That is the real cause of the empty completions, the JSON dying mid-array, the
# ignored style bans and the blown dialogue budgets. Measured 2026-08-21: qwen3.8 reports a
# trained context of 262144 with num_ctx unset in its Modelfile.
#
# Ollama passes an "options" object through its OpenAI-compatible endpoint to the runner, so
# the window is requested per call. Cloud endpoints and non-Ollama servers ignore an unknown
# top-level field, so this is safe to send everywhere.
_OLLAMA_NUM_CTX = int(os.environ.get("JOYECHO_NUM_CTX", "32768"))


def _is_local_ollama(base_url: str) -> bool:
    u = (base_url or "").lower()
    return (("localhost" in u or "127.0.0.1" in u or ":11434" in u or "://ollama" in u)
            and "cloud" not in u)


def _ollama_ctx(base_url: str) -> dict:
    """{"options": {"num_ctx": N}} for a local Ollama endpoint, {} otherwise."""
    if not _is_local_ollama(base_url):
        return {}
    return {"options": {"num_ctx": _OLLAMA_NUM_CTX}}


_REFS_STOPWORDS = frozenset(
    "a an the it he she they we you i at on in of and or but then when after over under with "
    "ends end camera shot scene night day morning evening cut cuts".split())

_AGE_PHRASE = None


def _point_identity_at_refs(prompts, premise=""):
    """refs_attached: make the identity sentence deterministic - but only for
    characters the PREMISE names.

    Render-verified 2026-08-17 (same seed): a written identity description
    beats attached reference photographs, so named characters' descriptions
    are replaced with a pointer at the photographs. Two later lessons folded
    in (2026-08-19/20):
      - photographs do not carry AGE - if the replaced sentence stated one,
        the age phrase is kept in the pointer sentence;
      - photographs only exist for characters the premise itself names. A
        name the LLM invented (the premise said "a news anchor", the writer
        called her "Diane") has no photo folder, and pointing her sentence at
        photographs that do not exist erased her real description and gave
        the render model nothing. Those sentences are now left as written.
    Returns (new_prompts, replaced_count, kept_count)."""
    import re as _re
    global _AGE_PHRASE
    if _AGE_PHRASE is None:
        _AGE_PHRASE = _re.compile(
            r"\b((?:[a-z]+-|\d{1,2}-)year-old(?:\s+(?:girl|boy|woman|man))?"
            r"|in (?:her|his|their) (?:teens|twenties|thirties|forties|fifties|sixties|seventies))\b",
            _re.I)
    allowed = set()
    for w in _re.findall(r"\b[A-Z][A-Za-z']+\b", premise or ""):
        if w.lower() not in _REFS_STOPWORDS:
            allowed.add(w.lower())
    out, n, kept = [], 0, 0
    for ptxt in prompts:
        def _rep(m):
            nonlocal n, kept
            if allowed and m.group(1).lower() not in allowed:
                kept += 1
                return m.group(0)          # invented name: no photos exist, keep the description
            n += 1
            age = _AGE_PHRASE.search(m.group(0))
            if age:
                a = age.group(1)
                if not a.lower().startswith('in '):
                    a = 'a ' + a          # "sixteen-year-old girl" -> "a sixteen-year-old girl"
                return ("%s, %s, looks exactly as in the reference photographs "
                        "- same face, hair and clothing." % (m.group(1), a))
            return ("%s looks exactly as in the reference photographs - same "
                    "face, hair and clothing." % m.group(1))
        out.append(_ID_SENTENCE.sub(_rep, str(ptxt)))
    return out, n, kept


def _spoken_words(shot_text):
    """Words actually spoken: <d>...</d> first, quoted speech second."""
    import re as _re
    d = _re.findall(r"<d>(?:\[[^\]]*\])?(.*?)</d>", shot_text, _re.S)
    if not d:
        d = _re.findall(r'"([^"]{4,})"', shot_text)
    return sum(len(x.split()) for x in d)

def _budget_warn(prompts, blo, bhi, slack=1.25):
    """Report shots whose spoken lines overrun the budget they were given.

    Only flags above bhi*slack - a shot at 21 words against a 20-word
    ceiling is not what this exists to catch; one at 45 is.
    """
    bad, speaking = [], 0
    for i, p in enumerate(prompts, 1):
        n = _spoken_words(p if isinstance(p, str) else str(p))
        if n:
            speaking += 1
        if n and n > bhi * slack:
            bad.append((i, n))
    # The check was one-sided: it only ever caught lines that were too
    # LONG, and skipped a shot with zero words as "silent". A script
    # with no dialogue at all therefore passed in silence - which is the
    # worse failure, because a crammed line is audible and a slideshow
    # is not discovered until the render is watched.
    if prompts and speaking == 0:
        print("[JoyEcho] WARNING: NO dialogue in any of the %d shot(s). "
              "This will render as a silent slideshow. If that is not "
              "deliberate, the writer has opted out of speech - say so "
              "in the brief, or check the shot count is not so high "
              "that it padded with wordless beats."
              % len(prompts), flush=True)
    elif prompts and speaking < len(prompts) / 3.0:
        print("[JoyEcho] WARNING: only %d of %d shot(s) contain "
              "dialogue. A long piece carried by that little speech "
              "plays as a slideshow."
              % (speaking, len(prompts)), flush=True)
    # A SILENT shot whose text still talks about lip movement / lip sync is
    # an under-specified mouth: with a voice anchor in the chain the model
    # played an EARLIER shot's line back into it (render-verified 2026-08-17,
    # shot 3 of a 4-shot take re-spoke shot 1's line word for word). Flag it
    # by name; the fix is writer-side (few-shot framing sentence).
    import re as _re
    for i, p in enumerate(prompts, 1):
        s = p if isinstance(p, str) else str(p)
        if not _spoken_words(s) and _re.search(r"lip[- ]?(movement|sync)", s, _re.I):
            print("[JoyEcho] WARNING shot %d is SILENT but its text mentions "
                  "lip movement/sync - an unassigned mouth invents speech, and "
                  "in a chained take with a voice anchor it can re-speak an "
                  "earlier line. Write the mouth positively (\"lips stay "
                  "pressed shut\") and keep the framing about the face only."
                  % i, flush=True)
    for i, n in bad:
        print("[JoyEcho] WARNING shot %d: %d spoken words against a "
              "%d-%d budget for this clip length. Speech that overruns "
              "the clip renders crammed or garbled - shorten the line, "
              "or raise frames_per_shot." % (i, n, blo, bhi), flush=True)
    if bad:
        print("[JoyEcho] %d of %d shot(s) overrun the dialogue budget."
              % (len(bad), len(prompts)), flush=True)
    return bad


def _speakable_budget(num_frames, fps, chained):
    """One clip length -> one word budget, shared by every writer path.

    Returns (secs, speakable, lo, hi) or None when no duration was supplied.

    Chained styles spend ~1s of each later block on the discarded replay and
    ~4s on the airlock and settle, so the speakable span is shorter than the
    block. Sizing dialogue to the full block is the failure that chopped the
    line head (render-verified 2026-08-10).

    This exists because the main path and the revise path each had their own
    copy of this arithmetic and they had already diverged - revise used the
    raw clip length and so over-budgeted every line on a chained render.
    """
    if not num_frames or num_frames <= 0:
        return None
    _fps = fps if (fps and fps > 0) else 25.0
    secs = num_frames / _fps
    if chained == "extend":
        # extend take: speech runs through the boundary; only the discarded
        # ~1 s replay head is unspeakable (0.9 s at pin 22)
        speak = max(3.0, secs - 1.0)
    else:
        speak = max(3.0, secs - 5.0) if chained else secs
    return secs, speak, max(6, int(round(speak * 1.2))), int(round(speak * 2.0))


def _no_duration_warning(where):
    """num_frames=0 means the writer is guessing. Say so - it used to be mute."""
    print("[JoyEcho] %s: num_frames is 0, so NO clip duration was given to "
          "the writer. It will size dialogue for the ~10 second clip assumed "
          "in the system prompt, which is wrong for any other shot length and "
          "is the usual cause of crammed, garbled speech. Set num_frames on "
          "this node to the SAME value as the sampler's frames_per_shot."
          % where, flush=True)


class JoyEcho_PromptFormat:
    """Helper node providing the official prompt writing system prompts.

    Use this with any LLM node in ComfyUI to generate properly formatted
    shot prompts from a short story description.

    The output can be fed directly into JoyEcho_TextEncode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["long_story (multi-shot)", "short_story (single-shot)"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "get_prompt"
    CATEGORY = "JoyAI-Echo"

    def get_prompt(self, mode: str):
        return (_load_system_prompt(mode),)


def _extract_json_object(text: str):
    """Return the first top-level {...} JSON object substring in `text`, matching
    braces while ignoring any that sit inside a JSON string (the shot prompts
    themselves contain no raw braces, but escaped quotes / stray prose might).
    Returns None if no balanced object is found. Lets the enhancer survive a
    model that wraps its answer in preamble or trailing commentary."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


_LLME_MODELS_CACHE = {"t": 0.0, "list": None}
# Models referenced by saved graphs. ALWAYS offered, so a workflow still loads
# with its stored model selected when the endpoint is unreachable.
_LLME_KNOWN = ("minimax-m3:cloud", "glm-5.2:cloud", "deepseek-v4-pro:cloud",
               "gpt-4o", "deepseek-chat")
_LLME_CUSTOM = "(custom - use model_name_custom below)"


def _llme_model_list():
    """Enumerate installed models for the model_name dropdown.

    Rebels local patch 2026-08-02. model_name was free text, so every model
    change was a typo risk against a 21-model Ollama install.

    Deliberately defensive, because INPUT_TYPES runs on EVERY node-list refresh
    and on ComfyUI startup:
      - 1.5 s timeout: a dead endpoint must never hang the node list
      - 60 s cache: refreshes do not hammer the server
      - _LLME_KNOWN is always merged in, so a saved workflow's stored model is
        still a valid choice when discovery fails (a COMBO whose stored value
        is absent from the list fails validation and blocks the render)
      - the custom sentinel keeps non-Ollama endpoints usable

    base_url is a sibling widget and is NOT visible here, so discovery targets
    the local Ollama default; JOYECHO_LLM_URL overrides it.
    """
    import time as _t
    if _LLME_MODELS_CACHE["list"] and _t.time() - _LLME_MODELS_CACHE["t"] < 60:
        return _LLME_MODELS_CACHE["list"]
    found = []
    try:
        import urllib.request
        url = os.environ.get("JOYECHO_LLM_URL", "http://localhost:11434")
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=1.5) as r:
            found = [m["name"] for m in json.load(r).get("models", [])
                     if m.get("name")]
    except Exception:
        found = []
    merged = sorted(set(found) | set(_LLME_KNOWN)) + [_LLME_CUSTOM]
    _LLME_MODELS_CACHE.update({"t": _t.time(), "list": merged})
    return merged


def _llme_shots_in(story_idea):
    """Normalise ANY story_idea into (shots, was_json). Never raises on shape.

    Rebels local patch 2026-08-02. Every mode used to parse its input its own
    way, so what happened to a .txt vs a .json depended on which dropdown entry
    was selected, and two shapes raised instead of coping:
      - a JSON dict with no 'prompts'/'shots' array raised ValueError
      - a string starting with '{' that was not valid JSON fell through
        inconsistently between modes
    Both now degrade to "treat the raw text as one finished prompt", which is
    the only sane reading of a paragraph. One helper, one behaviour, four modes.

    Returns [] for empty input; callers decide whether that is fatal.
    """
    src = (story_idea or "").strip()
    if not src:
        return [], False
    if src.startswith("{"):
        try:
            data = json.loads(src)
            if isinstance(data, dict):
                arr = data.get("prompts") or data.get("shots")
                if isinstance(arr, list) and arr:
                    return [str(s) for s in arr], True
        except (json.JSONDecodeError, AttributeError):
            pass
        # valid-but-unusable JSON, or malformed JSON: it is still prose
        return [src], False
    # Plain text: honour --- separators, the SAME rule the sampler uses in
    # h3_multishot_utils._parse_script. example_script.txt ships --- separated
    # and every doc says to write scripts that way, but this helper returned
    # the whole file as ONE shot - so a pasted multi-shot script in passthrough
    # mode rendered the entire text as shot 1 and then repeated it to fill
    # shot_count. A single paragraph still returns as one shot, so LPFF .txt
    # batches are unaffected.
    import re as _re_split
    _blocks = [b.strip() for b in _re_split.split(r"(?m)^---\s*$", src)
               if b.strip()]
    if len(_blocks) > 1:
        return _blocks, False
    return [src], False


class JoyEcho_LLMEnhance:
    """Call a cloud LLM API to expand a short story idea into JoyAI-Echo shot prompts.

    Supports OpenAI-compatible APIs (OpenAI, DeepSeek, etc.).
    The output JSON can be fed directly into JoyEcho_TextEncode or split via JoyEcho_PromptAtIndex.
    Uses only cloud API calls — zero local GPU memory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_idea": ("STRING", {
                    "multiline": True,
                    "default": "A young woman records a quiet evening vlog in her cozy room, reflecting on life and finding warmth in small things.",
                    "tooltip": "Describe your story or scene idea in a few sentences.",
                }),
                "mode": (["long_story (multi-shot)", "short_story (single-shot)",
                          "passthrough (raw JSON, skip LLM)",
                          "revise (keep structure, rewrite prose)"],),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your provider's API key (OpenAI, DeepSeek, GLM, Gemini...). "
                               "LEAVE BLANK for a local endpoint (localhost / 192.168.x / .local) "
                               "- those ignore it and a placeholder is sent automatically. Also "
                               "not needed in passthrough mode, which skips the LLM entirely.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": _DEFAULT_LONG_STORY_SYSTEM_PROMPT,
                    "tooltip": "System prompt for the LLM. Edit to customize prompt generation style.",
                }),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": "https://api.openai.com/v1",
                    "tooltip": "API base URL. Use https://api.deepseek.com/v1 for DeepSeek, etc.",
                }),
                # COMBO, but it stays at THIS position: widgets_values is
                # positional, so changing the type in place is safe while
                # moving or inserting a widget would shift every saved value
                # after it.
                "model_name": (_llme_model_list(), {
                    "default": "gpt-4o",
                    "tooltip": "Which model writes the script. PREFER A HOSTED "
                               "OR CLOUD MODEL (an Ollama ':cloud' tag, or any "
                               "OpenAI-compatible endpoint): a local writer big "
                               "enough to be good is 15-25 GB, and loading it "
                               "on the render box right before the video model "
                               "evicts that model from RAM/VRAM and cost a 24 GB "
                               "card a 12-minute reload (measured). Local is "
                               "fine when the writer runs on another machine "
                               "or your card has room. Installed models are "
                               "read live from Ollama (JOYECHO_LLM_URL to point "
                               "elsewhere; 60s cache); models referenced by "
                               "saved graphs are always listed. For an endpoint "
                               "this cannot enumerate, pick '(custom ...)' and "
                               "type the name in model_name_custom.",
                }),
                "num_shots": ("INT", {
                    "default": 0, "min": 0, "max": 30,
                    "tooltip": "Number of shots to generate (0 = let LLM decide, default 15 for long story).",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "num_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Frames per shot (match JoyEcho_Generate). 0 = "
                               "disabled (assume ~10s clips). When set, the LLM is "
                               "told each shot's real duration and scales action + "
                               "dialogue length to fit.",
                }),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Playback fps used to turn num_frames into seconds "
                               "per shot (match your CreateVideo / Generate fps).",
                }),
                # APPENDED LAST on purpose: a new widget anywhere earlier would
                # shift every widgets_values slot after it in saved workflows.
                "model_name_custom": ("STRING", {
                    "default": "",
                    "tooltip": "Only used when model_name is set to the "
                               "'(custom ...)' entry. Lets a non-enumerable "
                               "endpoint (OpenAI, DeepSeek, an internal gateway) "
                               "still be driven by hand.",
                }),
                # NEW WIDGETS APPEND LAST - inserting above shifts every saved
                # workflow's values by one slot.
                "join_style": ([
                    "cuts (independent setups)",
                    "continuous take (chained, no cuts)",
                    "continuous scene with cuts (chained)",
                    "extend take (one continuous speech across windows)"], {
                    "default": "cuts (independent setups)",
                    "tooltip": "How consecutive prompts join at render time.\n"
                               "cuts = every prompt is an independent setup; "
                               "boundaries are hard cuts (classic multishot).\n"
                               "continuous take = one unbroken take rendered "
                               "as a chain (context_pin / Motion Context): "
                               "every block opens holding the previous "
                               "block's closing arrangement (2s airlock, no "
                               "dialogue), lands settled, no camera cuts "
                               "anywhere, scene block byte-identical.\n"
                               "continuous scene with cuts = camera cuts "
                               "allowed INSIDE a block, but every block "
                               "boundary is a continuity join and follows "
                               "the same airlock/settle rules. Rules live in "
                               "prompts/h3_boundary_rules_continuous.md "
                               "(single source of truth - validator reads "
                               "the same file).",
                }),
                # forceInput: a link, never a widget, so saved graphs' widget
                # values keep their slots. Wire the same boolean that switches
                # your reference photographs on.
                "refs_attached": ("BOOLEAN", {
                    "forceInput": True,
                    "tooltip": "Wire this from the switch that turns your "
                               "reference photographs on. When true the "
                               "writer stops describing faces, hair, age and "
                               "build and writes 'looks exactly as in the "
                               "reference photographs' instead - measured on "
                               "the same seed: a written identity sentence "
                               "overrides the photographs and renders a "
                               "different person; the pointer sentence renders "
                               "the referenced one. Rules in "
                               "prompts/h3_refs_attached_rules.md.",
                }),
                            # APPENDED LAST on purpose: widgets_values is positional,
                # so a new widget at the very end cannot shift the ones
                # every saved graph already depends on.
                "engine": (["auto", "h3"], {
                    "default": "auto",
                    "tooltip": "Which video model these prompts are FOR. h3 loads the MiniMax-H3 variant of the system prompt, which tags speech as <d>[English] ...</d> rather than quoting it - H3 reads double quotes as text printed in the shot, so a quoted line gets painted on a wall and the audio is left unscripted. auto keeps the shared prompt, correct for the LTX/AutoTTS path.",
                }),
},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model_name=None, **kwargs):
        # model_name is a COMBO now. ComfyUI rejects a stored value that is not
        # in the current list, which would break a saved graph whenever the
        # endpoint is unreachable or a model has been pulled. The value is only
        # ever forwarded as a string to the API, so accept anything.
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts_json",)
    FUNCTION = "enhance"
    CATEGORY = "JoyAI-Echo"

    def enhance(
        self,
        story_idea: str,
        mode: str,
        api_key: str,
        system_prompt: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        model_name_custom: str = "",
        num_shots: int = 0,
        temperature: float = 0.7,
        num_frames: int = 0,
        fps: float = 25.0,
        join_style: str = "cuts (independent setups)",
        refs_attached: bool = False,
        # Defaulted so an older saved graph that never sends it still runs.
        engine: str = "auto",
    ):
        import urllib.request
        import urllib.error

        # Resolve the model BEFORE any mode branches, so every call site and
        # every log line reports the model actually used.
        if str(model_name).startswith("(custom"):
            _c = (model_name_custom or "").strip()
            if not _c:
                raise ValueError(
                    "LLMEnhance: model_name is set to the custom entry but "
                    "model_name_custom is empty - type the model name there, "
                    "or pick a discovered model from the dropdown.")
            model_name = _c
            print(f"[JoyEcho] LLMEnhance: using custom model {model_name!r}.",
                  flush=True)

        # Normalise the input ONCE, for every mode (Rebels patch 2026-08-02).
        # Either a .txt LPFF paragraph or a {"prompts":[...]} script is valid in
        # all four modes now; _llme_shots_in never raises on shape.
        _shots_in, _was_json = _llme_shots_in(story_idea)
        if not _shots_in:
            raise ValueError(
                f"LLMEnhance ({mode}): story_idea is empty. Give it either a "
                'premise, a finished prompt paragraph, or a {"prompts": [...]} '
                "script.")
        print(f"[JoyEcho] LLMEnhance: mode={mode!r}, input="
              f"{'JSON script' if _was_json else 'plain text'}, "
              f"{len(_shots_in)} shot(s) in.", flush=True)

        # NOTE: the old auto-passthrough (a JSON script silently forced
        # mode='passthrough', ignoring the dropdown) is GONE. It meant selecting
        # 'long_story' and feeding a script did nothing at all, with one log line
        # as the only clue. The dropdown already has an explicit passthrough
        # entry, so the override was redundant AND it overrode an explicit user
        # choice. Modes are now honoured as selected.

        # PASSTHROUGH: feed straight {"prompts":[...]} JSON in story_idea and skip the
        # LLM entirely. Lets the same node/wiring accept either an enhanced brief or a
        # finished script (e.g. from the Script Picker) via the mode toggle.
        if "passthrough" in mode.lower():
            # Both shapes handled by the shared normaliser. A plain LPFF
            # paragraph IS a finished one-shot script - refusing it used to
            # force every txt batch through the LLM rewrite, which silently
            # destroyed any experiment depending on exact prompt text (found by
            # the accent census, 2026-07-29). A JSON dict with no usable array
            # used to raise; it is now treated as prose rather than failing the
            # render.
            print(f"[JoyEcho] LLMEnhance PASSTHROUGH: {len(_shots_in)} shot(s), "
                  f"no LLM call.", flush=True)
            return (json.dumps({"prompts": _shots_in}, ensure_ascii=True),)

        # ── REVISE: rewrite the PROSE of an existing script, keep its SHAPE ──
        # Rebels local patch 2026-07-20. Built for vision-native reasoning
        # models (minimax-m3) that are strong at physical/spatial/camera
        # language but, like every LLM, will happily reword an identity
        # sentence or drop a shot. So structure is NOT trusted to the model:
        # the shot count, the byte-identical identity sentences and the ASCII
        # rule are all re-imposed in code after the call, and a structurally
        # bad reply falls back to the ORIGINAL script rather than failing the
        # render. Pairs with frame-aware pacing (num_frames/fps), which the
        # passthrough path can never use because it skips the LLM entirely.
        if "revise" in mode.lower():
            # PLAIN TEXT is valid input too (2026-07-20): a single LPFF block
            # off JoyEcho_PromptSource is one finished LTX prompt paragraph,
            # not a JSON script, so it is revised as a one-shot script. Output
            # is always {"prompts":[...]} because that is what TextEncode
            # downstream consumes. Parsing is now the shared normaliser.
            _oshots = list(_shots_in)
            if not _was_json:
                print("[JoyEcho] REVISE: input is plain text - treating as a "
                      "single-shot script (no cross-shot continuity available).",
                      flush=True)
            if num_shots and num_shots > 0 and num_shots != len(_oshots):
                print(f"[JoyEcho] REVISE: num_shots={num_shots} IGNORED - revise "
                      f"preserves the input's {len(_oshots)} shot(s) by design "
                      f"(a reply with a different count is rejected). To EXPAND "
                      f"into {num_shots} shots, use mode 'long_story' instead - "
                      f"it treats the input as a premise and writes a new "
                      f"script.", flush=True)
            # Fallback value is ALWAYS the JSON form, even when the input
            # was a plain LPFF paragraph - downstream TextEncode expects
            # {"prompts":[...]}, so a rejected revision must not hand it
            # back raw prose (2026-07-20).
            _fallback = json.dumps({"prompts": _oshots}, ensure_ascii=True)
            print(f"[JoyEcho] LLMEnhance REVISE: {len(_oshots)} shots in, "
                  f"structure will be re-imposed after the call.", flush=True)

            _rev_sys = (
                "You revise shot prompts for a MiniMax H3 multi-shot video harness. "
                "You are given a JSON object {\"prompts\":[...]} where each array item "
                "is ONE shot's complete prompt text.\n\n"
                "HARD RULES - violating any of these makes your output unusable:\n"
                "1. Return the SAME NUMBER of shots, in the SAME ORDER. Never add, "
                "drop, merge or reorder shots.\n"
                "2. If a shot contains a character-identity sentence (a sentence "
                "naming the character - by their name or a descriptive label like "
                "'the dark-haired woman' - and describing their appearance and "
                "wardrobe, e.g. 'Dana is a woman in her thirties with ...' or 'The "
                "dark-haired woman is a young woman with ...'), copy it "
                "BYTE-FOR-BYTE into your revision of that shot - never reword it "
                "even slightly, because cross-shot identity depends on it being "
                "character-identical. Not every script uses them; if there is none, "
                "keep the subject's described appearance and wardrobe unchanged "
                "instead. Likewise keep any trained-LoRA trigger token (a lowercase "
                "word ending in '_rift') exactly as written and in place.\n"
                "3. Keep each shot's quoted dialogue MEANING and its position in the "
                "story. You may re-word a line for rhythm, but never change what it "
                "says or move it to another shot.\n"
                "4. Keep the capture-medium sentence and the continuous-ambient-sound "
                "sentence in every shot, and keep each shot's trailing 'Audio:' and "
                "'Music:' lines intact.\n"
                "5. Pure ASCII only. No em dashes, smart quotes or unicode.\n\n"
                "WHAT YOU SHOULD IMPROVE: the physical, spatial and camera writing. "
                "Make the scene description more concrete and more renderable - named "
                "materials with a condition ('cracked wet asphalt', 'rust-streaked "
                "steel'), explicit light sources and their direction and falloff, one "
                "clear camera move stated in its OWN clause, never mixed into the "
                "subject's action sentence, with the frame described AFTER the move, "
                "ONE dominant action arc per shot with beats in order, and beats that "
                "are physically possible in the shot's duration. Use only the model's "
                "camera-motion vocabulary (Zoom In, Zoom Out, Push In, Pull Out, Pan "
                "Left, Pan Right, Truck Left, Truck Right, Tilt Up, Tilt Down, "
                "Pedestal Up, Pedestal Down, Arc Shot, Tracking Shot, Static Shot, "
                "Shake Slightly, Shake Strongly, POV, Roll Clockwise, Roll "
                "Counterclockwise), written as a natural English action inside the "
                "sentence. Prefer literal physical description over poetic or "
                "emotional abstraction; emotion belongs only in the performance and "
                "the dialogue itself. Never describe anything the camera cannot see. "
                "Keep the capture device's CHARACTER consistent: if the shot is "
                "documentary or found-footage, preserve and enrich its consumer-"
                "device behaviors (autofocus hunts, exposure lifts, breath sway, "
                "sensor noise) and remove cinema-rig vocabulary (dolly, jib, "
                "crane, gimbal, glide, cinematic, bokeh, slow motion, drone).\n\n"
                "Output ONLY the JSON object. No commentary, no markdown fence."
            )
            _rev_user = "SCRIPT TO REVISE:\n" + json.dumps(
                {"prompts": _oshots}, ensure_ascii=True)
            _fps_r = fps if (fps and fps > 0) else 25.0
            # Same budget as the main path, from the same function. This used
            # to size against the RAW clip length while the main path
            # subtracted the airlock and replay, so on a chained render revise
            # asked for lines that could not fit the block.
            _rev_chained = ("extend" if _is_extend(join_style)
                            else not join_style.startswith("cuts"))
            _rev_budget = _speakable_budget(num_frames, _fps_r, _rev_chained)
            _rev_lo = _rev_hi = None
            if _rev_budget is None:
                _no_duration_warning("REVISE")
            else:
                _secs, _rev_speak, _rev_lo, _rev_hi = _rev_budget
                _rev_user += (
                    f"\n\nEach shot renders as one continuous clip about "
                    f"{_secs:.1f} seconds long at {int(round(_fps_r))} fps. Pace "
                    f"every shot's action to fill that duration, and size each "
                    f"spoken line to roughly {_rev_lo}-{_rev_hi} words so the "
                    f"speech fits the clip with room for breath - lengthen or "
                    f"tighten the existing lines as needed WITHOUT changing what "
                    f"they say.")
                if _rev_chained:
                    _rev_user += (
                        f" Of those {_secs:.0f} seconds only about "
                        f"{_rev_speak:.0f} are speakable: the first ~2 seconds "
                        "of every shot after the first are a silent hold and "
                        "the last ~2 seconds are a silent settle.")

            _payload = json.dumps({
                "model": model_name,
                "messages": [{"role": "system", "content": _rev_sys},
                             {"role": "user", "content": _rev_user}],
                "temperature": temperature,
                "max_tokens": 65536,  # thinking models (minimax-m3) burn 30-40k tokens reasoning; 16384 cut mid-think -> empty content (measured 2026-08-06). Ceiling, not target.
                **_ollama_ctx(base_url),
            }).encode("utf-8")
            _url = base_url.rstrip("/") + "/chat/completions"
            _hdrs = {"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key.strip()}"}
            print(f"[JoyEcho] REVISE: calling {model_name}...", flush=True)
            try:
                _rq = urllib.request.Request(_url, data=_payload, headers=_hdrs,
                                             method="POST")
                with urllib.request.urlopen(_rq, timeout=86400) as _rs:
                    _rr = json.loads(_rs.read().decode("utf-8"))
                _rtext = _rr["choices"][0]["message"]["content"].strip()
            except Exception as _e:  # noqa: BLE001
                print(f"[JoyEcho] REVISE: call failed ({_e}); keeping ORIGINAL "
                      f"script.", flush=True)
                return (_fallback,)

            import re as _re2
            _rtext = _re2.sub(r"<think>.*?</think>", "", _rtext,
                              flags=_re2.DOTALL | _re2.IGNORECASE).strip()
            if _rtext.startswith("```"):
                _rtext = "\n".join(l for l in _rtext.split("\n")
                                   if not l.strip().startswith("```")).strip()
            if not (_rtext.startswith("{") and _rtext.endswith("}")):
                _carved = _extract_json_object(_rtext)
                if _carved is not None:
                    _rtext = _carved
            try:
                _new = json.loads(_rtext)
                _nshots = _new.get("prompts") or _new.get("shots")
                assert isinstance(_nshots, list)
                _nshots = [str(s) for s in _nshots]
            except Exception as _e:  # noqa: BLE001
                print(f"[JoyEcho] REVISE: unparsable reply ({_e}); keeping "
                      f"ORIGINAL script.", flush=True)
                return (_fallback,)

            if len(_nshots) != len(_oshots):
                print(f"[JoyEcho] REVISE: shot count changed "
                      f"({len(_oshots)} -> {len(_nshots)}); keeping ORIGINAL "
                      f"script.", flush=True)
                return (_fallback,)

            # Re-impose structure: ASCII-fold, then force every identity
            # sentence back to the ORIGINAL wording per shot.
            _idre = _ID_SENTENCE
            _fixed, _n_id = [], 0
            for _o, _n in zip(_oshots, _nshots):
                _n = (_n.replace("—", "-").replace("–", "-")
                        .replace("‘", "'").replace("’", "'")
                        .replace("“", '"').replace("”", '"')
                        .replace("…", "..."))
                _n = "".join(c for c in _n if ord(c) < 128)
                _om = _idre.search(_o)
                _nm = _idre.search(_n)
                if _om and _nm and _om.group(0) != _nm.group(0):
                    _n = _n.replace(_nm.group(0), _om.group(0), 1)
                    _n_id += 1
                elif _om and not _nm:
                    print("[JoyEcho] REVISE: a shot lost its identity sentence; "
                          "keeping ORIGINAL script.", flush=True)
                    return (_fallback,)
                # LoRA trigger guard: RIFT corpus prompts carry trained-face
                # triggers (e.g. alice_rift, bob_rift). Dropping one renders a
                # generic face, so a revision that loses any trigger the
                # original had is rejected outright (2026-07-20).
                _otrig = set(_re2.findall(r"\b[a-z][a-z0-9_]*_rift\b", _o))
                if _otrig and not _otrig.issubset(
                        set(_re2.findall(r"\b[a-z][a-z0-9_]*_rift\b", _n))):
                    print(f"[JoyEcho] REVISE: revision dropped LoRA trigger(s) "
                          f"{sorted(_otrig)}; keeping ORIGINAL script.", flush=True)
                    return (_fallback,)
                _fixed.append(_n)
            print(f"[JoyEcho] REVISE: {len(_fixed)} shots revised "
                  f"({_n_id} identity sentence(s) restored).", flush=True)
            # Revise returns ~270 lines before the main path's budget check,
            # so without this nothing ever looked at what came back.
            if _rev_lo and _rev_hi:
                _budget_warn(_fixed, _rev_lo, _rev_hi)
            if refs_attached:
                _fixed, _nrep, _nkept = _point_identity_at_refs(_fixed, story_idea)
                print(f"[JoyEcho] refs_attached: {_nrep} identity sentence(s) "
                      f"pointed at the reference photographs; {_nkept} kept as "
                      "written (name not in the premise, so no photographs "
                      "exist for them).", flush=True)
            return (json.dumps({"prompts": _fixed}, ensure_ascii=True),)

        if not api_key.strip():
            # Local OpenAI-compatible servers (Ollama, LM Studio, llama.cpp,
            # vLLM) ignore the Authorization header, but one still has to be
            # sent. Synthesize a placeholder instead of making every local user
            # discover they must type a fake key into a field their endpoint
            # never reads - that empty-field error was the single most common
            # first-run stumble on this node.
            from urllib.parse import urlparse as _urlparse
            _host = (_urlparse(base_url).hostname or "").lower()
            _is_local = (_host in ("localhost", "127.0.0.1", "0.0.0.0", "::1",
                                   "host.docker.internal")
                         or _host.startswith("192.168.")
                         or _host.startswith("10.")
                         or _host.endswith(".local"))
            if _is_local:
                api_key = "local"
                print(f"[JoyEcho] LLMEnhance: no api_key set and {base_url} is a local "
                      f"endpoint - sending a placeholder key.", flush=True)
            else:
                # Cloud endpoint with a blank key field: check the environment
                # before failing. Keys must never be typed into the widget -
                # workflow metadata embeds into every output file.
                import os as _os
                _env_names = ["JOYECHO_API_KEY"]
                if "deepseek" in _host:
                    _env_names.append("DEEPSEEK_API_KEY")
                elif "openai" in _host:
                    _env_names.append("OPENAI_API_KEY")
                elif "bigmodel" in _host or "zhipu" in _host:
                    _env_names.append("GLM_API_KEY")
                _env_key = next((_os.environ[n] for n in _env_names
                                 if _os.environ.get(n, "").strip()), "").strip()
                if _env_key:
                    api_key = _env_key
                    print(f"[JoyEcho] LLMEnhance: api_key read from environment "
                          f"(checked: {', '.join(_env_names)}).", flush=True)
                else:
                    raise ValueError(
                        f"API key is required for {base_url}. Set it in the environment "
                        f"({' or '.join(_env_names)}) before launching ComfyUI - "
                        f"preferred, keys never touch the graph - or enter your "
                        f"provider's key. Local endpoints such as "
                        f"http://localhost:11434/v1 do not need one - leave this blank.")

        if system_prompt.strip():
            sys_prompt = system_prompt.strip()
        else:
            sys_prompt = _load_system_prompt(mode, engine)

        # Chained join styles append the boundary discipline AFTER whatever
        # system prompt is in use - the user's custom prompt included, so a
        # hand-tuned voice still gets the render-verified boundary rules.
        _rules = _load_boundary_rules(join_style)
        if _rules:
            sys_prompt = sys_prompt.rstrip() + "\n\n" + _rules
            print(f"[JoyEcho] LLMEnhance join_style='{join_style}': "
                  "boundary rules appended to the system prompt.", flush=True)
        if refs_attached:
            sys_prompt = sys_prompt.rstrip() + "\n\n" + _load_refs_rules()
            print("[JoyEcho] LLMEnhance refs_attached: identity sentences will "
                  "point at the reference photographs instead of describing "
                  "the person (rules appended to the system prompt).",
                  flush=True)

        # long_story / short_story treat the input as SOURCE MATERIAL. A plain
        # premise goes in as-is; a finished script is handed over as its shots
        # in order, as readable prose rather than raw JSON, with the job stated
        # explicitly - otherwise the model sees a wall of escaped JSON and
        # tends to echo it back instead of writing (Rebels patch 2026-08-02).
        if _was_json:
            _numbered = "\n\n".join(f"SHOT {i+1}: {s}"
                                    for i, s in enumerate(_shots_in))
            user_msg = (
                "Below is an EXISTING shot script. Use it as source material: "
                "keep what works, and rewrite it into a stronger, coherent "
                "story that renders well. You may re-pace it, deepen it, and "
                "add or remove beats.\n\n" + _numbered)
            print(f"[JoyEcho] LLMEnhance {mode}: rewriting an existing "
                  f"{len(_shots_in)}-shot script as source material.", flush=True)
        else:
            user_msg = _shots_in[0]
        if num_shots > 0:
            user_msg += f"\n\nGenerate exactly {num_shots} shots."

        # Rebels local patch: frame-aware pacing. The enhancer otherwise assumes
        # a fixed ~10s clip and sizes dialogue for it; when the workflow renders
        # longer shots (e.g. 361 frames), tell the LLM the REAL per-shot duration
        # so it fills the time with beats + proportionally longer speech instead
        # of a 10s line stranded in a 14s clip. num_frames=0 -> unchanged.
        _fps = fps if (fps and fps > 0) else 25.0
        _have_budget = bool(num_frames and num_frames > 0)
        if not (num_frames and num_frames > 0):
            _no_duration_warning("LLMEnhance %s" % mode)
        if num_frames and num_frames > 0:
            secs = num_frames / _fps
            # Chained styles spend ~1s of every later block on the discarded
            # replay and ~4s on the airlock + settle, so the SPEAKABLE span
            # is shorter than the block. Sizing dialogue to the full block
            # is exactly the failure that dropped the airlock and chopped
            # the line head (render-verified 2026-08-10).
            _chained = ("extend" if _is_extend(join_style)
                        else not join_style.startswith("cuts"))
            _, _speak, lo, hi = _speakable_budget(num_frames, _fps, _chained)
            if _chained == "extend":
                _n = max(1, int(num_shots or 1))
                _tot_s = secs + (_n - 1) * (secs - 1.0)
                user_msg += (
                    "\n\nEXTEND TAKE: this is ONE unbroken take of about "
                    f"{_tot_s:.0f} seconds, rendered as {_n} consecutive "
                    f"windows of {secs:.1f} seconds each (at "
                    f"{int(round(_fps))} fps). Write ONE continuous speech for "
                    f"the whole take - about {lo * _n}-{hi * _n} words in "
                    f"total - then cut it into the {_n} blocks at phrase "
                    f"boundaries so each block carries about {lo}-{hi} words - "
                    f"aim for the UPPER half of that range ({(lo + hi) // 2}-{hi}), "
                    f"because a thin block renders as seconds of dead air (a "
                    f"10-second window with 11 words measured a 5-second hole) - "
                    f"and every block speaks. Speech runs through the "
                    f"boundaries with no pauses; block k+1 begins with the next "
                    f"words after block k. Same visual block, verbatim, in "
                    f"every window. These numbers override any default clip "
                    f"length mentioned above."
                )
            else:
              user_msg += (
                f"\n\nEach shot is a single continuous clip about {secs:.1f} "
                f"seconds long (at {int(round(_fps))} fps). Pace every shot to "
                f"fill roughly {secs:.0f} seconds: give the action enough small "
                f"beats to occupy the full duration without any fast or complex "
                f"motion, and do not leave long dead air. For a speaking shot, "
                f"scale the dialogue to about {lo}-{hi} words, "
                f"delivered as one natural line or a short two-line exchange, with "
                f"room for pauses, breath, and reaction. These per-shot timing "
                f"numbers override any default clip length mentioned above."
              )
            if _chained and _chained != "extend":
                user_msg += (
                    f" Of those {secs:.0f} seconds, only about {_speak:.0f} "
                    "are speakable: the first ~2 seconds of every shot after "
                    "the first are a silent hold and the last ~2 seconds of "
                    "every shot are a silent settle, per the block-boundary "
                    "rules."
                )

        url = base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps({
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temperature,
            "max_tokens": 65536,  # thinking models (minimax-m3) burn 30-40k tokens reasoning; 16384 cut mid-think -> empty content (measured 2026-08-06). Ceiling, not target.
            **_ollama_ctx(base_url),
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }

        print(f"[JoyEcho] Calling LLM ({model_name}) to enhance prompt...", flush=True)

        def _clean_to_json(raw):
            """Peel everything that is not the {"prompts":[...]} envelope."""
            import re as _re
            c = _re.sub(r"<think>.*?</think>", "", raw,
                        flags=_re.DOTALL | _re.IGNORECASE).strip()
            # A fence may sit on its own line OR share the line with the JSON
            # (```json {"prompts": ...). Dropping whole fence lines threw the
            # payload away in the second case, so strip the token itself first.
            c = _re.sub(r"^\s*```[a-zA-Z0-9_-]*[ \t]*", "", c)
            c = _re.sub(r"\s*```\s*$", "", c).strip()
            c = "\n".join(l for l in c.split("\n")
                          if not l.strip().startswith("```")).strip()
            if not (c.startswith("{") and c.endswith("}")):
                _obj = _extract_json_object(c)
                if _obj is not None:
                    c = _obj
            if (not (c.startswith("{") and c.endswith("}"))
                    and "integrated_multimodal_description:" in c):
                _blocks = [b.strip() for b in
                           _re.split(r"(?m)^---\s*$", c) if b.strip()]
                if _blocks:
                    c = json.dumps({"prompts": _blocks})
                    print(f"[JoyEcho] LLMEnhance: wrapped {len(_blocks)} raw H3 "
                          f"block(s) into the prompts envelope.", flush=True)
            return c

        def _salvage_prompts(raw):
            """Last resort: pull COMPLETE shot strings out of a broken payload.

            Local models truncate mid-string at the token ceiling, and some
            narrate their way out of the array. Neither leaves a balanced
            object, so json.loads cannot help. Every element that DID close
            cleanly is still a usable prompt - take those and drop the partial
            tail rather than losing the whole render.
            """
            import re as _re
            m = _re.search(r'"prompts"\s*:\s*\[', raw)
            if not m:
                return []
            body = raw[m.end():]
            n = len(body)
            out = []
            i = 0
            while i < n:
                if body[i] != '"':
                    i += 1
                    continue
                j = i + 1
                buf = []
                closed = False
                while j < n:
                    ch = body[j]
                    if ch == "\\" and j + 1 < n:
                        buf.append(body[j:j + 2])
                        j += 2
                        continue
                    if ch == '"':
                        closed = True
                        break
                    buf.append(ch)
                    j += 1
                if not closed:
                    break
                try:
                    out.append(json.loads('"' + "".join(buf) + '"'))
                except Exception:
                    pass
                i = j + 1
            return [s for s in out if len(s) > 120]

        # Ollama cloud models intermittently return EMPTY completions, and
        # local models regularly emit prose around or inside the envelope.
        # BOTH are retried here: parsing used to sit OUTSIDE this loop, so one
        # malformed answer killed a render that a re-ask would have fixed.
        _raw = ""
        data = None
        _perr = None
        for _attempt in range(3):
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=86400) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"LLM API error {e.code}: {body}")
            _raw = result["choices"][0]["message"]["content"].strip()
            if not _raw:
                print(f"[JoyEcho] LLMEnhance: EMPTY completion from {model_name} "
                      f"(attempt {_attempt+1}/3) - retrying...", flush=True)
                import time as _t
                _t.sleep(3)
                continue
            content = _clean_to_json(_raw)
            try:
                _d = json.loads(content)
                if "prompts" not in _d or not isinstance(_d["prompts"], list):
                    raise ValueError("output missing 'prompts' array")
                if not _d["prompts"]:
                    raise ValueError("'prompts' array is empty")
                # First thing that checks the budget computed above
                # was actually honoured. lo/hi only exist when num_frames > 0.
                if _have_budget:
                    _budget_warn(_d["prompts"], lo, hi)
                data = _d
                break
            except (json.JSONDecodeError, ValueError) as e:
                _perr = e
                print(f"[JoyEcho] LLMEnhance: unparseable output from "
                      f"{model_name} ({e}) (attempt {_attempt+1}/3) - "
                      f"retrying...", flush=True)
                import time as _t
                _t.sleep(3)

        if data is None:
            _sal = _salvage_prompts(_raw)
            if _sal:
                data = {"prompts": _sal}
                print(f"[JoyEcho] LLMEnhance: output never parsed cleanly; "
                      f"SALVAGED {len(_sal)} complete shot(s) and dropped the "
                      f"truncated tail. Check SCRIPT PREVIEW before rendering.",
                      flush=True)
            elif not _raw:
                raise RuntimeError(
                    f"{model_name} returned 3 EMPTY completions in a row "
                    f"(~2 min each). Thinking models can exhaust their "
                    f"reasoning budget on long enhancer prompts and emit no "
                    f"answer. Switch the model widget to a vetted writer - no "
                    f"restart needed, just re-run.")
            else:
                raise RuntimeError(
                    f"{model_name} returned unusable output 3 times and "
                    f"nothing could be salvaged: {_perr}\n\n"
                    f"Smaller local models often narrate instead of emitting "
                    f"JSON, or truncate mid-array. Try a larger model, or set "
                    f"mode to 'passthrough (raw JSON, skip LLM)' and supply "
                    f"the script yourself.\n\nRaw output:\n{_raw[:800]}")
        num = len(data["prompts"])

        if refs_attached:
            data["prompts"], _nrep, _nkept = _point_identity_at_refs(
                data["prompts"], story_idea)
            content = json.dumps(data, ensure_ascii=True)
            print(f"[JoyEcho] refs_attached: {_nrep} identity sentence(s) "
                  f"pointed at the reference photographs; {_nkept} kept as "
                  "written (name not in the premise, so no photographs exist "
                  "for them). A written description beats attached photos - "
                  "same-seed measured.", flush=True)

        print(f"[JoyEcho] LLM generated {num} shot prompt(s).", flush=True)

        # Persist + echo the generated prompts so you can inspect exactly what
        # the enhancer produced (this JSON is what feeds JoyEcho_TextEncode).
        try:
            import os
            import folder_paths
            _outdir = os.path.join(folder_paths.get_output_directory(), "joyecho")
            os.makedirs(_outdir, exist_ok=True)
            _dump = os.path.join(_outdir, "enhanced_prompts_latest.json")
            with open(_dump, "w", encoding="utf-8") as _f:
                _f.write(content)
            print(f"[JoyEcho] enhancer output written to: {_dump}", flush=True)
        except Exception as _e:
            print(f"[JoyEcho] could not write enhancer output file: {_e}", flush=True)
        print("[JoyEcho] ---------- enhancer output (prompts) ----------", flush=True)
        print(content, flush=True)
        print("[JoyEcho] ---------- end enhancer output ----------", flush=True)
        return (content,)


class JoyEcho_PromptAtIndex:
    """Extract a single prompt from a JSON prompts array by index.

    Connect the output to a SingleShotGenerate node's prompt input to override
    the text box with LLM-generated content. This is optional — if not connected,
    the SingleShot node uses its own text box.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "JSON string with 'prompts' array (from LLM Enhance or file)",
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 29,
                    "tooltip": "0-based shot index to extract",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "extract"
    CATEGORY = "JoyAI-Echo"

    def extract(self, prompts_json: str, index: int):
        text = prompts_json.strip()
        if not text:
            raise ValueError("No prompts JSON provided.")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        prompt_list = data.get("prompts") or data.get("shots") or []
        if not prompt_list:
            raise ValueError("JSON must contain a 'prompts' or 'shots' array.")

        if index >= len(prompt_list):
            raise ValueError(
                f"Index {index} out of range (only {len(prompt_list)} prompts available)."
            )

        return (str(prompt_list[index]).strip(),)
