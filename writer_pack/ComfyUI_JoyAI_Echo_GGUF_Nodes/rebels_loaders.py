"""
rebels_loaders.py — discrete ComfyUI loader nodes for JoyAI-Echo on low VRAM.
Patched for Single-File Gemma intake and Key Remapping.
"""
from __future__ import annotations
import os, json, gc
_LOADER_DIR = os.path.dirname(os.path.abspath(__file__))
_LOADER_CFG = os.path.join(_LOADER_DIR, "configs", "joyai_echo_config.json")
import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf
from gguf import GGUFReader, GGMLQuantizationType as QT

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator, X0Model
from ltx_core.model.video_vae import (VAE_DECODER_COMFY_KEYS_FILTER, VAE_ENCODER_COMFY_KEYS_FILTER,
                                      VideoDecoderConfigurator, VideoEncoderConfigurator)
from ltx_core.model.audio_vae import (AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
                                      VOCODER_COMFY_KEYS_FILTER, AudioDecoderConfigurator,
                                      AudioEncoderConfigurator, VocoderConfigurator)
from ltx_core.text_encoders.gemma import (EMBEDDINGS_PROCESSOR_KEY_OPS, EmbeddingsProcessorConfigurator)
from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper
from ltx_distillation.models.vae_wrapper import VideoVAEWrapper, AudioVAEWrapper
from ltx_distillation.models.text_encoder_wrapper import GemmaTextEncoderWrapper

CAT = "Rebels/JoyAI-Echo"

# ---------------------------------------------------------------- config
def _full_config(src: str) -> dict:
    src = src.strip().strip('"')
    if src.lower().endswith(".json"):
        with open(src, "r", encoding="utf-8") as f: return json.load(f)
    from safetensors import safe_open
    with safe_open(src, framework="pt") as f:
        meta = f.metadata() or {}
    if "config" not in meta:
        raise ValueError(f"No 'config' in metadata of {src}. Point at the checkpoint or a config.json.")
    return json.loads(meta["config"])

_GGUF_MARKER = "__rebels_gguf_no_safetensors__"

def _gemma_remap_key(k):
    """Map raw checkpoint key names onto the wrapper's module tree. Shared by
    the safetensors load path and the GGUF swap so both match identically."""
    nk = k.replace("cond_stage_model.", "").replace("text_model.", "").replace("text_encoder.", "")
    if "embed_tokens" in nk:
        nk = "model.model.language_model.embed_tokens.weight"
    elif "layers" in nk:
        nk = nk.replace("model.layers", "model.model.language_model.layers")
    elif "norm" in nk and "language_model" not in nk:
        nk = nk.replace("model.norm", "model.model.language_model.norm")
    return nk

class _CfgLoader(SafetensorsModelStateDictLoader):
    """Stock safetensors weight load, but metadata() returns the shared config."""
    def __init__(self, config: dict, map_gemma=False, *a, **k):
        super().__init__(*a, **k)
        self._cfg = config
        self._map_gemma = map_gemma
        
    def metadata(self, path): return self._cfg
    
    def load(self, paths, sd_ops=None, device=None):
        plist = [str(x) for x in (paths if isinstance(paths, (list, tuple)) else [paths])]
        if any(_GGUF_MARKER in x or x.lower().endswith(".gguf") for x in plist):
            import types
            return types.SimpleNamespace(sd={})  # build() only touches .sd
        sd_obj = super().load(paths, sd_ops, device)
        
        if self._map_gemma:
            new_sd = {}
            for k, v in sd_obj.sd.items():
                new_sd[_gemma_remap_key(k)] = v
            return dataclasses.replace(sd_obj, sd=new_sd)
        return sd_obj

# ---------------------------------------------------------------- gguf dit
# Keep GGUFReaders alive for the process lifetime so their memory-mapped data
# (which the GGUFLinear weights below reference WITHOUT copying) stays valid.
_OPEN_GGUF_READERS = []


def _gguf_entries(path):
    r = GGUFReader(path)
    _OPEN_GGUF_READERS.append(r)
    out = {}
    for t in r.tensors:
        out[t.name] = {"data": np.asarray(t.data), "qtype": t.tensor_type,
                       "shape": tuple(int(d) for d in reversed(t.shape))}
    return out

def _dequant(entry, dtype):
    data = np.asarray(entry["data"])
    q = int(entry["qtype"])
    # Unquantized tensors (norms etc. are stored F32/F16 in the GGUF): skip the
    # old `.astype(np.float32)` which COPIED every one of them into a fresh f32
    # array before making a second bf16 copy. from_numpy on the memmap view is
    # zero-copy; the single .to(dtype) below is the only allocation.
    if q in (int(QT.F32), int(QT.F16)):
        return torch.from_numpy(data).to(dtype)
    # Quantized tensors: prefer city96's pure-torch kernels (they run fine on
    # CPU) dequanting STRAIGHT to the target dtype -- no numpy grouped-rows
    # machinery and no f32 staging copy. This halves peak RAM per tensor vs the
    # old path, which is what was tipping the Windows commit limit during load.
    if _CITY_DEQUANT is not None and _GPU_DEQUANT_OK:
        try:
            raw = torch.from_numpy(data)
            t = _CITY_DEQUANT(raw, QT(q), tuple(entry["shape"]), dtype=dtype)
            return t.to(dtype)
        except Exception:
            pass  # fall through to the numpy reference path
    deq = gguf.quants.dequantize(data, QT(q))
    t = torch.from_numpy(deq).to(dtype)
    del deq
    return t
    # NOTE: we deliberately do NOT delete entry["data"] anymore. It is a view
    # into the memory-mapped file (costs no resident RAM), and the post-build
    # meta-materialization sweep needs entries to stay readable.

# --- optional GPU dequant kernels, borrowed from city96's ComfyUI-GGUF -------
# city96 dequantizes packed GGUF weights with pure-torch kernels that run ON THE
# GPU. That is the single biggest reason his loader is fast and ours was slow:
# our old path dequantized every weight on the CPU through numpy on EVERY
# forward. If the user has ComfyUI-GGUF installed (Noah does), import its
# dequant module and use it; otherwise fall back to the numpy path.
_CITY_DEQUANT = None
_GPU_DEQUANT_OK = True  # legacy flag (kept for _dequant)
_GPU_DEQUANT_BAD = set()  # qtypes whose GPU kernels failed; per-type, never global


def _patch_gemma3_rope_compat():
    """transformers >=~4.56 moved per-layer RoPE attrs (rope_local_base_freq)
    into a rope_parameters dict; the LTX/JoyAI libs read the old attribute
    directly. Install a __getattr__ fallback on Gemma3TextConfig that derives
    the value from rope_parameters or returns the Gemma-3 default (10000.0),
    chaining to any pre-existing __getattr__. No-op on transformers versions
    that still have the attribute."""
    try:
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    except Exception:
        return
    if getattr(Gemma3TextConfig, "_rebels_rope_compat", False):
        return
    _orig_ga = Gemma3TextConfig.__getattr__ if "__getattr__" in vars(Gemma3TextConfig) else None
    _base_ga = getattr(super(Gemma3TextConfig, Gemma3TextConfig), "__getattr__", None)

    def _ga(self, name):
        if name == "rope_local_base_freq":
            rp = self.__dict__.get("rope_parameters")
            if isinstance(rp, dict):
                for key in ("sliding_attention", "local_attention", "local"):
                    sub = rp.get(key)
                    if isinstance(sub, dict) and sub.get("rope_theta"):
                        return sub["rope_theta"]
            return 10000.0
        if _orig_ga is not None:
            return _orig_ga(self, name)
        if _base_ga is not None:
            return _base_ga(self, name)
        raise AttributeError(name)

    Gemma3TextConfig.__getattr__ = _ga
    Gemma3TextConfig._rebels_rope_compat = True


_patch_gemma3_rope_compat()


def gpu_dequant_supported(qtype_value):
    """Probe whether city96's GPU kernels can handle this qtype, WITHOUT
    running a real layer. Probes a tiny zeros tensor on CUDA; failures are
    remembered in _GPU_DEQUANT_BAD. Used as a pre-flight so GPU encode/denoise
    never silently grinds on CPU fallback for an unsupported quant type."""
    q = int(qtype_value)
    if q in (int(QT.F32), int(QT.F16)):
        return True
    if _CITY_DEQUANT is None or not torch.cuda.is_available():
        return False
    if q in _GPU_DEQUANT_BAD:
        return False
    try:
        block_size, type_size = gguf.GGML_QUANT_SIZES[QT(q)]
        data = torch.zeros((1, type_size), dtype=torch.uint8, device="cuda")
        _CITY_DEQUANT(data, QT(q), (1, block_size), dtype=torch.bfloat16)
        return True
    except Exception:
        _GPU_DEQUANT_BAD.add(q)
        return False
try:
    import importlib.util as _ilu
    _cn_dir = os.path.dirname(_LOADER_DIR)  # .../custom_nodes
    for _cand in ("ComfyUI-GGUF", "ComfyUI-GGUF-main", "comfyui-gguf"):
        _dq = os.path.join(_cn_dir, _cand, "dequant.py")
        if os.path.isfile(_dq):
            _spec = _ilu.spec_from_file_location("rebels_city96_dequant", _dq)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _CITY_DEQUANT = getattr(_mod, "dequantize", None)
            if _CITY_DEQUANT:
                print(f"[Rebels JE] GPU dequant kernels loaded from {_cand}/dequant.py", flush=True)
            break
except Exception as _e:
    print(f"[Rebels JE] city96 dequant unavailable ({_e}); using CPU numpy dequant.", flush=True)


class GGUFLinear(nn.Module):
    def __init__(self, entry, bias=None, compute_dtype=torch.bfloat16):
        super().__init__()
        self.qtype_value = int(entry["qtype"])
        self.weight_shape = tuple(entry["shape"])
        # The packed weight is kept as a PLAIN attribute (not a registered buffer):
        #  - it stays a numpy view into the memory-mapped GGUF file, so it costs
        #    ~zero resident RAM (the .copy() that duplicated the whole DiT in RAM
        #    is gone for good);
        #  - module.to(device) / state_dict / pin_memory all ignore it, so the
        #    sequential offloader can shuttle blocks to the GPU without dragging
        #    9GB of packed weights along or pinning them.
        # Each forward streams just this layer's packed bytes to the GPU and
        # dequantizes there (city96 kernels) -- the same per-layer streaming that
        # makes Noah's other big LTX GGUF models run fine on 8GB.
        self._qweight = entry["data"]
        self.bias = nn.Parameter(bias.to(compute_dtype), requires_grad=False) if bias is not None else None

    def forward(self, x):
        q = self.qtype_value
        # F16 / F32 tensors (llama-quantize leaves some in K-quant files) are
        # not in city96's kernel table -- they don't need kernels at all.
        # Previously ONE of these raised a KeyError that tripped a GLOBAL kill
        # switch, silently dropping every linear in the whole run to CPU numpy
        # dequant (the 40-minute generations). Handle them directly:
        if q in (int(QT.F32), int(QT.F16)):
            w = torch.from_numpy(np.asarray(self._qweight))
            w = w.reshape(self.weight_shape).to(device=x.device, dtype=x.dtype)
            return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)
        if _CITY_DEQUANT is not None and q not in _GPU_DEQUANT_BAD:
            try:
                data = torch.from_numpy(np.asarray(self._qweight)).to(x.device, non_blocking=True)
                w = _CITY_DEQUANT(data, QT(q), tuple(self.weight_shape), dtype=x.dtype)
                if tuple(w.shape) != tuple(self.weight_shape):
                    w = w.reshape(self.weight_shape)
                w = w.to(dtype=x.dtype)
                return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)
            except Exception as e:
                # disable the GPU path for THIS qtype only -- never globally
                _GPU_DEQUANT_BAD.add(q)
                print(f"[Rebels JE] GPU dequant unavailable for {QT(q).name} "
                      f"({type(e).__name__}: {e}); that qtype uses CPU dequant.", flush=True)
        raw = np.asarray(self._qweight)
        w = torch.from_numpy(gguf.quants.dequantize(raw, QT(self.qtype_value)).astype(np.float32))
        w = w.reshape(self.weight_shape).to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def _set_sub(root, dotted, new):
    *parents, leaf = dotted.split(".")
    p = root
    for a in parents: p = getattr(p, a)
    setattr(p, leaf, new)

class _GGUFDiTLoader(SafetensorsModelStateDictLoader):
    def __init__(self, config, entries, consumed, dtype):
        super().__init__()
        self._cfg = config
        self._e = entries
        self._consumed = consumed
        self._dt = dtype
    def metadata(self, path): return self._cfg
    def load(self, paths, sd_ops=None, device=None):
        sd, size, n = {}, 0, 0
        for k, e in self._e.items():
            if k in self._consumed: continue
            t = _dequant(e, self._dt)
            # THE META-TENSOR FIX: leftover GGUF keys still carry JD's checkpoint
            # prefix ("model.diffusion_model."), but the bare LTXModel's module
            # names do not. With model_sd_ops=None nothing strips it, so
            # load_state_dict(strict=False) silently skipped every one of these
            # tensors and they stayed empty 'meta' placeholders -- which is the
            # "Tensor on device meta" crash in patchify_proj. Emit the stripped
            # key alongside the raw one (same tensor object, costs nothing).
            sd[k] = t
            for p in ("model.diffusion_model.", "diffusion_model."):
                if k.startswith(p):
                    sd[k[len(p):]] = t
                    break
            size += t.numel() * t.element_size()
            n += 1
            if n % 32 == 0:
                # Periodic (not per-tensor) gc: keeps Windows commit pressure
                # down on a 16GB machine without 1700 collector passes.
                gc.collect()
        gc.collect()
        return StateDict(sd=sd, device=device or torch.device("cpu"), size=size, dtype=self._dt)

# The bare LTXModel modules are named e.g. "transformer_blocks.0.attn1.to_q",
# but the GGUF keys keep JD's checkpoint prefix "model.diffusion_model.". The
# configurator strips that prefix at load time, so we must match across it here
# or the swap fires on nothing and the whole DiT dequantizes into RAM -> OOM.
_DIT_PREFIXES = ("", "model.diffusion_model.", "diffusion_model.")


def _find_entry(entries, base):
    for p in _DIT_PREFIXES:
        k = p + base
        if k in entries:
            return k
    return None


# Some LTX components (the transformer-args preprocessors) capture DIRECT
# OBJECT REFERENCES to modules like patchify_proj at construction time, outside
# the registered module tree. When the mutator swaps those modules for
# GGUFLinear, the preprocessor keeps pointing at the ORIGINAL meta nn.Linear --
# invisible to load_state_dict AND to the meta sweep (which is why the sweep
# reports 0 while the forward still hits a meta tensor). We record every
# old->new swap and then re-bind stale references across the whole object graph.
_SWAP_MAP = {}

def _rebind_swapped(root, max_objs=50000):
    import types as _types
    if not _SWAP_MAP:
        return 0
    _SKIP = (type, _types.FunctionType, _types.MethodType, _types.BuiltinFunctionType,
             _types.ModuleType, str, bytes, int, float, bool, complex, torch.Tensor,
             np.ndarray)
    def _swapped_to(x):
        # _SWAP_MAP values are (new_mod, old_mod). Only a genuine stale reference
        # still points at an old swapped module (always an nn.Linear); the
        # isinstance guard rejects any residual id() collision with a non-Linear
        # object. (Rebels local patch)
        e = _SWAP_MAP.get(id(x))
        if e is None or not isinstance(x, nn.Linear):
            return None
        return e[0]

    seen, queue, fixed = set(), [root], 0
    while queue and len(seen) < max_objs:
        obj = queue.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        containers = []
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            containers.append(d)
        m = getattr(obj, "_modules", None)
        if isinstance(m, dict) and m is not d:
            containers.append(m)
        for cont in containers:
            for k, v in list(cont.items()):
                nv = _swapped_to(v)
                if nv is not None and v is not nv:
                    cont[k] = nv
                    fixed += 1
                    continue
                if isinstance(v, (list,)):
                    for i, item in enumerate(v):
                        nvi = _swapped_to(item)
                        if nvi is not None and item is not nvi:
                            v[i] = nvi
                            fixed += 1
                        elif isinstance(item, nn.Module):
                            queue.append(item)
                    continue
                if isinstance(v, dict):
                    for dk, item in list(v.items()):
                        nvi = _swapped_to(item)
                        if nvi is not None and item is not nvi:
                            v[dk] = nvi
                            fixed += 1
                        elif isinstance(item, nn.Module):
                            queue.append(item)
                    continue
                if isinstance(v, _SKIP) or v is None:
                    continue
                if isinstance(v, nn.Module) or hasattr(v, "__dict__"):
                    queue.append(v)
    if fixed:
        print(f"[Rebels JE] re-bound {fixed} stale reference(s) to swapped GGUF layers.", flush=True)
    return fixed


def _dit_module_ops(entries, consumed, compute_dtype):
    def mutator(model):
        n_lin = n_hit = 0
        miss = []
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            n_lin += 1
            wk = _find_entry(entries, name + ".weight")
            if wk is None:
                if len(miss) < 5:
                    miss.append(name)
                continue
            bk = _find_entry(entries, name + ".bias")
            bias = _dequant(entries[bk], compute_dtype) if bk else None
            new_mod = GGUFLinear(entries[wk], bias, compute_dtype)
            # Value keeps BOTH the replacement and the ORIGINAL module alive:
            # holding `mod` prevents Python from freeing it and reusing its id()
            # for a later allocation (e.g. the X0Model built after this swap),
            # which would make _rebind_swapped rebind an unrelated object to a
            # GGUFLinear and corrupt generator.model. (Rebels local patch)
            _SWAP_MAP[id(mod)] = (new_mod, mod)
            _set_sub(model, name, new_mod)
            consumed.add(wk)
            if bk:
                consumed.add(bk)
            n_hit += 1
        print(f"[Rebels JE] DiT GGUF swap: matched {n_hit}/{n_lin} Linear layers "
              f"({len(consumed)} tensors kept packed).", flush=True)
        if n_hit == 0 and n_lin:
            print(f"[Rebels JE]   NO matches -> whole DiT would dequantize. "
                  f"sample model Linears={miss}", flush=True)
            print(f"[Rebels JE]   sample GGUF keys={list(entries.keys())[:5]}", flush=True)
        return model
    return (ModuleOps("gguf_linear_swap", matcher=lambda m: True, mutator=mutator),)

# ---------------------------------------------------------------- gemma fp8
class Fp8Linear(nn.Module):
    def __init__(self, qweight_u8, shape, scale, bias=None, compute_dtype=torch.bfloat16):
        super().__init__()
        self.weight_shape = tuple(shape)
        self.register_buffer("qweight", qweight_u8)
        self.register_buffer("scale_weight", torch.tensor(float(scale), dtype=torch.float32))
        self.bias = nn.Parameter(bias.to(compute_dtype), requires_grad=False) if bias is not None else None
    def forward(self, x):
        w = self.qweight.view(torch.float8_e4m3fn).reshape(self.weight_shape).to(torch.float32) * self.scale_weight
        w = w.to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def _swap_gemma_gguf(model, gguf_path, compute_dtype):
    """Load Gemma from OUR quantized GGUF (made by make_gemma_gguf.py, which
    preserves HF key names -- so matching is direct, no llama.cpp renaming).
    Every nn.Linear becomes a packed GGUFLinear backed by the memory-mapped
    file (~zero resident RAM); embeddings/norms are dequanted to compute dtype.
    Resident footprint ~2.5GB vs ~12GB for the fp8 path."""
    raw = _gguf_entries(gguf_path)
    entries = {}
    for k, v in raw.items():
        entries[_gemma_remap_key(k)] = v
        entries.setdefault(k, v)  # keep originals too; harmless duplicates
    names = list(entries.keys())

    def find(base):
        if base in entries:
            return base
        cands = [n for n in names if n.endswith("." + base) or base.endswith("." + n)]
        return cands[0] if len(cands) == 1 else None

    used = set()
    n_hit = n_lin = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        n_lin += 1
        wk = find(name + ".weight")
        if wk is None:
            continue
        bk = find(name + ".bias")
        if bk is not None:
            bias = _dequant(entries[bk], compute_dtype)
            used.add(bk)
        elif mod.bias is not None and mod.bias.device.type != "meta":
            bias = mod.bias.detach()
        else:
            bias = None
        _set_sub(model, name, GGUFLinear(entries[wk], bias, compute_dtype))
        used.add(wk)
        n_hit += 1
    print(f"[Rebels JE] Gemma GGUF swap: matched {n_hit}/{n_lin} Linear layers "
          f"(packed, memory-mapped).", flush=True)
    if n_hit == 0 and n_lin:
        print(f"[Rebels JE]   sample GGUF keys={names[:5]}", flush=True)
    # Fill everything else (embeddings, norms) from the GGUF. vision_tower /
    # multi_modal_projector / lm_head have no weights anywhere and get stripped
    # by the staged node right after this -- skip them instead of aborting.
    _materialize_meta(model, entries, used, compute_dtype, strict=False,
                      skip_substrings=("vision_tower", "multi_modal_projector", "lm_head"))
    return model


def _swap_gemma_fp8(model, fp8_dir, compute_dtype):
    """Replace every nn.Linear whose fp8 weight+scale exist in the shards with an
    Fp8Linear. Two on-disk layouts are supported (Rebels local patch):
      legacy "our_fp8":  <module>          = fp8 weight, <module>.scale_weight = scalar,
                         keys already named for the wrapper tree.
      comfy fp8 (e.g. *fp8mixed): <module>.weight = fp8 weight,
                         <module>.weight_scale = scalar float32, HF checkpoint
                         naming (model.layers.*) remapped onto the wrapper tree
                         (model.model.language_model.*) via _gemma_remap_key.
    Per-channel (non-scalar) scales are skipped (Fp8Linear is per-tensor only).
    Weights are .clone()d INSIDE the open file: get_tensor returns a VIEW into
    the mmap, which is unmapped when safe_open closes -- registering that view
    as a buffer dangles and faults natively on a later read."""
    from safetensors import safe_open
    from pathlib import Path
    shards = sorted(Path(fp8_dir).glob("model*.safetensors"))
    entries = {}  # module path -> (shard, weight_key, scale)
    for sh in shards:
        with safe_open(str(sh), framework="pt") as f:
            keys = set(f.keys())
            for k in keys:
                if k.endswith(".scale_weight"):
                    base = k[: -len(".scale_weight")]
                    wkey = base
                elif k.endswith(".weight_scale"):
                    base = k[: -len(".weight_scale")]
                    wkey = base + ".weight"
                else:
                    continue
                if wkey not in keys:
                    continue
                t = f.get_tensor(k)
                if t.numel() != 1:
                    continue
                scale = float(t)
                names = {base}
                # HF naming -> wrapper-tree naming. Skip keys already in wrapper
                # naming: _gemma_remap_key's substring replace would corrupt them.
                if not base.startswith("model.model."):
                    names.add(_gemma_remap_key(base))
                for nm in names:
                    entries.setdefault(nm, (str(sh), wkey, scale))
    n = 0
    by_shard = {}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and name in entries:
            sh, wkey, scale = entries[name]
            by_shard.setdefault(sh, []).append((name, mod, wkey, scale))
    for sh, items in by_shard.items():
        with safe_open(sh, framework="pt") as f:
            for name, mod, wkey, scale in items:
                qw = f.get_tensor(wkey).clone()
                bias = (mod.bias.detach()
                        if (mod.bias is not None and mod.bias.device.type != "meta")
                        else None)
                _set_sub(model, name, Fp8Linear(qw.view(torch.uint8), tuple(qw.shape),
                                                scale, bias, compute_dtype))
                n += 1
    print(f"[Rebels JE] Gemma fp8 swap: matched {n} Linear layers.", flush=True)
    if n == 0:
        print("[Rebels JE]   WARNING: fp8 swap matched NOTHING - the file needs "
              "paired fp8 weights + .scale_weight/.weight_scale scalars; encoder "
              "would stay (mis-scaled) bf16.", flush=True)
    return model

# ================================================================ NODES
def _dev(): return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ---------------------------------------------------------------- model cache
# Some setups run ComfyUI with --cache-none (VRAM hygiene), making loader
# nodes re-execute on EVERY queued prompt. Re-reading ~50GB of weights per
# queue is slow and has crashed natively (Windows mmap reads in a long-lived
# process degrade: access violation in safetensors get_tensor on the 2nd
# prompt of a session, after a clean 2h19m first run). Keep the BUILT models
# in a module-level cache keyed by file identity (path, size, mtime) + build
# params so re-queues reuse them. One entry per node kind; changing any
# path/param evicts and rebuilds. (Rebels local patch)
_NODE_CACHE = {}


def _cache_key(*parts):
    out = []
    for p in parts:
        if isinstance(p, str):
            q = p.strip().strip('"')
            if os.path.isfile(q):
                st = os.stat(q)
                out.append((q, st.st_size, int(st.st_mtime)))
                continue
        out.append(p)
    return tuple(out)


def _cache_get(kind, key):
    hit = _NODE_CACHE.get(kind)
    if hit is not None and hit[0] == key:
        print(f"[Rebels JE] {kind}: reusing cached models from this session "
              f"(skips re-reading weights under --cache-none).", flush=True)
        return hit[1]
    return None


def _cache_put(kind, key, value):
    _NODE_CACHE[kind] = (key, value)
    return value

class RebelsJE_Config:
    CATEGORY = CAT
    RETURN_TYPES = ("JOYECHO_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config_source": ("STRING", {"default": _LOADER_CFG})}}
    def run(self, config_source): return (_full_config(config_source),)

def _materialize_meta(root, entries, consumed, dtype, strict=True, skip_substrings=()):
    """Safety net: find every parameter/buffer still on the 'meta' device after
    build and fill it with real data from the GGUF. Resolves names by trying the
    known checkpoint prefixes first, then by unique longest-suffix match (covers
    modules the configurator registers under different paths, e.g. preprocessor
    wrappers around patchify_proj). Raises a CLEAR error naming any weight it
    cannot find, instead of letting a cryptic 'Tensor on device meta' crash
    happen 40 minutes into a run."""
    leftover = [k for k in entries if k not in consumed]
    fixed, missing = 0, []

    def resolve(pname):
        k = _find_entry(entries, pname)
        if k is not None and k not in consumed:
            return k
        parts = pname.split(".")
        for i in range(len(parts)):
            suf = ".".join(parts[i:])
            cands = [n for n in leftover if n == suf or n.endswith("." + suf)]
            if len(cands) == 1:
                return cands[0]
        return None

    items = [(n, p, True) for n, p in root.named_parameters()] \
          + [(n, b, False) for n, b in root.named_buffers()]
    skipped = 0
    for name, t, is_param in items:
        if t is None or t.device.type != "meta":
            continue
        if any(sub in name for sub in skip_substrings):
            skipped += 1
            continue
        k = resolve(name)
        if k is None:
            missing.append(name)
            continue
        new = _dequant(entries[k], dtype)
        if new.numel() == t.numel() and tuple(new.shape) != tuple(t.shape):
            new = new.reshape(t.shape)
        *path, leaf = name.split(".")
        mod = root
        for a in path:
            mod = getattr(mod, a)
        if is_param:
            mod._parameters[leaf] = nn.Parameter(new, requires_grad=False)
        else:
            mod._buffers[leaf] = new
        fixed += 1

    print(f"[Rebels JE] materialized {fixed} meta tensors from GGUF"
          + (f" ({skipped} skipped by filter)" if skipped else "") + ".", flush=True)
    if missing and not strict:
        print(f"[Rebels JE] WARNING: {len(missing)} tensors left unresolved "
              f"(non-strict): {missing[:6]}", flush=True)
    if missing and strict:
        raise RuntimeError(
            f"[Rebels JE] {len(missing)} model weights are still empty (meta) and "
            f"could not be located in the GGUF: {missing[:8]}. The GGUF may be "
            f"missing these tensors -- re-check the quantization export.")


class RebelsJE_DiTLoader:
    CATEGORY = CAT
    RETURN_TYPES = ("JOYECHO_GENERATOR",)
    RETURN_NAMES = ("generator",)
    FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("JOYECHO_CONFIG",),
            "dit_gguf": ("STRING", {"default": ""}),
            "video_height": ("INT", {"default": 736}), "video_width": ("INT", {"default": 1280})}}
    def run(self, config, dit_gguf, video_height, video_width):
        key = _cache_key("dit", dit_gguf, video_height, video_width)
        cached = _cache_get("dit", key)
        if cached is not None:
            return cached
        dtype = torch.bfloat16
        _SWAP_MAP.clear()  # ids are only valid for THIS build
        entries = _gguf_entries(dit_gguf)
        consumed = set()
        builder = Builder(
            model_class_configurator=LTXModelConfigurator,
            model_path=dit_gguf,
            model_sd_ops=None,
            module_ops=_dit_module_ops(entries, consumed, dtype),
            model_loader=_GGUFDiTLoader(config, entries, consumed, dtype),
        )
        transformer = builder.build(device=torch.device("cpu"), dtype=dtype)
        gen = LTX2DiffusionWrapper(model=X0Model(transformer), video_height=video_height, video_width=video_width)
        gen.eval()
        # Sweep the FULL wrapper (not just the transformer) so anything the
        # configurator or wrapper registered late gets real weights too.
        _materialize_meta(gen, entries, consumed, dtype)
        # Fix stale direct references (e.g. args-preprocessor patchify_proj)
        # that still point at pre-swap meta modules.
        _rebind_swapped(gen)
        _SWAP_MAP.clear()
        return _cache_put("dit", key, (gen,))

class RebelsJE_TextEncoder:
    CATEGORY = CAT
    RETURN_TYPES = ("JOYECHO_TEXTENC",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("JOYECHO_CONFIG",),
            "gemma_path": ("STRING", {"default": ""}),
            "gemma_format": (["our_fp8", "bf16"], {"default": "our_fp8"}),
            "connector_path": ("STRING", {"default": ""}),
            "low_vram": ("BOOLEAN", {"default": True})}}
    def run(self, config, gemma_path, gemma_format, connector_path, low_vram):
        from ltx_core.text_encoders.gemma import GemmaTextEncoderConfigurator, GEMMA_MODEL_OPS, module_ops_from_gemma_root
        from ltx_core.utils import find_matching_file
        from pathlib import Path

        key = _cache_key("text_encoder", gemma_path, gemma_format, connector_path, low_vram)
        cached = _cache_get("text_encoder", key)
        if cached is not None:
            return cached

        dtype = torch.bfloat16
        dev = torch.device("cpu") if low_vram else _dev()
        
        # --- SINGLE FILE INTAKE PATCH START ---
        gemma_path_str = str(gemma_path)
        is_gguf = gemma_path_str.lower().endswith(".gguf")
        if os.path.isfile(gemma_path_str):
            parent_dir = os.path.dirname(gemma_path_str)
            temp_folder = os.path.join(parent_dir, ".gemma_virtual_folder")
            # The sidecars are STAGED into .gemma_virtual_folder next to the
            # model. If that folder can't be written - a read-only or
            # restricted mount, common for /media auto-mounts on Linux - the
            # sidecars silently never land and the load fails with "sidecar
            # files missing" even though the source files exist. Verify the
            # folder is writable; if not, stage under the system temp dir
            # instead (keyed to the model path so it is stable across runs).
            def _writable(d):
                try:
                    os.makedirs(d, exist_ok=True)
                    _p = os.path.join(d, ".write_test")
                    with open(_p, "w") as _fh:
                        _fh.write("ok")
                    os.remove(_p)
                    return True
                except OSError:
                    return False
            if not _writable(temp_folder):
                import tempfile, hashlib
                _key = hashlib.sha1(os.path.abspath(gemma_path_str).encode()).hexdigest()[:12]
                temp_folder = os.path.join(tempfile.gettempdir(), "joyecho_gemma_" + _key)
                os.makedirs(temp_folder, exist_ok=True)
                print(f"[Rebels JE] model folder '{parent_dir}' is not writable "
                      f"(read-only mount?); staging Gemma sidecars in "
                      f"'{temp_folder}' instead.", flush=True)
            
            temp_model = os.path.join(temp_folder, "model.safetensors")
            if os.path.exists(temp_model):
                try: os.remove(temp_model)
                except OSError: pass
            
            # GGUF gemma: do NOT link the weights as model.safetensors (the
            # builder would try to parse a GGUF as safetensors). The virtual
            # folder only carries the HF sidecars; weights come from the GGUF.
            if not is_gguf:
                try: os.link(gemma_path_str, temp_model)
                except OSError:
                    import shutil
                    shutil.copyfile(gemma_path_str, temp_model)
                
            # Gemma needs its HF sidecar files (tokenizer + config jsons) next to
            # the weights. A single-file fp8 download has none of them, so we search
            # several places, in order: the weights' own folder, a 'gemma_assets' or
            # 'gemma' subfolder beside them, and a 'gemma_assets' folder bundled in
            # this node pack (so they can ship with the pack).
            _node_dir = os.path.dirname(os.path.abspath(__file__))
            sidecar_sources = [
                parent_dir,
                os.path.join(parent_dir, "gemma_assets"),
                os.path.join(parent_dir, "gemma"),
                os.path.join(_node_dir, "gemma_assets"),
            ]
            sidecar_files = ["tokenizer.model", "tokenizer_config.json", "config.json",
                             "special_tokens_map.json", "preprocessor_config.json"]
            def _normalize_gemma_config(path):
                """Load ANY Gemma-3 config.json (our bundled text-only one OR the
                multimodal one from google/gemma-3-12b-it that users often
                download) and return the text-only dict this pipeline needs,
                with every field JD's encoder configurator reads guaranteed
                present. Returns None if the file isn't a Gemma-3 config at all."""
                import json as _json
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        cj = _json.load(fh)
                except Exception:
                    return None
                if "gemma3" not in str(cj.get("model_type", "")) and "text_config" not in cj:
                    return None
                # multimodal google layout -> pull the nested text config
                if "text_config" in cj and isinstance(cj["text_config"], dict):
                    cj = dict(cj["text_config"])
                cj["model_type"] = "gemma3_text"
                cj.setdefault("architectures", ["Gemma3ForCausalLM"])
                # fields the encoder configurator / rotary init require:
                rs = cj.get("rope_scaling")
                if not isinstance(rs, dict):
                    rs = {"rope_type": "linear", "factor": 8.0}
                if "rope_type" not in rs:
                    rs["rope_type"] = rs.get("type", "linear")
                cj["rope_scaling"] = rs
                cj.setdefault("rope_local_base_freq", 10000.0)
                cj.setdefault("rope_theta", 1000000.0)
                return cj

            for t_file in sidecar_files:
                dst = os.path.join(temp_folder, t_file)
                if os.path.exists(dst):
                    continue
                for srcdir in sidecar_sources:
                    src = os.path.join(srcdir, t_file)
                    if os.path.exists(src):
                        if t_file == "config.json":
                            cj = _normalize_gemma_config(src)
                            if cj is None:
                                print(f"[Rebels JE] skipping {src}: not a Gemma-3 "
                                      f"config.", flush=True)
                                continue
                            import json as _json
                            with open(dst, "w", encoding="utf-8") as fh:
                                _json.dump(cj, fh, indent=2)
                            if "text_config" not in cj:
                                pass  # already text-only
                            print(f"[Rebels JE] config.json normalized from {src} "
                                  f"(text-only layout, rotary fields ensured).", flush=True)
                            break
                        try: os.link(src, dst)
                        except OSError:
                            import shutil
                            shutil.copyfile(src, dst)
                        break

            # Fail with a clear, actionable message instead of a cryptic one later.
            missing = [f for f in sidecar_files
                       if not os.path.exists(os.path.join(temp_folder, f))]
            if missing:
                # Report exactly which source dirs were searched and whether
                # each missing file was actually found in any of them - turns
                # "still missing after I added it" into a self-diagnosis.
                diag = []
                for f in missing:
                    where = [d for d in sidecar_sources if os.path.exists(os.path.join(d, f))]
                    diag.append(f"    {f}: " + ("FOUND in " + "; ".join(where) +
                                " but could not be staged into " + temp_folder +
                                " (folder not writable?)" if where else
                                "not found in any searched location"))
                raise FileNotFoundError(
                    "Gemma sidecar files missing: " + ", ".join(missing) + ".\n"
                    + "\n".join(diag) + "\n"
                    "Searched, in order:\n    " + "\n    ".join(sidecar_sources) + "\n"
                    "Staging folder: " + temp_folder + "\n"
                    "If a file shows FOUND-but-could-not-stage, the staging folder is "
                    "read-only - the pack falls back to a temp dir automatically, so "
                    "update the pack. If a file is 'not found in any searched location', "
                    "your pack's gemma_assets folder is incomplete: update to the latest "
                    "patch (it now bundles gemma_assets), OR copy the six sidecar files "
                    "(config.json, preprocessor_config.json, processor_config.json, "
                    "special_tokens_map.json, tokenizer_config.json, tokenizer.model) "
                    "directly into the same folder as your Gemma weights. Do NOT pull "
                    "config.json from the google repo - that is the multimodal variant "
                    "and is NOT compatible with this pipeline."
                )
                        
            model_folder = Path(temp_folder)
            gemma_op_path = str(temp_folder)
        else:
            model_folder = find_matching_file(gemma_path, "model*.safetensors").parent
            gemma_op_path = gemma_path
        # --- SINGLE FILE INTAKE PATCH END ---

        weight_paths = (_GGUF_MARKER,) if is_gguf else tuple(str(p) for p in model_folder.rglob("*.safetensors"))
        te_builder = Builder(
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_path=weight_paths,
            module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(gemma_op_path)),
            model_loader=_CfgLoader(config, map_gemma=True),
        )
        text_encoder = te_builder.build(device=dev, dtype=dtype)
        if is_gguf:
            _swap_gemma_gguf(text_encoder, gemma_path_str, dtype)
        elif gemma_format == "our_fp8":
            _swap_gemma_fp8(text_encoder, str(model_folder), dtype)

        # META-TENSOR FIX: the text-only Gemma file has no vision_tower /
        # multi_modal_projector / lm_head, so those modules stay on the meta
        # device and poison model.device (vision_tower is the first registered
        # param). base_encoder.encode() then builds input_ids/attention_mask on
        # meta and dies with "Cannot copy out of meta tensor". None are used for
        # text encoding -- drop them so the first real parameter (language_model)
        # defines the device. (StagedJE does this after calling .run(); doing it
        # here fixes the discrete multishot path too. Idempotent: StagedJE's own
        # strip becomes a no-op.)
        try:
            gm = getattr(text_encoder, "model", None)
            if gm is not None:
                inner = getattr(gm, "model", None)
                for parent, attr in ((inner, "vision_tower"),
                                     (inner, "multi_modal_projector"),
                                     (gm, "lm_head")):
                    if parent is not None and getattr(parent, attr, None) is not None:
                        try: setattr(parent, attr, None)
                        except Exception: pass
                try:
                    print(f"[Rebels JE] text-encoder device after meta-strip = "
                          f"{next(gm.parameters()).device}", flush=True)
                except StopIteration:
                    pass
        except Exception as e:
            print(f"[Rebels JE] meta-strip skipped: {e}", flush=True)


        ep_builder = Builder(
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_path=connector_path, model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            model_loader=_CfgLoader(config, map_gemma=False),
        )
        embeddings_processor = ep_builder.build(device=dev, dtype=dtype)

        # DEVICE UNIFY: the wrapper's forward feeds Gemma hidden_states straight
        # into embeddings_processor with no device move (it never uses
        # self.device). A GGUF-swapped Gemma effectively runs on CPU (packed
        # layers stream per-layer), so hidden_states land on CPU while the
        # connector was built on `dev` (cuda when low_vram=False) -> a
        # cpu-vs-cuda addmm crash in feature_extractor. Pin embeddings_processor
        # to the text encoder's ACTUAL device so both halves match. (StagedJE
        # unifies both halves the same way after calling .run().)
        try:
            _te_dev = next(text_encoder.model.parameters()).device
            embeddings_processor.to(_te_dev)
            print(f"[Rebels JE] embeddings_processor pinned to {_te_dev} "
                  f"(matches text encoder).", flush=True)
        except StopIteration:
            _te_dev = _dev()

        wrapper = GemmaTextEncoderWrapper(text_encoder=text_encoder, embeddings_processor=embeddings_processor,
                                          device=_te_dev, dtype=dtype)
        return _cache_put("text_encoder", key, (wrapper,))

class RebelsJE_VAELoader:
    CATEGORY = CAT
    RETURN_TYPES = ("JOYECHO_VVAE", "JOYECHO_AVAE", "INT")
    RETURN_NAMES = ("video_vae", "audio_vae", "audio_sample_rate")
    FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("JOYECHO_CONFIG",),
            "video_vae_path": ("STRING", {"default": ""}),
            "audio_vae_path": ("STRING", {"default": ""}),
            "vocoder_path": ("STRING", {"default": ""}),
            "with_encoders": ("BOOLEAN", {"default": True})}}
    def _build(self, cfg, configurator, sd_ops, path):
        return Builder(model_class_configurator=configurator, model_path=path,
                       model_sd_ops=sd_ops, model_loader=_CfgLoader(cfg)).build(
                       device=torch.device("cpu"), dtype=torch.bfloat16)
    def run(self, config, video_vae_path, audio_vae_path, vocoder_path, with_encoders):
        key = _cache_key("vae", video_vae_path, audio_vae_path, vocoder_path, with_encoders)
        cached = _cache_get("vae", key)
        if cached is not None:
            return cached
        dtype = torch.bfloat16
        v_dec = self._build(config, VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER, video_vae_path)
        a_dec = self._build(config, AudioDecoderConfigurator, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, audio_vae_path)
        voc   = self._build(config, VocoderConfigurator, VOCODER_COMFY_KEYS_FILTER, vocoder_path)
        v_enc = self._build(config, VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER, video_vae_path) if with_encoders else None
        a_enc = self._build(config, AudioEncoderConfigurator, AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, audio_vae_path) if with_encoders else None
        video_vae = VideoVAEWrapper(encoder=v_enc, decoder=v_dec, device=_dev(), dtype=dtype)
        audio_vae = AudioVAEWrapper(encoder=a_enc, decoder=a_dec, vocoder=voc, device=_dev(), dtype=dtype)
        video_vae.eval()
        audio_vae.eval()
        sr = audio_vae.get_output_sample_rate() or 24000
        return _cache_put("vae", key, (video_vae, audio_vae, sr))

class RebelsJE_Assemble:
    CATEGORY = CAT
    RETURN_TYPES = ("JOYECHO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"generator": ("JOYECHO_GENERATOR",), "text_encoder": ("JOYECHO_TEXTENC",),
                             "video_vae": ("JOYECHO_VVAE",), "audio_vae": ("JOYECHO_AVAE",),
                             "audio_sample_rate": ("INT", {"default": 24000})}}
    def run(self, generator, text_encoder, video_vae, audio_vae, audio_sample_rate):
        model = {"text_encoder": text_encoder, "generator": generator, "video_vae": video_vae,
                 "audio_vae": audio_vae, "audio_sample_rate": audio_sample_rate,
                 "device": _dev(), "dtype": torch.bfloat16}
        return (model,)

NODE_CLASS_MAPPINGS = {
    "RebelsJE_Config": RebelsJE_Config,
    "RebelsJE_DiTLoader": RebelsJE_DiTLoader,
    "RebelsJE_TextEncoder": RebelsJE_TextEncoder,
    "RebelsJE_VAELoader": RebelsJE_VAELoader,
    "RebelsJE_Assemble": RebelsJE_Assemble,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsJE_Config": "Rebels JE • Config",
    "RebelsJE_DiTLoader": "Rebels JE • DiT GGUF Loader (UNet)",
    "RebelsJE_TextEncoder": "Rebels JE • Text Encoder (Gemma fp8 + Connector)",
    "RebelsJE_VAELoader": "Rebels JE • VAE Loader (video+audio)",
    "RebelsJE_Assemble": "Rebels JE • Assemble Model",
          }
