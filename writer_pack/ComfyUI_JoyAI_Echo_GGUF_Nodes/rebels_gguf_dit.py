"""
rebels_gguf_dit.py — GGUF loader that builds JoyAI-Echo's REAL architecture.

Drop-in replacement for UnetLoaderGGUF on the DiT. ComfyUI's stock loader guesses
a generic LTX config (scale_shift_table=6, connectors=3840) and the shapes don't
match JoyAI-Echo (scale_shift_table=9, connectors=4096/2048). This node builds the
network with JoyAI's own configurator fed the real config, then loads the GGUF
(Linears stay quantized via GGUFLinear, so 41GB bf16 never materializes).

Outputs a MODEL — wire it exactly where UnetLoaderGGUF was.

Place in the ComfyUI_JoyAI_Echo repo root and register (see __init__ patch).
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf
from gguf import GGUFReader, GGMLQuantizationType as QT
import folder_paths

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer import LTXModelConfigurator, X0Model
from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper

CAT = "Rebels/JoyAI-Echo"

# ---- config: full dict from a config.json or the checkpoint HEADER (no weights) ----
def _full_config(src):
    src = src.strip().strip('"')
    if src.lower().endswith(".json"):
        with open(src, "r", encoding="utf-8") as f:
            return json.load(f)
    from safetensors import safe_open
    with safe_open(src, framework="pt") as f:
        meta = f.metadata() or {}
    if "config" not in meta:
        raise ValueError(f"No 'config' in metadata of {src}. Point config_source at the "
                         f"checkpoint (.safetensors) or a config.json.")
    return json.loads(meta["config"])

# ---- gguf ----
def _gguf_entries(path):
    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        out[t.name] = {"data": np.asarray(t.data), "qtype": t.tensor_type,
                       "shape": tuple(int(d) for d in reversed(t.shape))}
    return out

def _dequant(entry, dtype):
    deq = gguf.quants.dequantize(entry["data"], entry["qtype"]).astype(np.float32)
    return torch.from_numpy(deq.reshape(entry["shape"])).to(dtype)

class GGUFLinear(nn.Module):
    def __init__(self, entry, bias=None, compute_dtype=torch.bfloat16):
        super().__init__()
        self.qtype_value = int(entry["qtype"])
        self.weight_shape = tuple(entry["shape"])
        self.register_buffer("qweight", torch.from_numpy(entry["data"].copy()))
        self.bias = nn.Parameter(bias.to(compute_dtype), requires_grad=False) if bias is not None else None
    def forward(self, x):
        raw = self.qweight.detach().cpu().numpy()
        w = torch.from_numpy(gguf.quants.dequantize(raw, QT(self.qtype_value)).astype(np.float32))
        w = w.reshape(self.weight_shape).to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def _set_sub(root, dotted, new):
    *parents, leaf = dotted.split(".")
    p = root
    for a in parents: p = getattr(p, a)
    setattr(p, leaf, new)

class _GGUFDiTLoader(SafetensorsModelStateDictLoader):
    """metadata()->config; load()->only NON-Linear tensors (dequant, small).
    Big Linear weights stay quantized in GGUFLinear (swapped via module_ops)."""
    def __init__(self, config, entries, consumed, dtype):
        super().__init__()
        self._cfg = config
        self._e = entries
        self._consumed = consumed
        self._dt = dtype
    def metadata(self, path): return self._cfg
    def load(self, paths, sd_ops=None, device=None):
        sd, size = {}, 0
        for k, e in self._e.items():
            if k in self._consumed: continue
            t = _dequant(e, self._dt)
            sd[k] = t
            size += t.numel() * t.element_size()
        return StateDict(sd=sd, device=device or torch.device("cpu"), size=size, dtype=self._dt)

def _dit_module_ops(entries, consumed, compute_dtype):
    def mutator(model):
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear) and (name + ".weight") in entries:
                bk = name + ".bias"
                bias = _dequant(entries[bk], compute_dtype) if bk in entries else None
                _set_sub(model, name, GGUFLinear(entries[name + ".weight"], bias, compute_dtype))
                consumed.add(name + ".weight")
                if bk in entries: consumed.add(bk)
        return model
    return (ModuleOps(matcher=lambda m: True, mutator=mutator),)

class _GenCarrier:
    """MODEL-typed object; JoyEcho_ModelLoader pulls .model as the generator."""
    def __init__(self, model): self.model = model

def _gguf_list():
    out = []
    for folder in ("diffusion_models", "unet"):
        try: out += [f for f in folder_paths.get_filename_list(folder) if f.lower().endswith(".gguf")]
        except Exception: pass
    return sorted(set(out)) or ["(put a .gguf in models/diffusion_models)"]

class RebelsJE_GGUF_DiT:
    CATEGORY = CAT
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (_gguf_list(),),
            "config_source": ("STRING", {"default": r"D:\JoyAI-Echo-release.safetensors",
                "tooltip": "config.json (preferred) OR the checkpoint .safetensors (HEADER read only, no weights)"}),
            "video_height": ("INT", {"default": 736}),
            "video_width": ("INT", {"default": 1280}),
        }}

    def load(self, unet_name, config_source, video_height, video_width):
        path = folder_paths.get_full_path("diffusion_models", unet_name) \
            or folder_paths.get_full_path("unet", unet_name) or unet_name
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"GGUF not found: {unet_name}")
        dtype = torch.bfloat16
        config = _full_config(config_source)
        entries = _gguf_entries(path)
        consumed = set()
        builder = Builder(
            model_class_configurator=LTXModelConfigurator,
            model_path=path,
            model_sd_ops=None,                      # JoyAI DiT GGUF keys already match JD names
            module_ops=_dit_module_ops(entries, consumed, dtype),
            model_loader=_GGUFDiTLoader(config, entries, consumed, dtype),
        )
        transformer = builder.build(device=torch.device("cpu"), dtype=dtype)   # meta + GGUFLinear swap
        gen = LTX2DiffusionWrapper(model=X0Model(transformer),
                                   video_height=video_height, video_width=video_width)
        gen.eval()
        qsummary = {}
        for e in entries.values():
            qsummary[QT(int(e["qtype"])).name] = qsummary.get(QT(int(e["qtype"])).name, 0) + 1
        print(f"[Rebels] JoyAI DiT built from real config + GGUF. qtypes: {qsummary}", flush=True)
        return (_GenCarrier(gen),)

NODE_CLASS_MAPPINGS = {"RebelsJE_GGUF_DiT": RebelsJE_GGUF_DiT}
NODE_DISPLAY_NAME_MAPPINGS = {"RebelsJE_GGUF_DiT": "Rebels JE • GGUF DiT Loader (real arch → MODEL)"}
