"""MiniMax-H3 Multishot / Seamless Chain node pack.

Every optional module is imported defensively. One module failing (a
missing third-party dependency, a ComfyUI API change) must not take the
whole pack down with it - otherwise every node vanishes from the menu at
once and saved workflows open as a wall of red boxes.
"""
import logging

from .h3_multishot_utils import (NODE_CLASS_MAPPINGS,
                                 NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "WEB_DIRECTORY"]


def _merge(modname):
    try:
        mod = __import__(f"{__name__}.{modname}", fromlist=["*"])
    except Exception as e:                                # pragma: no cover
        logging.warning("[H3-Multishot] %s not loaded (%s) - its nodes are "
                        "unavailable; the rest of the pack still works",
                        modname, e)
        return
    NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))


for _m in ("h3_keyframes",       # keyframe anchors
           "h3_cartridge",       # portable character cartridges
           "h3_ref_folder",      # reference-folder picker
           "h3_advanced",        # advanced sampling helpers
           "h3_lora_stack",      # H3LoraStack
           "h3_episode_tools",   # StudioControls / StudioSwitches / AnySwitch
           "h3_avbank_probe",   # AV bank diagnostics
           "h3_chain_normalize",  # post-chain texture/colour leveller
           "rift_engine_script",  # RiftEngineScript (H3 + LTX-2.5)
           "rift_prompt_source",  # RiftPromptSource (txt briefs + json)
           "rift_script_picker",  # RiftScriptPicker + speaker stash
           "rift_writer_unload",  # adds a free-VRAM switch to the writer
           "h3_remote_encode",   # CLIP-over-LAN + the /encode server route
           "h3_tae_decode",     # 9 MB draft decode for seed hunts
           "h3_speed_boosters",  # switch panel for optional accelerators
           "h3_extend",          # H3ExtendTake: windows from a take length
           "h3_retake",          # H3Retake: redo one time window of a finished clip
           "ltx25_multishot",
           "h3_qgraft"):       # H3QGraftPatch: donor attention-envelope graft (experimental)   # LTX-2.5 multishot sampler (AV-extend joins)
    _merge(_m)

# Teaches ComfyUI-GGUF the minimax_h3 architecture, in memory, at startup.
# apply_gguf_arch_patch.py is the on-disk fallback for installs where this
# import cannot reach ComfyUI-GGUF. Harmless when GGUF is not installed.
try:
    from . import h3_ltx_patches        # noqa: F401  (LTX x1.5 upsampler, partial-load fix)
except Exception as _e:                                   # pragma: no cover
    logging.info("[H3-Multishot] LTX upsampler patch skipped (%s)", _e)

try:
    from . import h3_gguf_arch          # noqa: F401
except Exception as _e:                                   # pragma: no cover
    logging.info("[H3-Multishot] GGUF arch hook not installed (%s). "
                 "Safetensors checkpoints are unaffected; for GGUF, install "
                 "ComfyUI-GGUF and run apply_gguf_arch_patch.py once.", _e)
