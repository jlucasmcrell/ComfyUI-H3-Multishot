"""Give the LLM prompt writer an "unload after writing" switch, at runtime.

A local writer and the video model want the same card, and ComfyUI's own
eviction cannot help: it frees models inside the ComfyUI process, while Ollama
is a separate process with its own allocator. Ollama's OpenAI-compatible
endpoint cannot be asked either - its request type has no `keep_alive` field
and the shim never sets one, so the parameter is silently dropped and the model
sits for the server default of five minutes. That is the whole of shot 1, with
the writer's weights still on the card.

The unload belongs ON THE WRITER, because the writer already carries the
`base_url` and `model_name` it needs. A separate node would duplicate both,
and a duplicate that drifts fails SILENTLY - wrong endpoint, nothing resident,
reported as success, VRAM still gone.

Editing the writer's own file would work on one machine and reach nobody:
users install that pack from its own repo. So this patches the class in memory
at startup instead - the same approach h3_gguf_arch.py uses for ComfyUI-GGUF.
Nothing of theirs is copied or redistributed; their installed code is wrapped
where it runs.

ATTRIBUTION: the writer node (`JoyEcho_LLMEnhance`) is RealRebelAI's, from
ComfyUI_JoyAI_Echo_GGUF_Nodes. This file adds one optional widget to it and
wraps its function. It is not their code, and it is inert when their pack is
absent.

The widget defaults to OFF, so the node behaves exactly as its author wrote it
until someone opts in.
"""
import json
import logging
import threading
import time
import urllib.request

TARGETS = ("JoyEcho_LLMEnhance",)
WIDGET = "unload_model_after"
_PATCHED = set()


def _native_root(base_url):
    """Ollama's native API root from the OpenAI-compatible URL."""
    u = (base_url or "").strip().rstrip("/")
    return u[:-3] if u.endswith("/v1") else u


def _resident(root, timeout=10.0):
    """Returns a list, or None meaning 'no native API here'.

    A timeout or a connection reset used to return None too, which the caller
    reads as "hosted endpoint, nothing on this card to free" - so it printed
    that and returned WITHOUT attempting the unload, leaving the writer's
    weights resident and the DiT streaming around them. Distinguish "the
    server answered and has no native API" (real None) from "the call failed"
    (unknown - say so, and let the caller try the unload anyway).
    """
    try:
        with urllib.request.urlopen(root + "/api/ps", timeout=timeout) as r:
            return [(m.get("name") or m.get("model") or "?",
                     int(m.get("size_vram") or 0))
                    for m in json.load(r).get("models", [])]
    except urllib.error.HTTPError:
        return None                      # answered; this API is not there
    except Exception as _e:              # noqa: BLE001  timeout / refused / DNS
        print("[H3 Multishot] writer unload: could not query %s/api/ps (%s) - "
              "attempting the unload anyway rather than assuming there is "
              "nothing to free." % (root, type(_e).__name__), flush=True)
        return "unknown"


def _unload(base_url, model_name, timeout=10.0):
    """Free the writer's model. Never raises - a render must not die for this."""
    root = _native_root(base_url)
    if not root:
        return
    before = _resident(root, timeout)
    if before is None:
        print("[H3 Multishot] writer unload: %s has no native API (hosted endpoint?) "
              "- nothing on this card to free." % root, flush=True)
        return
    if before == "unknown":
        before = []                      # proceed to the unload blind
    elif not before:
        print("[H3 Multishot] writer unload: nothing resident.", flush=True)
        return
    wanted = (model_name or "").strip()
    targets = [wanted] if wanted else [n for n, _ in before]
    for name in targets:
        try:
            req = urllib.request.Request(
                root + "/api/generate",
                json.dumps({"model": name, "prompt": "",
                            "keep_alive": 0}).encode(),
                {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                r.read()
        except Exception as e:
            print("[H3 Multishot] writer unload: %s did not unload (%s) - continuing."
                  % (name, e), flush=True)
    # the unload call returns BEFORE the eviction completes; reading /api/ps
    # immediately reports the model still resident and makes a success look
    # like a failure (measured). Poll briefly for the truth.
    after = _resident(root, timeout) or []
    deadline = time.monotonic() + min(10.0, timeout)
    while any(n in targets for n, _ in after) and time.monotonic() < deadline:
        time.sleep(0.25)
        after = _resident(root, timeout) or []
    freed = (sum(v for _, v in before) - sum(v for _, v in after)) / 1e9
    still = ", ".join(n for n, _ in after)
    print("[H3 Multishot] writer unload: freed %.1f GB from %s%s"
          % (freed, root, ("; still resident: " + still) if still
             else "; card is clear"), flush=True)


def _patch(cls, name):
    if name in _PATCHED or getattr(cls, "_rift_unload_patched", False):
        return False
    fn_name = getattr(cls, "FUNCTION", None)
    orig_fn = getattr(cls, fn_name, None) if fn_name else None
    orig_inputs = getattr(cls, "INPUT_TYPES", None)
    if orig_fn is None or orig_inputs is None:
        return False

    @classmethod
    def INPUT_TYPES(_cls):
        spec = orig_inputs()
        opt = spec.setdefault("optional", {})
        # appended LAST: saved graphs map widgets_values by index, and a widget
        # inserted mid-list shifts every value after it on the next load
        opt[WIDGET] = ("BOOLEAN", {
            "default": False,
            "label_on": "free VRAM after writing", "label_off": "off",
            "tooltip": "After the script is written, unload this model from "
                       "the server so the video model has the card. Uses this "
                       "node's own base_url and model_name. Local Ollama only "
                       "- harmless on a hosted endpoint. Added by "
                       "ComfyUI-H3-Multishot."})
        return spec

    def wrapped(self, *a, **kw):
        do = bool(kw.pop(WIDGET, False))
        try:
            return orig_fn(self, *a, **kw)
        finally:
            # FINALLY, not on success: a writer that raises - bad JSON, a
            # refused model, a timeout - still left its weights on the card,
            # and that is exactly when you want them gone, because the next
            # thing you do is fix the prompt and queue again.
            if do:
                try:
                    _unload(kw.get("base_url", ""),
                            (kw.get("model_name_custom") or "").strip()
                            or kw.get("model_name", ""))
                except Exception as e:                   # never break a render
                    print("[H3 Multishot] writer unload skipped: %s" % e, flush=True)

    cls.INPUT_TYPES = INPUT_TYPES
    setattr(cls, fn_name, wrapped)
    cls._rift_unload_patched = True
    _PATCHED.add(name)
    print("[H3 Multishot] %s: added '%s' switch (frees the writer's VRAM before the "
          "video model loads)." % (name, WIDGET), flush=True)
    return True


def _wait_and_patch(budget_s=90.0):
    """The writer's pack may load after ours, so watch for it briefly."""
    try:
        import nodes
    except Exception:
        return
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        missing = [t for t in TARGETS if t not in _PATCHED]
        if not missing:
            return
        for t in missing:
            cls = nodes.NODE_CLASS_MAPPINGS.get(t)
            if cls is not None:
                try:
                    _patch(cls, t)
                except Exception as e:
                    logging.warning("[H3 Multishot] could not patch %s: %s", t, e)
                    _PATCHED.add(t)
        time.sleep(1.0)


threading.Thread(target=_wait_and_patch, name="rift-writer-unload",
                 daemon=True).start()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
