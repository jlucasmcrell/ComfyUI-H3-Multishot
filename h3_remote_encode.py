"""Remote text encode for MiniMax-H3: keep the 18 GB Qwen3-VL off the render card.

The TE is needed for seconds per shot and is dead weight the rest of the run -
and on 16-24 GB cards it cannot sit beside the DiT at all. This moves the encode
to another box (any ComfyUI instance with this pack loaded) over HTTP. Any
spare PC with a mid-size GPU works; avoid hosting it on a machine whose card
is already busy rendering - the encoder wants to stay resident.

WHY A CLIP PROXY AND NOT AN ENCODE NODE: the multishot samplers encode
internally - clip.tokenize(prompt, minimax_ref_items=...) then
clip.encode_from_tokens_scheduled(tokens) - per shot, including the vision path
for reference images. A node that returned CONDITIONING could not feed that.
So H3RemoteClip returns an object that answers exactly those calls by POSTing
the raw inputs (text + image tensors) to the server, which tokenizes and
encodes with a real, resident CLIP and ships the conds back.

THE SWITCH: this is opt-in twice over. Nothing imports or calls any of this
unless the H3RemoteClip node is in the graph, and the shipped workflow wires it
through H3AnySwitch (lazy A/B) with the boolean OFF - lazy means the remote
branch does not execute when off AND the local loader does not execute when on.
Flip one boolean to move the TE off the card.

SERVER: importing this module registers POST /h3multishot/encode on whatever
ComfyUI instance loads the pack. Loading a clip happens only when a request
names one, and the loaded clip stays resident keyed by name - residency is the
entire point. The route reuses the pack's own H3ClipLoaderAny so GGUF+mmproj
pairing behaves identically to a local load.

HONESTY: fails loud. No silent local fallback - a fallback would quietly load
the 18 GB the whole feature exists to avoid. If the remote is down, the error
says so and says where the switch is.
"""
import hashlib
import io
import json
import os
import urllib.error
import urllib.request

_PACK_VER = "2.7.2"
_ROUTE = "/h3multishot/encode"


# ---------------------------------------------------------------------------
# Codec: arbitrary python structure <-> (JSON skeleton, safetensors blob).
# Generic on purpose - tokenize() output and ref-item schemas can change
# upstream without breaking this, because the codec never assumes a schema.
# ---------------------------------------------------------------------------

def _pack_obj(obj):
    import torch
    tensors = {}

    def walk(o):
        if torch.is_tensor(o):
            k = "t%d" % len(tensors)
            tensors[k] = o.detach().to("cpu").contiguous()
            return {"__t__": k}
        if isinstance(o, dict):
            return {"__d__": [[walk(k), walk(v)] for k, v in o.items()]}
        if isinstance(o, list):
            return {"__l__": [walk(x) for x in o]}
        if isinstance(o, tuple):
            return {"__u__": [walk(x) for x in o]}
        if isinstance(o, (str, int, float, bool)) or o is None:
            return {"__v__": o}
        raise TypeError("h3 remote encode cannot serialise %r" % type(o))

    return walk(obj), tensors


def _unpack_obj(skel, tensors):
    def walk(s):
        if "__t__" in s:
            return tensors[s["__t__"]]
        if "__d__" in s:
            return {walk(k): walk(v) for k, v in s["__d__"]}
        if "__l__" in s:
            return [walk(x) for x in s["__l__"]]
        if "__u__" in s:
            return tuple(walk(x) for x in s["__u__"])
        return s["__v__"]

    return walk(skel)


def _to_bytes(skel, tensors):
    from safetensors.torch import save
    meta = {"skeleton": json.dumps(skel), "pack": _PACK_VER}
    if not tensors:                      # safetensors refuses an empty dict
        import torch
        tensors = {"__empty__": torch.zeros(1)}
    return save(tensors, metadata=meta)


def _from_bytes(blob):
    from safetensors.torch import load
    tensors = load(blob)
    # safe_open only takes paths; the header is length-prefixed JSON, so read
    # the metadata straight out of the bytes instead of a temp file.
    hlen = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8:8 + hlen])
    meta = header.get("__metadata__") or {}
    skel = json.loads(meta["skeleton"])
    return _unpack_obj(skel, tensors)


# ---------------------------------------------------------------------------
# Server side: registers on import, does nothing until called.
# ---------------------------------------------------------------------------

_SERVER_CLIPS = {}          # (clip_name, clip_type) -> loaded CLIP, resident


def _server_load_clip(clip_name, clip_type):
    key = (clip_name, clip_type)
    if key in _SERVER_CLIPS:
        return _SERVER_CLIPS[key]
    from .h3_multishot_utils import H3ClipLoaderAny
    print("[H3RemoteEncode] loading %r type=%r (stays resident for future "
          "requests)" % (clip_name, clip_type), flush=True)
    clip = H3ClipLoaderAny().load(clip_name, clip_type)[0]
    _SERVER_CLIPS[key] = clip
    return clip


def _register_route():
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:
        return  # CLI / headless import - no server to register on

    ps = getattr(PromptServer, "instance", None)
    if ps is None:
        return

    @ps.routes.post(_ROUTE)
    async def h3_remote_encode(request):  # noqa: ANN001
        import asyncio
        body = await request.read()
        try:
            req = _from_bytes(body)
            clip_name = req["clip_name"]
            clip_type = req.get("clip_type", "minimax")
            calls = req["calls"]         # list of tokenize-kwarg dicts

            def run():
                clip = _server_load_clip(clip_name, clip_type)
                out = []
                for kw in calls:
                    prompt = kw.pop("__prompt__")
                    tokens = clip.tokenize(prompt, **kw)
                    out.append(clip.encode_from_tokens_scheduled(tokens))
                return out

            # comfy is not re-entrant under its execution lock; encode requests
            # run in a worker thread so a render on THIS box is not blocked.
            conds = await asyncio.get_event_loop().run_in_executor(None, run)
            skel, tensors = _pack_obj({"conds": conds, "pack": _PACK_VER})
            return web.Response(body=_to_bytes(skel, tensors),
                                content_type="application/octet-stream")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return web.Response(status=500, text="%s: %s" % (type(e).__name__, e))

    print("[H3RemoteEncode] serving POST %s (clips load on first request)"
          % _ROUTE, flush=True)


_register_route()


# ---------------------------------------------------------------------------
# Client side: the CLIP proxy.
# ---------------------------------------------------------------------------

class _NoOpTEModel:
    def to(self, *a, **k):
        return self


class _RemoteTEPatcher:
    """Just enough .patcher for the samplers' TE-offload housekeeping.

    The samplers probe clip.patcher.load_device to decide whether the text
    encoder shares the DiT's card and must be offloaded between shots. A
    remote encoder occupies no local memory at all, which is exactly the
    "separate device, nothing to reclaim" branch - load_device "remote"
    routes it there, and the no-op model absorbs any offload .to() calls.
    """
    load_device = "remote"
    offload_device = "remote"
    model = _NoOpTEModel()


class _RemoteClipProxy:
    """Answers the two calls the H3 samplers make, remotely.

    tokenize() only CAPTURES its arguments - tokenisation needs the model's own
    tokenizer, which lives on the server. encode_from_tokens_scheduled() ships
    the captured call and returns real conds. Anything else the caller tries to
    use fails loudly with the method name, because a proxy that half-works is
    worse than one that says what it cannot do.
    """

    def __init__(self, endpoint, clip_name, clip_type, timeout, cache_dir):
        self._endpoint = endpoint.rstrip("/")
        self._clip_name = clip_name
        self._clip_type = clip_type
        self._timeout = timeout
        self._cache_dir = cache_dir
        self.patcher = _RemoteTEPatcher()

    # -- the supported surface ------------------------------------------------
    def tokenize(self, prompt, **kwargs):
        call = dict(kwargs)
        call["__prompt__"] = prompt
        return {"__h3_remote_call__": call}

    def encode_from_tokens_scheduled(self, tokens):
        call = tokens.get("__h3_remote_call__") if isinstance(tokens, dict) \
            else None
        if call is None:
            raise RuntimeError(
                "H3RemoteClip received tokens it did not produce. The remote "
                "proxy only works with nodes that call clip.tokenize() and "
                "clip.encode_from_tokens_scheduled() as a pair - the H3 "
                "multishot samplers do. For anything else, switch back to the "
                "local CLIP loader.")
        return self._post([call])[0]

    # -- transport ------------------------------------------------------------
    def _post(self, calls):
        skel, tensors = _pack_obj({"clip_name": self._clip_name,
                                   "clip_type": self._clip_type,
                                   "calls": calls})
        body = _to_bytes(skel, tensors)

        # Key on CANONICAL content, never on the serialized blob: safetensors'
        # header is rust-HashMap-ordered, so the same request serializes to
        # different bytes call-to-call and blob-hashing misses every time
        # (measured: two cache files for one identical prompt).
        h = hashlib.sha256()
        h.update((self._endpoint + "|" + self._clip_name + "|"
                  + self._clip_type).encode())
        h.update(json.dumps(skel, sort_keys=True).encode())
        for name in sorted(tensors):
            h.update(name.encode())
            h.update(tensors[name].numpy().tobytes())
        key = h.hexdigest()[:16]
        cpath = os.path.join(self._cache_dir, "remote_%s.bin" % key)
        if os.path.isfile(cpath):
            with open(cpath, "rb") as f:
                print("[H3RemoteClip] cond cache HIT (%s) - no network, "
                      "no encode." % os.path.basename(cpath), flush=True)
                return _from_bytes(f.read())["conds"]

        url = self._endpoint + _ROUTE
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                blob = r.read()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:400]
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                "Remote encode failed on %s (HTTP %s). Server said: %s\n"
                "Check the remote box loaded this pack version and has "
                "%r in models/text_encoders." % (url, e.code, detail,
                                                 self._clip_name)) from e
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not reach the remote encoder at %s (%s). Either start "
                "ComfyUI with this pack on that box, or flip the encode "
                "switch back to the local CLIP loader." % (url, e)) from e

        os.makedirs(self._cache_dir, exist_ok=True)
        with open(cpath, "wb") as f:
            f.write(blob)
        return _from_bytes(blob)["conds"]

    # -- everything else fails loudly ------------------------------------------
    def __getattr__(self, name):
        raise AttributeError(
            "H3RemoteClip does not implement CLIP.%s(). The remote proxy "
            "covers tokenize/encode_from_tokens_scheduled (what the H3 "
            "multishot samplers use). Use the local CLIP loader for nodes "
            "that need more." % name)


class H3RemoteTextEncoderClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint": ("STRING", {
                    "default": "http://OTHER-PC:8188",
                    "tooltip": "The ComfyUI instance that will hold the text "
                               "encoder resident and serve encodes. It must "
                               "have this node pack installed (the route "
                               "registers on load) and the encoder file in "
                               "its models/text_encoders.",
                }),
                "clip_name": ("STRING", {
                    "default": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    "tooltip": "Encoder filename ON THE REMOTE BOX. GGUF "
                               "encoders auto-pair their -mmproj exactly as a "
                               "local load would.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 600, "min": 30, "max": 86400,
                    "tooltip": "First request may include the remote box "
                               "loading an 18 GB encoder from disk.",
                }),
                "clip_type": ("STRING", {
                    "default": "minimax",
                    "tooltip": "CLIP type as core's CLIPLoader names it. "
                               "'minimax' for H3. A STRING rather than a "
                               "dropdown because the valid list lives on the "
                               "remote box.",
                }),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "make"
    CATEGORY = "loaders/minimax"
    DESCRIPTION = ("A CLIP whose encode runs on another machine, so the 18 GB "
                   "text encoder never loads on this card. Wire through "
                   "H3 Any Switch against the local loader; identical "
                   "requests are served from a local disk cache without "
                   "touching the network.")

    def make(self, endpoint, clip_name, timeout_seconds, clip_type="minimax"):
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "cond_cache")
        self._preflight(endpoint)
        return (_RemoteClipProxy(endpoint, clip_name, clip_type,
                                 int(timeout_seconds), cache),)

    @staticmethod
    def _preflight(endpoint):
        """Fail here, not at the first encode.

        Without this the run reaches the sampler before anyone finds out the
        box is unreachable - and on a writer-driven graph that is after the
        LLM has spent minutes producing a script that is then thrown away
        (reported 2026-08-21: three attempts, ~7 minutes of writer time, the
        placeholder endpoint never edited). Two seconds of connect timeout
        here turns that into an instant, obvious failure.
        """
        url = (endpoint or "").rstrip("/")
        if not url:
            raise RuntimeError(
                "H3 Remote Text Encoder: the endpoint is empty. Put in the "
                "address of the ComfyUI PC that will serve encodes, e.g. "
                "http://your-other-pc:8188 . It needs this pack installed and "
                "the encoder file in its models/text_encoders. (Or turn "
                "remote_encoder off on the H3 Studio Switches panel to "
                "encode on this machine.)")
        if "OTHER-PC" in url.upper():
            raise RuntimeError(
                "H3 Remote Text Encoder is still set to the placeholder "
                "address %s. Replace it with the address of the ComfyUI PC "
                "that will serve encodes - it needs this pack installed (the "
                "route registers on load) and the encoder file in its "
                "models/text_encoders. (Or turn remote_encoder off on the H3 "
                "Studio Switches panel to encode on this machine.)" % url)
        try:
            req = urllib.request.Request(url + "/system_stats",
                                         headers={"Accept": "application/json"})
            urllib.request.urlopen(req, timeout=5).read(1)
        except Exception as e:
            raise RuntimeError(
                "H3 Remote Text Encoder cannot reach %s (%s). Check that "
                "ComfyUI is running on that box with this pack installed, and "
                "that the address and port are right - a hostname that does "
                "not resolve gives exactly this error, so try the IP. (Or "
                "turn remote_encoder off on the H3 Studio Switches panel to "
                "encode on this machine.)" % (url, e)) from e


NODE_CLASS_MAPPINGS = {"H3RemoteTextEncoderClip": H3RemoteTextEncoderClip}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RemoteTextEncoderClip": "H3 Remote Text Encoder (CLIP over LAN)"}
