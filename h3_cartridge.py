# -*- coding: utf-8 -*-
"""H3 Cartridge Loader - .riftcast character cartridges for MiniMax-H3.

Reads the SAME .riftcast files the JoyEcho/LTX pipeline uses (spec v1.0:
manifest.json with name / speaker_tag / voice / refs / dna, optional accent
line, rooms, environment stills). Nothing about the format changes; only the
consumer is new. Existing cartridges load as-is.

Mapping to H3's reference system:
  refs (images)  -> up to 9 SEPARATE IMAGE outputs (a batch would silently
                    collapse to one reference - core resizes img[:1])
  voice anchor   -> AUDIO output for ref_audio_0. Wire it through
                    H3 Reference Audio (the stereo guard): the audio VAE
                    encodes [B, 2, L] and a mono clip dies deep in the model.
  dna text       -> dna_text, and pre-assembled into subject_block, a ready
                    subject_definitions fragment for ref-mode prompts:
                      <Subject 1> is NAME, DNA...
                      <Audio 1> is the voice-timbre reference for <Subject 1> (S1).
  speaker_tag / accent_line / rooms -> STRING outputs for prompt assembly.

Kept from the reference implementation: path-traversal rejection and the
sha256 integrity check on the voice anchor. LoRA entries are ignored (H3
identity comes from references, not LoRAs) and noted in the report.
"""
import hashlib
import io
import json
import os
import shutil
import zipfile

SPEC_REQUIRED = ("riftcast", "name", "speaker_tag", "voice", "refs", "dna")
MAX_SLOTS = 9


class RiftcastError(ValueError):
    pass


def _safe_name(entry_name):
    n = entry_name.replace("\\", "/")
    if n.startswith("/") or ".." in n.split("/") or (len(n) > 1 and n[1] == ":"):
        raise RiftcastError(f"unsafe path in archive: {entry_name!r}")
    return n


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(pack_path):
    with zipfile.ZipFile(pack_path) as z:
        try:
            manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        except KeyError:
            raise RiftcastError("not a riftcast: manifest.json missing")
    if "riftcast" not in manifest and "joypack" in manifest:
        manifest["riftcast"] = manifest.pop("joypack")
    for k in SPEC_REQUIRED:
        if k not in manifest:
            raise RiftcastError(f"manifest missing required key: {k}")
    return manifest


def extract_for_h3(pack_path, cache_root):
    """Unpack refs + voice + texts into a cache dir. Returns a dict of paths
    and texts. Re-extracts only when the pack is newer than the cache stamp."""
    manifest = read_manifest(pack_path)
    name = manifest["name"]
    dst = os.path.join(cache_root, name)
    stamp = os.path.join(dst, ".pack_mtime")
    mtime = str(os.path.getmtime(pack_path))
    with_stamp = ""
    if os.path.isfile(stamp):
        with io.open(stamp, encoding="utf-8") as _f:
            with_stamp = _f.read()
    if with_stamp != mtime:
        # rmtree(ignore_errors=True) silently leaves the old tree in place when
        # Windows has a handle open or a delete is still pending. The extract
        # below then writes ON TOP, the stamp is written regardless, and refs
        # are rebuilt from os.listdir() - so the previous cartridge's refs get
        # merged into this one and, because the stamp now matches, it never
        # re-extracts. Verify the removal and fall back to a fresh directory
        # rather than merging two cartridges' identities.
        shutil.rmtree(dst, ignore_errors=True)
        if os.path.isdir(dst) and os.listdir(dst):
            import tempfile
            dst = tempfile.mkdtemp(prefix=name + "_", dir=cache_root)
            stamp = os.path.join(dst, ".pack_mtime")
            print("[RiftCast] cartridge cache for %r could not be cleared "
                  "(files still held); extracting to a fresh directory %s "
                  "instead of merging with the stale one."
                  % (name, os.path.basename(dst)), flush=True)
        os.makedirs(dst, exist_ok=True)
        with zipfile.ZipFile(pack_path) as z:
            names = {_safe_name(n) for n in z.namelist()}

            def ext(member, sub):
                member = _safe_name(member)
                if member not in names:
                    raise RiftcastError(f"manifest references missing file: {member}")
                d = os.path.join(dst, sub)
                os.makedirs(d, exist_ok=True)
                p = os.path.join(d, os.path.basename(member))
                with z.open(member) as src, open(p, "wb") as f:
                    shutil.copyfileobj(src, f)
                return p

            vfile = manifest["voice"]["file"]
            vpath = ext(vfile, "voice")
            want = manifest.get("sha256", {}).get(vfile)
            if want and _sha256(vpath) != want:
                os.remove(vpath)
                raise RiftcastError("voice anchor failed sha256 verification")
            for r in manifest["refs"]:
                ext(r, "refs")
            # context managers, not bare io.open(...).write(...): a handle left
            # open inside dst is exactly what makes the NEXT run's rmtree fail
            # and merge two cartridges (see above).
            with io.open(os.path.join(dst, "dna.txt"), "w",
                         encoding="utf-8") as _f:
                _f.write(z.read(_safe_name(manifest["dna"]))
                         .decode("utf-8").strip())
            rooms = manifest.get("environment", {}).get("rooms")
            rooms_text = (z.read(_safe_name(rooms)).decode("utf-8").strip()
                          if rooms and _safe_name(rooms) in names else "")
            with io.open(os.path.join(dst, "rooms.txt"), "w",
                         encoding="utf-8") as _f:
                _f.write(rooms_text)
        with io.open(stamp, "w", encoding="utf-8") as _f:
            _f.write(mtime)

    refs_dir = os.path.join(dst, "refs")
    refs = sorted(os.path.join(refs_dir, f) for f in os.listdir(refs_dir)) \
        if os.path.isdir(refs_dir) else []
    vdir = os.path.join(dst, "voice")
    voice = sorted(os.path.join(vdir, f) for f in os.listdir(vdir))[0]
    return {
        "manifest": manifest,
        "name": name,
        "speaker_tag": manifest["speaker_tag"],
        "accent_line": manifest.get("voice", {}).get("accent_line", ""),
        "dna": io.open(os.path.join(dst, "dna.txt"), encoding="utf-8").read(),
        "rooms": io.open(os.path.join(dst, "rooms.txt"), encoding="utf-8").read(),
        "refs": refs,
        "voice": voice,
        "loras_skipped": sum(len(v or []) for v in
                             (manifest.get("loras", {}) or {}).values()),
    }


def build_subject_block(name, dna, accent_line):
    """A ready-to-paste subject_definitions fragment for ref-mode prompts."""
    dna_one = " ".join(dna.split())
    lines = [f"<Subject 1> is {name}, {dna_one}"]
    lines.append("<Audio 1> is the voice-timbre reference for <Subject 1> (S1)."
                 + (f" Delivery: {accent_line.strip()}" if accent_line.strip() else ""))
    return "\n".join(lines)


class H3CartridgeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        d = os.path.join(folder_paths.get_input_directory(), "riftcast")
        packs = sorted(f for f in os.listdir(d)
                       if f.lower().endswith(".riftcast")) if os.path.isdir(d) else []
        return {"required": {
            "cartridge": (packs or ["(no cartridges in input/riftcast/)"], {}),
            "max_refs": ("INT", {"default": MAX_SLOTS, "min": 1, "max": MAX_SLOTS}),
        }, "optional": {
            "manual_path": ("STRING", {"default": "", "tooltip":
                "Full path to any .riftcast, bypassing the dropdown."}),
        }}

    RETURN_TYPES = tuple(["IMAGE"] * MAX_SLOTS
                         + ["AUDIO", "STRING", "STRING", "STRING", "STRING",
                            "STRING", "INT", "STRING"])
    RETURN_NAMES = tuple([f"ref_{i+1}" for i in range(MAX_SLOTS)]
                         + ["voice", "subject_block", "dna_text", "speaker_tag",
                            "accent_line", "rooms_text", "ref_count", "report"])
    FUNCTION = "load"
    CATEGORY = "loaders/minimax"

    @classmethod
    def IS_CHANGED(cls, cartridge, max_refs, manual_path=""):
        p = cls._path(cartridge, manual_path)
        try:
            return f"{p}:{os.path.getmtime(p)}:{max_refs}"
        except Exception:
            return float("nan")

    @staticmethod
    def _path(cartridge, manual_path):
        if (manual_path or "").strip():
            return manual_path.strip().strip('"')
        import folder_paths
        return os.path.join(folder_paths.get_input_directory(), "riftcast", cartridge)

    @staticmethod
    def _load_image(path):
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return torch.from_numpy(
            (np.asarray(img, dtype=np.float32) / 255.0))[None,]

    @staticmethod
    def _load_audio(path, cache_dir):
        """Decode with PyAV, which ComfyUI bundles - no external ffmpeg needed.

        The cartridge voice anchors are .mp4; a subprocess-ffmpeg dependency
        would silently require ffmpeg on PATH on every rig. av 18 decodes the
        real anchors (verified on WREN: stereo, 48 kHz). Falls back to
        torchaudio for plain wav only if av is somehow absent.
        """
        import numpy as np
        import torch
        try:
            import av
            frames = []
            with av.open(path) as c:
                stream = next(s for s in c.streams if s.type == "audio")
                sr = stream.rate
                for fr in c.decode(stream):
                    a = fr.to_ndarray()
                    if a.dtype.kind == "i":
                        a = a.astype("float32") / float(1 << (8 * a.itemsize - 1))
                    frames.append(a.astype("float32"))
            wav = np.concatenate(frames, axis=-1)
            if wav.ndim == 1:
                wav = wav[None, :]
            waveform = torch.from_numpy(wav)
        except ImportError:
            import torchaudio
            waveform, sr = torchaudio.load(path)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sr}

    def load(self, cartridge, max_refs, manual_path=""):
        import folder_paths
        pack = self._path(cartridge, manual_path)
        if not os.path.isfile(pack):
            raise RiftcastError(f"cartridge not found: {pack}")
        cache_root = os.path.join(folder_paths.get_input_directory(),
                                  "riftcast", "_cache", "h3")
        c = extract_for_h3(pack, cache_root)

        images = [self._load_image(p) for p in c["refs"][:max_refs]]
        voice = self._load_audio(c["voice"],
                                 os.path.join(cache_root, c["name"], "voice"))
        subject = build_subject_block(c["name"], c["dna"], c["accent_line"])
        report = (f"{c['name']} ({c['speaker_tag']}): {len(images)} of "
                  f"{len(c['refs'])} refs, voice {os.path.basename(c['voice'])}"
                  + (f", {c['loras_skipped']} lora entr(y/ies) ignored - H3 "
                     f"identity comes from references" if c["loras_skipped"] else ""))
        print(f"[H3Cartridge] {report}", flush=True)
        out = images + [None] * (MAX_SLOTS - len(images))
        return tuple(out + [voice, subject, c["dna"], c["speaker_tag"],
                            c["accent_line"], c["rooms"], len(images), report])


NODE_CLASS_MAPPINGS = {"H3CartridgeLoader": H3CartridgeLoader}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3CartridgeLoader": "H3 Cartridge Loader (.riftcast)"}
