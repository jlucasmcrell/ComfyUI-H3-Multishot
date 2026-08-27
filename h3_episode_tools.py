# -*- coding: utf-8 -*-
"""Episode-assembly helpers for the two-stage reference pipeline.

Stage A renders block 1 with MiniMaxH3ReferenceToVideo (identity in pixels),
stage B chains blocks 2..N through H3MultishotSampler from stage A's last
frame, and the segments are joined in-graph. These three nodes are the glue:

  H3EpisodeSplit  - one pasted episode script -> stage A prompt + stage B script
  H3LastFrame    - stage A frames -> the single chain frame for start_image
  H3ConcatAV     - stage A + stage B frames/audio -> one episode
"""
import json
import math
import re

_BLOCK_SPLIT = re.compile(r"(?m)^---\s*$")


class H3EpisodeSplit:
    """Split an episode script into the stage A prompt and stage B envelope.

    Accepts either the one-line JSON envelope {"prompts": [...]} (the LPFF
    entry format) or raw blocks separated by --- lines. `bindings` is
    prepended to block 1 only - reference-image identity lines like
    "<Picture 1>, <Picture 2> are the same person (Rae)." belong with the
    ref2va stage, never in the I2V chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "script": ("STRING", {"multiline": True, "default": "", "tooltip":
                "The whole episode: either the one-line JSON envelope "
                '{"prompts": [...]} or raw blocks separated by --- lines.'}),
            "bindings": ("STRING", {"multiline": True, "default": "", "tooltip":
                "<Picture N> identity lines, prepended to block 1 only. "
                "Keep in sync with which reference images are unmuted."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("block1_prompt", "rest_script", "info")
    FUNCTION = "split"
    CATEGORY = "video/minimax"

    def split(self, script, bindings):
        text = (script or "").strip()
        if not text:
            raise ValueError("[H3EpisodeSplit] script is empty - paste the "
                             "episode envelope or --- separated blocks.")
        if text.startswith("{"):
            try:
                blocks = json.loads(text)["prompts"]
            except Exception as e:
                raise ValueError(f"[H3EpisodeSplit] script looks like JSON but "
                                 f"failed to parse: {e}")
        else:
            blocks = [b.strip() for b in _BLOCK_SPLIT.split(text) if b.strip()]
        blocks = [str(b).strip() for b in blocks if str(b).strip()]
        b = (bindings or "").strip()
        block1 = (b + "\n" + blocks[0]) if b else blocks[0]
        rest = json.dumps({"prompts": blocks[1:]}, ensure_ascii=False)
        if len(blocks) < 2:
            # Single-block scene: stage A alone is the whole render. Do not
            # raise - a one-shot scene is a legitimate thing to want, and the
            # graph supports it as soon as stage B is muted (H3ConcatAV takes
            # its B inputs as optional and passes A straight through).
            info = ("1 block -> single-shot render. MUTE the stage B group "
                    "(sampler + last-frame) with Ctrl-M; H3ConcatAV then "
                    "passes stage A through unchanged. ~15s total.")
            print(f"[H3EpisodeSplit] {info}", flush=True)
            return (block1, rest, info)
        info = (f"{len(blocks)} blocks -> stage A renders block 1, "
                f"stage B chains {len(blocks) - 1} block(s) "
                f"(~{len(blocks) * 15.1:.0f}s total at 362f/block)")
        return (block1, rest, info)


class H3LastFrame:
    """Return the last frame of a batch - the I2V chain frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", {"tooltip":
            "Stage A frames; the final frame seeds stage B's start_image."})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "take"
    CATEGORY = "image/batch"

    def take(self, images):
        return (images[-1:],)


class H3ConcatAV:
    """Concatenate two video+audio segments into one continuous take.

    match_b (default on): measure the seam and match segment B's texture to
    segment A. The two-stage chain renders A on ref2va (soft - reference
    conditioning pulls the image toward the refs' texture) and B on fl2va
    (crisper, higher micro-contrast); measured on a real take the seam was a
    +119..174% Laplacian sharpness step plus +5% luma - a visible focus
    snap. Steps parity cannot equalize two checkpoints, so B gets an
    auto-tuned gaussian (sigma searched until B's Laplacian lands on A's)
    and a luma affine, in float tensors before any encode. A's softer look
    IS the intended consumer-camcorder texture, so matching downward is the
    correct direction. Skips itself when the seam is already within 15%.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images_a": ("IMAGE",), "audio_a": ("AUDIO",),
        }, "optional": {
            "images_b": ("IMAGE", {"tooltip":
                "Stage B. Leave unconnected, or MUTE the stage B group, for a "
                "single-shot render - stage A then passes through unchanged."}),
            "audio_b": ("AUDIO",),
            "match_b": (["match_to_a", "off"], {"default": "match_to_a",
                "tooltip": "Match segment B's sharpness/tone to segment A "
                "at the seam (the two stages render on different "
                "checkpoints with different texture character)."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "concat"
    CATEGORY = "video/minimax"

    @staticmethod
    def _gray(x):
        return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])

    @classmethod
    def _lap_var(cls, x):
        import torch
        g = cls._gray(x).unsqueeze(1)
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         dtype=g.dtype, device=g.device).view(1, 1, 3, 3)
        return float(torch.nn.functional.conv2d(g, k, padding=1).var())

    @staticmethod
    def _gauss(x, sigma):
        """Separable gaussian on [T,H,W,C], returns same shape."""
        import math
        import torch
        r = max(1, int(math.ceil(3 * sigma)))
        t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
        k = torch.exp(-(t ** 2) / (2 * sigma * sigma))
        k = (k / k.sum()).view(1, 1, 1, -1)
        v = x.permute(0, 3, 1, 2)                       # [T,C,H,W]
        c = v.shape[1]
        kh = k.expand(c, 1, 1, k.shape[-1])
        v = torch.nn.functional.conv2d(v, kh, padding=(0, r), groups=c)
        kv = k.view(1, 1, -1, 1).expand(c, 1, k.shape[-1], 1)
        v = torch.nn.functional.conv2d(v, kv, padding=(r, 0), groups=c)
        return v.permute(0, 2, 3, 1)

    def _match(self, images_a, images_b):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # WINDOWED: A's tail vs B's HEAD - those are the two sides the viewer
        # actually sees touching. Sampling across all of B (the old behaviour)
        # measures B's internal average, which sits above its head because the
        # chain ratchets, so the head ends up over-blurred (measured -50..-72%
        # at the seam - a softening snap instead of a match).
        nwin = 24
        na = images_a[-min(nwin, images_a.shape[0]):].to(dev)
        nb = images_b[:min(nwin, images_b.shape[0])].to(dev)
        lap_a, lap_b = self._lap_var(na), self._lap_var(nb)
        ga, gb = self._gray(na), self._gray(nb)
        ma, sa = float(ga.mean()), float(ga.std())
        mb, sb = float(gb.mean()), float(gb.std())
        sigma = 0.0
        if lap_b > lap_a * 1.15:
            ref = nb[len(nb) // 2:len(nb) // 2 + 1]
            # CLAMPED: only accept a sigma whose result stays at or above
            # 0.9 x A's level. Undershooting the target leaves a small step;
            # overshooting produces the far more visible softening snap.
            floor = lap_a * 0.9
            best = (float("inf"), 0.0)
            s = 0.05
            while s <= 1.6 + 1e-9:
                got = self._lap_var(self._gauss(ref, s))
                if got >= floor:
                    d = abs(got - lap_a)
                    if d < best[0]:
                        best = (d, s)
                s += 0.05
            sigma = best[1]
        gain = max(0.85, min(1.15, sa / max(sb, 1e-6)))
        off = ma - mb * gain
        if sigma == 0.0 and abs(off) < 0.006 and abs(gain - 1) < 0.02:
            print(f"[H3ConcatAV] seam already matched (sharp "
                  f"{lap_b / max(lap_a, 1e-9):+.0%} vs A) - no-op", flush=True)
            return images_b
        out = torch.empty_like(images_b)
        for i in range(0, images_b.shape[0], 32):
            ch = images_b[i:i + 32].to(dev)
            if sigma > 0:
                ch = self._gauss(ch, sigma)
            ch = (ch * gain + off).clamp(0, 1)
            out[i:i + 32] = ch.to(images_b.device)
        print(f"[H3ConcatAV] matched B to A: sigma {sigma:.2f}, gain "
              f"{gain:.3f}, offset {off:+.4f} (sharp was "
              f"{lap_b / max(lap_a, 1e-9):+.0%} vs A)", flush=True)
        return out

    def concat(self, images_a, audio_a, images_b=None, audio_b=None,
               match_b="match_to_a"):
        import torch
        if images_b is None or audio_b is None:
            # Single-shot render: stage B is muted, so there is nothing to
            # join and stage A IS the finished take. Pass it through rather
            # than failing - a one-block scene is a legitimate thing to want.
            print("[H3ConcatAV] stage B absent (muted) - passing stage A "
                  "through unchanged, single-shot render.", flush=True)
            return (images_a, audio_a)
        if tuple(images_a.shape[1:3]) != tuple(images_b.shape[1:3]):
            raise ValueError(
                f"[H3ConcatAV] frame sizes differ: A "
                f"{tuple(images_a.shape[1:3])} vs B "
                f"{tuple(images_b.shape[1:3])} - both stages must render at "
                "the same width/height.")
        if match_b == "match_to_a":
            images_b = self._match(images_a, images_b)
        images = torch.cat((images_a, images_b), dim=0)
        wa, wb = audio_a["waveform"], audio_b["waveform"]
        sa = int(audio_a["sample_rate"])
        if sa != int(audio_b["sample_rate"]):
            raise ValueError(f"[H3ConcatAV] sample rates differ: {sa} vs "
                             f"{int(audio_b['sample_rate'])}")
        if wa.shape[1] != wb.shape[1]:
            # mono/stereo mismatch: upmix the narrower side
            c = max(wa.shape[1], wb.shape[1])
            wa = wa.expand(-1, c, -1) if wa.shape[1] == 1 else wa
            wb = wb.expand(-1, c, -1) if wb.shape[1] == 1 else wb
        # 40ms equal-power crossfade at the seam: the two stages are sampled
        # independently, so a butt-join puts a step discontinuity in the
        # waveform that reads as a click (confirmed by two independent audio
        # reviewers at the stage A/B boundary).
        k = min(int(sa * 0.04), wa.shape[-1], wb.shape[-1])
        if k >= 8:
            t = torch.linspace(0, 1, k, dtype=wa.dtype, device=wa.device)
            fade_out = torch.cos(t * 3.14159265 / 2)
            fade_in = torch.sin(t * 3.14159265 / 2)
            seam = wa[..., -k:] * fade_out + wb[..., :k] * fade_in
            audio = {"waveform": torch.cat((wa[..., :-k], seam, wb[..., k:]),
                                           dim=-1), "sample_rate": sa}
        else:
            audio = {"waveform": torch.cat((wa, wb), dim=-1), "sample_rate": sa}
        return (images, audio)


class H3AutoRefs:
    """Auto-select reference images per character from folders.

    The RiftCast Studio pattern (JoyEcho_RefPicker) adapted to ref2va: a refs
    root holds one subfolder per character; the prompt's descriptive prose is
    scanned for folder names as whole words (dialogue is stripped first, so a
    character merely TALKED ABOUT never matches); each matched character
    contributes up to max_per_character images, in first-mention order, up to
    the model's 9-slot cap; and the matching <Picture N> identity bindings are
    generated and prepended to the prompt automatically.
    """

    MAX_SLOTS = 9

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "max_per_character": ("INT", {"default": 3, "min": 1, "max": 9,
                "tooltip": "Images per matched character (sorted order; "
                "front/three-quarter/profile sets hold identity best)."}),
        }, "optional": {
            # 2.6.0: optional, so the node can run BEFORE the writer on the
            # premise alone (wire the scene idea here or into extra_text) -
            # then `found` can drive the writer's refs_attached and a miss
            # is known before two minutes of writing are spent.
            "prompt_text": ("STRING", {"forceInput": True, "tooltip":
                "Text scanned for character folder names (the premise, or "
                "the writer's output)."}),
            "refs_root": ("STRING", {"default": "", "tooltip":
                "Folder holding one subfolder per character. Relative paths "
                "resolve under input/. Empty = input/h3_refs/."}),
            "characters": ("STRING", {"default": "", "tooltip":
                "Explicit comma list of character folders - overrides the "
                "prompt scan entirely when set."}),
            "overrides": ("STRING", {"default": "", "tooltip":
                "Folder remaps, e.g. 'rae=rae_night' to swap a "
                "character's ref set for specific scenes. Comma-separated."}),
            "on_no_match": (["error", "no_reference"], {"default": "no_reference",
                "tooltip": "no_reference (default) = warn in the console and "
                "render WITHOUT photos when no character folder matches; "
                "the writer is told (found=false) and describes the person "
                "normally. error = stop the run instead."}),
            "enabled": ("BOOLEAN", {"forceInput": True, "tooltip":
                "Wire USE AUTO REFS here. Off = no scan, no photos, "
                "found=false, and the run continues."}),
            # 2.6.0: the writer may anonymise names (ID_A) or a script may
            # only IMPLY the character; the premise / scene idea almost always
            # names them. Wire the scene-idea text here and both are scanned.
            "extra_text": ("STRING", {"forceInput": True, "tooltip":
                "Optional second text to scan for character folder names - "
                "wire the scene idea / premise here so a writer that renames "
                "the character to ID_A still casts the right photos."}),
        }}

    # refs_batch (2.6.0, appended last so saved graphs keep their slots):
    # every picked photo in one IMAGE batch, resized to the first photo's
    # size. Wire THIS to the sampler - the per-slot outputs feed a chain of
    # core ImageBatch nodes that crash when a character has fewer photos
    # than the chain expects.
    # found (2.6.0, appended last): True when at least one photo was picked.
    # Wire it to the writer's refs_attached so the writer only points at
    # photographs that exist.
    RETURN_TYPES = tuple(["IMAGE"] * 9 + ["STRING", "STRING", "IMAGE", "BOOLEAN"])
    RETURN_NAMES = tuple([f"ref_{i+1}" for i in range(9)]
                         + ["prompt_out", "report", "refs_batch", "found"])
    FUNCTION = "pick"
    CATEGORY = "video/minimax"

    @staticmethod
    def _root(refs_root):
        import os
        import folder_paths
        r = (refs_root or "").strip() or "h3_refs"
        if not os.path.isabs(r):
            r = os.path.join(folder_paths.get_input_directory(), r)
        return r

    @classmethod
    def IS_CHANGED(cls, max_per_character, prompt_text="", refs_root="",
                   characters="", overrides="", on_no_match="no_reference",
                   extra_text="", enabled=True):
        import os
        root = cls._root(refs_root)
        sig = [str(hash((prompt_text or "") + "|" + (extra_text or ""))),
               str(max_per_character), characters, overrides, root,
               str(bool(enabled))]
        try:
            for d in sorted(os.listdir(root)):
                p = os.path.join(root, d)
                if os.path.isdir(p):
                    fs = sorted(os.listdir(p))
                    sig.append(f"{d}:{len(fs)}")
        except OSError:
            pass
        return "|".join(sig)

    def pick(self, max_per_character, prompt_text="", refs_root="",
             characters="", overrides="", on_no_match="no_reference",
             extra_text="", enabled=True):
        import os
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        if not enabled:
            return tuple([None] * self.MAX_SLOTS
                         + [prompt_text or "", "(auto refs off)", None, False])
        root = self._root(refs_root)
        try:
            dirs = sorted(d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d)))
        except OSError:
            dirs = []
        remap = {}
        for pair in (overrides or "").split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                remap[k.strip().lower()] = v.strip()

        if (characters or "").strip():
            matched = [c.strip() for c in characters.split(",") if c.strip()]
            matched = matched[:self.MAX_SLOTS]
        else:
            # strip dialogue so absent characters spoken about never match
            # (same scrub as JoyEcho_RefPicker, field-proven)
            scrub = prompt_text or ""
            try:
                _d = json.loads(scrub)
                arr = _d.get("prompts") if isinstance(_d, dict) else None
                if isinstance(arr, list) and arr:
                    scrub = " ".join(str(x) for x in arr)
            except (ValueError, TypeError):
                pass
            scrub = re.sub(r"<d>.*?</d>", " ", scrub, flags=re.S)
            scrub = re.sub(r'\\"(?:[^"\\]|\\.)*?\\"', " ", scrub)
            scrub = re.sub(r'"(?:[^"\\]|\\.)*?"', " ", scrub)
            scrub = re.sub(r"says,\s*'(?:[^'])*?'", " ", scrub)
            if (extra_text or "").strip():
                # the premise names people the writer may have anonymised;
                # its dialogue (if any) gets the same scrub
                _x = re.sub(r'"(?:[^"\\]|\\.)*?"', " ", str(extra_text))
                scrub = scrub + " " + _x
            low = scrub.lower()
            found = []
            for d in dirs:
                m = re.search(r"\b" + re.escape(d.lower()) + r"\b", low)
                if m:
                    found.append((m.start(), d))
            found.sort()
            # Until 2.6.2 this was found[:3]: a five-character script silently
            # cast only the first three mentioned, while refs_attached had the
            # writer point ALL of them at photographs - the two uncast ones
            # rendered as random people (reported on Civitai 2026-08-21).
            matched = [d for _, d in found[:self.MAX_SLOTS]]

        # Split the model's 9 reference slots across everyone who matched,
        # instead of letting the first characters exhaust them: 2 characters
        # keep 3 photos each, 4 get 2, five or more get 1. One photo per
        # character holds identity less firmly than a 3-photo set - fewer
        # characters per run is still the stronger setup - but every named
        # character casting SOMETHING beats two of them casting nobody.
        per_char = max_per_character
        if matched:
            per_char = min(max_per_character,
                           max(1, self.MAX_SLOTS // len(matched)))
            if per_char < max_per_character:
                print(f"[H3AutoRefs] {len(matched)} characters matched - "
                      f"{per_char} photo(s) each so everyone fits the model's "
                      f"{self.MAX_SLOTS} reference slots.", flush=True)

        images, binds, lines, pic = [], [], [], 1
        for name in matched:
            folder = remap.get(name.lower(), name)
            fdir = os.path.join(root, folder)
            try:
                files = sorted(f for f in os.listdir(fdir) if
                               f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
            except OSError:
                files = []
            if not files:
                print(f"[H3AutoRefs] {folder}/ matched but has no images; "
                      "skipping.", flush=True)
                continue
            nums = []
            for f in files[:per_char]:
                if len(images) >= self.MAX_SLOTS:
                    break
                img = Image.open(os.path.join(fdir, f))
                img = ImageOps.exif_transpose(img).convert("RGB")
                arr = np.asarray(img).astype(np.float32) / 255.0
                images.append(torch.from_numpy(arr)[None, ...])
                nums.append(f"<Picture {pic}>")
                lines.append(f"{folder}/{f} -> <Picture {pic}>")
                pic += 1
            if nums:
                disp = name.replace("_", " ").title()
                binds.append(f"{', '.join(nums)} "
                             f"{'is' if len(nums) == 1 else 'are'} "
                             f"the same person ({disp}).")

        if not images:
            msg = (f"[H3AutoRefs] no character matched. Folders under "
                   f"{root}: {dirs or '(none)'}")
            if on_no_match == "error":
                raise ValueError(msg + " - name the character in the prose, "
                                 "set `characters`, or switch on_no_match.")
            print(msg + " - continuing WITHOUT references (the writer is "
                  "told found=false and describes the person normally).",
                  flush=True)
            return tuple([None] * self.MAX_SLOTS
                         + [prompt_text or "", "(no references)", None, False])

        prompt_out = "\n".join(binds) + "\n" + (prompt_text or "")
        report = f"{len(images)} ref(s): " + "; ".join(lines)
        print(f"[H3AutoRefs] {report}", flush=True)
        out = images + [None] * (self.MAX_SLOTS - len(images))
        # one batch for the sampler: photos of different sizes are fitted to
        # the first photo's size, the same way core ImageBatch does it
        batch = images[0]
        if len(images) > 1:
            import comfy.utils as _cu
            _h, _w = images[0].shape[1], images[0].shape[2]
            fitted = [images[0]]
            for im in images[1:]:
                if im.shape[1] != _h or im.shape[2] != _w:
                    im = _cu.common_upscale(im.movedim(-1, 1), _w, _h,
                                            "bilinear", "center").movedim(1, -1)
                fitted.append(im)
            batch = torch.cat(fitted, dim=0)
        return tuple(out + [prompt_out, report, batch, True])


class H3RefBatch:
    """Adapt JoyEcho_RefPicker's output to MiniMaxH3ReferenceToVideo.

    The RefPicker returns one IMAGE batch (one frame per picked reference)
    plus a `picked_path` string ("path; path; ..."). H3's reference node
    wants SEPARATE ref_image_N inputs and <Picture N> identity bindings in
    the prompt. This node splits the batch into up to 9 slots, dedupes the
    re-entry duplicates the picker schedules for LTX (meaningless for
    ref2va - all refs bind at t=0), derives character names from each
    path's parent folder, and prepends the binding lines to the prompt.
    """

    MAX_SLOTS = 9

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"forceInput": True}),
        }, "optional": {
            "reference_image": ("IMAGE",),
            "picked_path": ("STRING", {"forceInput": True, "default": ""}),
        }}

    RETURN_TYPES = tuple(["IMAGE"] * 9 + ["STRING", "STRING"])
    RETURN_NAMES = tuple([f"ref_{i+1}" for i in range(9)]
                         + ["prompt_out", "report"])
    FUNCTION = "adapt"
    CATEGORY = "video/minimax"

    def adapt(self, prompt, reference_image=None, picked_path=""):
        import os
        if reference_image is None or (picked_path or "").startswith("("):
            print("[H3RefBatch] no references from picker; prompt passes "
                  "through unchanged.", flush=True)
            return tuple([None] * self.MAX_SLOTS + [prompt, "(no references)"])

        paths = [p.strip() for p in (picked_path or "").split(";") if p.strip()]
        frames = [reference_image[i:i+1] for i in range(reference_image.shape[0])]
        # dedupe the picker's re-entry duplicates (same path repeated)
        keep, seen = [], set()
        for i, fr in enumerate(frames):
            p = paths[i] if i < len(paths) else f"(slot {i})"
            if p in seen:
                continue
            seen.add(p)
            keep.append((fr, p))
        keep = keep[:self.MAX_SLOTS]

        binds, lines, per_char, order = [], [], {}, []
        for idx, (fr, p) in enumerate(keep, start=1):
            char = os.path.basename(os.path.dirname(p)) or "character"
            if char not in per_char:
                per_char[char] = []
                order.append(char)
            per_char[char].append(idx)
            lines.append(f"{char}/{os.path.basename(p)} -> <Picture {idx}>")
        for char in order:
            nums = [f"<Picture {i}>" for i in per_char[char]]
            # variant folders (rae_night, rae-wet) are the SAME
            # person - bind the base name, i.e. the first separator token
            disp = re.split(r"[-_]", char)[0].title()
            binds.append(f"{', '.join(nums)} "
                         f"{'is' if len(nums) == 1 else 'are'} "
                         f"the same person ({disp}).")

        prompt_out = "\n".join(binds) + "\n" + (prompt or "")
        report = f"{len(keep)} ref(s): " + "; ".join(lines)
        print(f"[H3RefBatch] {report}", flush=True)
        out = [fr for fr, _ in keep] + [None] * (self.MAX_SLOTS - len(keep))
        return tuple(out + [prompt_out, report])


class H3StudioControls:
    """ONE set of widgets that drives BOTH stages of the studio graph.

    The two-stage chain has duplicated settings (stage A's conditioning node
    and stage B's multishot sampler each carry width/height/frames/steps/
    sampler/scheduler). Editing one and forgetting the other produces
    mismatched renders that fail at the concat - or worse, succeed at two
    different qualities. This node is the single source: wire its outputs to
    both stages and there is exactly one place to change anything.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 768, "min": 32, "max": 4096,
                              "step": 32}),
            "height": ("INT", {"default": 1344, "min": 32, "max": 4096,
                               "step": 32}),
            "frames_per_shot": ("INT", {
                "default": 362, "min": 5, "max": 481, "step": 17,
                "tooltip": "H3's 17k+5 grid at 24fps. 362 = ~15.1s, the "
                           "trained max. Drives stage A's length AND stage "
                           "B's per-shot length."}),
            "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
            "sampler_name": (_sampler_names_sc(), {"default": "euler"}),
            "scheduler": (_scheduler_names_sc(), {"default": "beta"}),
        }, "optional": {
            # NEW WIDGETS APPEND LAST, and new OUTPUTS append last too -
            # inserting either above shifts every saved workflow.
            "shot_count": ("INT", {
                "default": 0, "min": 0, "max": 30,
                "tooltip": "How many shots the scene is, in TOTAL - not shots per "
                           "prompt. 0 = one shot per --- block in the script (and lets "
                           "the prompt writer decide). Wire this to BOTH the sampler's "
                           "shot_count and the writer's num_shots so they "
                           "cannot disagree."}),
            "use_file_prompts": ("BOOLEAN", {
                "default": False,
                "label_on": "file / prompt set",
                "label_off": "manual entry",
                "tooltip": "Drives the prompt-source switch. OFF = type the "
                           "scene yourself; ON = pull it from the prompt-set "
                           "file or folder. Wire to an H3 Any Switch."}),
            # EXTEND TAKE (2026-08-17): give it a length instead of shots x
            # frames. 0 = off (everything above behaves exactly as before).
            "take_seconds": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.5,
                "tooltip": "EXTEND TAKE: 0 = off. Any other value = the length "
                           "of the finished take; frames_per_shot above then "
                           "sets the WINDOW length (snapped to the grid), "
                           "shot_count is replaced by the computed count "
                           "that fills the time. Set the writer's join_style "
                           "to 'extend take' so it writes ONE speech across "
                           "the windows."}),
            "window": (["auto", "fit this card (VRAM auto)",
                        "243", "226", "209", "192", "175", "158",
                        "141", "124", "107", "90"], {
                "default": "auto",
                "tooltip": "Frames per window when take_seconds is set. auto "
                           "= the largest window whose estimated activation "
                           "pool fits with most of the weights resident on "
                           "THIS card (wire model for a real weight size; 15 "
                           "GB assumed otherwise). Bigger window = fewer "
                           "joins; smaller = less VRAM."}),
            "model": ("MODEL", {
                "tooltip": "Optional, EXTEND TAKE only: the loaded H3 model, "
                           "so 'auto' can size its weights."}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING", "STRING",
                    "INT", "BOOLEAN")
    RETURN_NAMES = ("width", "height", "frames_per_shot", "steps",
                    "sampler_name", "scheduler",
                    "shot_count", "use_file_prompts")
    FUNCTION = "emit"
    CATEGORY = "video/minimax"

    def emit(self, width, height, frames_per_shot, steps,
             sampler_name, scheduler, shot_count=0,
             use_file_prompts=False, take_seconds=0.0, window="auto",
             model=None):
        if take_seconds and take_seconds > 0:
            from .h3_extend import plan_take, _WINDOWS
            # ONE number rules both modes (2.6.3): 'auto' follows the
            # frames_per_shot widget above, snapped to the legal grid, so the
            # panel no longer has two silent authorities over window length.
            # The old VRAM-based sizing lives on as its own combo option.
            if window == "auto":
                snapped = min(_WINDOWS, key=lambda w: abs(w - frames_per_shot))
                if snapped != frames_per_shot:
                    print("[H3StudioControls] extend take: frames_per_shot %d "
                          "is off the 17k+5 grid - snapped to %d"
                          % (frames_per_shot, snapped), flush=True)
                print("[H3StudioControls] extend take: window follows the "
                      "frames_per_shot widget -> %df (pick 'fit this card "
                      "(VRAM auto)' on 'window' for the old auto sizing)"
                      % snapped, flush=True)
                window = str(snapped)
            elif window.startswith("fit this card"):
                window = "auto"
            n, f, total, summary = plan_take(take_seconds, window, width,
                                             height, 24, 22, model)
            frames_per_shot, shot_count = f, n
            print("[H3StudioControls] " + summary, flush=True)
        print(f"[H3StudioControls] {width}x{height}, {frames_per_shot}f/shot, "
              f"{steps} steps, {sampler_name}/{scheduler}, "
              f"shots={'auto' if not shot_count else shot_count}, "
              f"prompts={'file' if use_file_prompts else 'manual'}",
              flush=True)
        return (width, height, frames_per_shot, steps,
                sampler_name, scheduler, shot_count, use_file_prompts)


def _sampler_names_sc():
    try:
        import comfy.samplers
        return comfy.samplers.KSampler.SAMPLERS
    except Exception:
        return ["euler", "res_multistep", "res_2s"]


def _scheduler_names_sc():
    try:
        import comfy.samplers
        return comfy.samplers.KSampler.SCHEDULERS
    except Exception:
        return ["beta", "normal", "simple", "beta57"]




class H3SamplerByName:
    """STRING -> SAMPLER. Lets one master widget drive samplers everywhere:
    combo-to-combo LINKS are rejected by the frontend's type check, but a
    STRING output into this adapter gives core SamplerCustomAdvanced a real
    SAMPLER object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler_name": ("STRING", {
            "forceInput": True,
            "tooltip": "e.g. euler / res_multistep / res_2s - any name "
                       "KSamplerSelect would offer."})}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get"
    CATEGORY = "video/minimax"

    def get(self, sampler_name):
        import comfy.samplers
        name = (sampler_name or "").strip()
        try:
            return (comfy.samplers.sampler_object(name),)
        except Exception:
            raise ValueError(
                "[H3SamplerByName] unknown sampler %r. Valid: %s"
                % (name, ", ".join(comfy.samplers.KSampler.SAMPLERS[:20])))


class H3SigmasByName:
    """model + STRING scheduler + steps -> SIGMAS (BasicScheduler with the
    scheduler chosen by a linked string instead of a per-node combo)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "scheduler": ("STRING", {"forceInput": True}),
            "steps": ("INT", {"default": 12, "min": 1, "max": 100,
                              "forceInput": True}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get"
    CATEGORY = "video/minimax"

    def get(self, model, scheduler, steps, denoise=1.0):
        from comfy_extras import nodes_custom_sampler as ncs
        name = (scheduler or "").strip()
        return (ncs.BasicScheduler().get_sigmas(model, name, steps, denoise)[0],)


NODE_CLASS_MAPPINGS = {
    "H3EpisodeSplit": H3EpisodeSplit,
    "H3LastFrame": H3LastFrame,
    "H3ConcatAV": H3ConcatAV,
    "H3AutoRefs": H3AutoRefs,
    "H3RefBatch": H3RefBatch,
    "H3StudioControls": H3StudioControls,
    "H3SamplerByName": H3SamplerByName,
    "H3SigmasByName": H3SigmasByName,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3EpisodeSplit": "H3 Episode Split (stage A + B)",
    "H3LastFrame": "H3 Last Frame",
    "H3ConcatAV": "H3 Concat A/V",
    "H3AutoRefs": "H3 Auto Refs (folders, by prompt)",
    "H3RefBatch": "H3 Ref Batch (RefPicker -> ref slots)",
    "H3StudioControls": "H3 Studio Controls (one source, both stages)",
    "H3SamplerByName": "H3 Sampler by Name (STRING -> SAMPLER)",
    "H3SigmasByName": "H3 Sigmas by Name (STRING -> SIGMAS)",
}


class H3IntScale:
    """INT scaler for derived resolutions: value / divide_by, snapped to a multiple.

    Built for the TESTLAB two-pass upscale: pass-1 renders at master/1.5 and the
    latent upscaler's x1.5 lands back exactly on the master resolution, so the
    ConcatAV frame-size guard stays happy. Keep master width/height divisible by
    (multiple * divide_by) - 48 for the defaults - or the round trip drifts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "value": ("INT", {"forceInput": True}),
            "divide_by": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 16.0, "step": 0.05}),
            "multiple": ("INT", {"default": 32, "min": 1, "max": 256}),
        }}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "scale"
    CATEGORY = "H3/episode"

    def scale(self, value, divide_by=1.5, multiple=32):
        out = max(multiple, int(round(value / divide_by / multiple)) * multiple)
        return (out,)

NODE_CLASS_MAPPINGS["H3IntScale"] = H3IntScale
NODE_DISPLAY_NAME_MAPPINGS["H3IntScale"] = "H3 Int Scale (divide + snap)"


class _H3AnyT(str):
    """Wildcard socket type: compares unequal to nothing, so any link is accepted."""
    def __ne__(self, other):
        return False

_h3_any = _H3AnyT("*")


class H3AnySwitch:
    """Lazy A/B switch: only the branch the boolean selects is ever executed.

    The unselected branch's upstream nodes are skipped entirely (ComfyUI lazy
    inputs), so a switched-off feature costs zero compute and zero VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"use_on": ("BOOLEAN", {"forceInput": True})},
            "optional": {
                "off_path": (_h3_any, {"lazy": True}),
                "on_path": (_h3_any, {"lazy": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (_h3_any,)
    RETURN_NAMES = ("value",)
    FUNCTION = "pick"
    CATEGORY = "H3/episode"

    def check_lazy_status(self, use_on, off_path=None, on_path=None,
                          prompt=None, unique_id=None):
        want = "on_path" if use_on else "off_path"
        # Only request an input that is actually LINKED in the submitted
        # prompt. Requesting an unlinked lazy input raises NodeInputError and
        # kills the whole queue item - and a stale browser tab that survived a
        # server restart can silently drop a link the saved canvas still has
        # (2026-08-21: 20 queue items died in 0.11 s each on a link the disk
        # copy had). Falling through lets pick() take the other path with a
        # loud warning instead.
        try:
            links = (prompt or {}).get(str(unique_id), {}).get("inputs", {})
            if not isinstance(links.get(want), list):        # a link is [node_id, slot]
                other = "off_path" if use_on else "on_path"
                if isinstance(links.get(other), list):
                    print("[H3AnySwitch] %s is selected but NOT CONNECTED in "
                          "the submitted graph (stale browser tab? reload the "
                          "workflow from the sidebar) - falling back to %s."
                          % (want, other), flush=True)
                    return [other]
                return []                                    # neither wired; pick() reports
        except Exception:
            pass
        return [want]

    def pick(self, use_on, off_path=None, on_path=None, prompt=None,
             unique_id=None):
        name = "on_path" if use_on else "off_path"
        v = on_path if use_on else off_path
        if v is None:
            # an unwired selection is a canvas mistake - say so. A WIRED
            # input that delivered nothing (an OFF gate upstream, e.g. the
            # manual-refs gate) is a valid "nothing" and passes through.
            wired = False
            try:
                wired = isinstance(
                    prompt[str(unique_id)]["inputs"].get(name), list)
            except Exception:
                wired = False
            if not wired:
                raise ValueError(
                    "H3AnySwitch: the selected input (%s) is not connected"
                    % name)
            print("[H3AnySwitch] %s is wired but delivered nothing (a gate "
                  "upstream is off) - passing nothing." % name, flush=True)
        return (v,)


class H3StudioSwitches:
    """One labeled panel of feature toggles; each BOOLEAN output drives H3AnySwitch gates."""

    # 2.6.0: only flags that drive something in a shipped workflow. Removed:
    # two_pass_upscale (the feature itself was removed from the sampler in
    # 2.1.3 - the flag drove nothing), spectrum (owned by the Speed Boosters
    # node), block_cache (same - one switch there instead of un-bypass+flip
    # here), dual_clock_sampler and hybrid_cond (never wired anywhere).
    # Widget layout change is handled by web/js/h3_widget_persistence.js,
    # which maps the old 7/8-value array by position on load; new saves
    # persist by name.
    #
    # OUTPUT SLOTS ARE NOT SHRUNK. Saved graphs link by slot INDEX, and a
    # 2.5.x canvas wires sol_attn from slot 1, chunk_ffn from slot 2 and the
    # remote encoder from slot 7. Shrinking to three outputs made every one
    # of those graphs fail validation ("tuple index out of range") or, worse,
    # silently drive the wrong gate. So the panel keeps the 2.5.5 slot
    # layout: eight BOOLEAN outputs in the original order, three of them
    # driven by widgets, the removed five always False (their gates then take
    # the off_path, i.e. the plain model - exactly what a user who never
    # installed those packs was getting anyway).
    FLAGS = [
        ("sol_attn", False),
        ("chunk_ffn", False),
        ("remote_encoder", False),
    ]
    LEGACY_SLOTS = ["two_pass_upscale", "sol_attn", "chunk_ffn", "spectrum",
                    "block_cache", "dual_clock_sampler", "hybrid_cond",
                    "remote_encoder"]
    LEGACY_LABEL = {"two_pass_upscale": "two_pass (removed)",
                    "spectrum": "spectrum (see Speed Boosters)",
                    "block_cache": "block_cache (see Speed Boosters)",
                    "dual_clock_sampler": "dual_clock (removed)",
                    "hybrid_cond": "hybrid_cond (removed)"}

    TIPS = {
        "sol_attn":
            "Memory-efficient attention: lowers peak VRAM on large canvases, "
            "slightly slower. Two steps to use: install ComfyUI-sol-attn and "
            "un-bypass the attention patch node (Ctrl+B), then turn this on. "
            "The 24 GB card's tool for 243-frame windows.",
        "chunk_ffn":
            "Runs the model's feed-forward layers in chunks to lower peak "
            "VRAM, slightly slower. Same two steps: install ComfyUI-sol-attn, "
            "un-bypass the chunk FFN node (Ctrl+B), then turn this on.",
        "remote_encoder":
            "Encode prompts on a second ComfyUI PC so this card never loads "
            "the 15+ GB text encoder. Fill in the H3 Remote Text Encoder "
            "node first (address + encoder file) - the how-to note beside it "
            "walks through setup. OFF = normal local encoding.",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            n: ("BOOLEAN", {"default": d, "tooltip": cls.TIPS.get(n, "")})
            for n, d in cls.FLAGS}}

    # map(), not a comprehension: a comprehension in a class body cannot see
    # class-level names (LEGACY_LABEL) - that mistake took the whole module
    # down on load.
    RETURN_TYPES = ("BOOLEAN",) * len(LEGACY_SLOTS)
    RETURN_NAMES = tuple(map(LEGACY_LABEL.get, LEGACY_SLOTS, LEGACY_SLOTS))
    FUNCTION = "emit"
    CATEGORY = "H3/episode"

    def emit(self, **kw):
        live = {n: bool(kw.get(n, d)) for n, d in self.FLAGS}
        return tuple(live.get(n, False) for n in self.LEGACY_SLOTS)


class H3LTXTakeControls:
    """MASTER CONTROLS for the LTX-2.5 single-generation canvas (Joy-LTX 2.5).

    LTX renders the whole take in ONE generation. This panel does for LTX what
    plan_take does for H3: it sizes the render to the card. LTX's cost model is
    tokens = (W/32)(H/32) x latent frames, and pass 2 (the upscaled refine)
    runs on the OUTPUT grid - so the upscale factor decides how much a take
    costs: x2 = 4x the pixels of pass 1, x1.5 = 2.25x, none = pass 1 only.
    Measured 2026-08-17: 960x544 -> x2 -> 1920x1088 at 193 frames (51k pass-2
    tokens) renders on 24 GB; 481 frames (124k) hangs a 32 GB card fully
    offloaded. So `auto` keeps the render size and picks the largest output
    that fits: x2, then x1.5 (the LTX-2.3 x1.5 spatial upscaler, verified to
    work on 2.5 latents), then a single pass. It prints the plan.
    """

    TOKENS_PER_GB = 2200      # pass-2 tokens per GB of card (measured, see docstring)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 960, "min": 256, "max": 1920, "step": 32,
                "tooltip": "RENDER (pass-1) width. Output = this x the upscale factor "
                "(960x544 x2 = 1920x1088, x1.5 = 1440x832). Multiples of 32."}),
            "height": ("INT", {"default": 544, "min": 256, "max": 1920, "step": 32,
                "tooltip": "Render (pass-1) height; output = this x the upscale factor."}),
            "take_seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "step": 0.5,
                "tooltip": "Length of the take, rendered in ONE generation. Longer takes "
                "cost tokens; past what the card fits, auto steps the upscale down "
                "(x2 -> x1.5 -> none) and prints which. 8 s x2 / ~14 s x1.5 at "
                "960x544 is the 24 GB comfort zone."}),
            "beat_seconds": ("FLOAT", {"default": 8.0, "min": 3.0, "max": 20.0, "step": 0.5,
                "tooltip": "How long each written sentence/beat should be; the writer "
                "gets take_seconds / beat_seconds beats to write."}),
            "upscale": (["auto", "x2", "x1.5", "none"], {"default": "auto",
                "tooltip": "auto = the largest output that fits your card. x2 = the "
                "official LTX-2.5 spatial upscaler (4x pixels in pass 2). x1.5 = "
                "the LTX-2.3 spatial x1.5 upscaler file - works on 2.5 latents "
                "(verified render), 2.25x pixels, so about 1.8x the seconds of "
                "x2 on the same card. none = pass 1 only at the render size."}),
        }, "optional": {
            "vram_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 1.0,
                "tooltip": "0 = read the card. Set a number to plan for another card."}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "BOOLEAN", "BOOLEAN", "INT", "INT", "STRING")
    RETURN_NAMES = ("pass1_width", "pass1_height", "frames", "two_pass", "use_x15", "beats", "beat_frames", "summary")
    FUNCTION = "emit"
    CATEGORY = "H3/episode"

    @staticmethod
    def _snap32(v):
        return max(256, int(round(v / 32.0)) * 32)

    @staticmethod
    def _up15(v):
        # the rational x1.5 head shuffles x3 then blur-downs /2 -> ceil on odd latent counts
        return int(math.ceil((v // 32) * 1.5)) * 32

    @staticmethod
    def _tokens(w, h, frames):
        return (w // 32) * (h // 32) * ((frames - 1) // 8 + 1)

    def emit(self, width, height, take_seconds, beat_seconds=8.0, upscale="auto", vram_gb=0.0):
        import math
        frames = int(round(float(take_seconds) * 24))
        frames = max(9, int(round((frames - 1) / 8.0)) * 8 + 1)
        beats = max(1, int(math.ceil(float(take_seconds) / float(beat_seconds))))
        beat_frames = max(9, int(round((float(beat_seconds) * 24 - 1) / 8.0)) * 8 + 1)
        total = float(vram_gb) if vram_gb and vram_gb > 0 else 0.0
        if total <= 0:
            try:
                import comfy.model_management as mm
                total = mm.get_total_memory(mm.get_torch_device()) / (1024 ** 3)
            except Exception:
                total = 24.0
        budget = self.TOKENS_PER_GB * total
        w, h = self._snap32(width), self._snap32(height)
        t1 = self._tokens(w, h, frames)
        t2 = self._tokens(2 * w, 2 * h, frames)
        w15, h15 = self._up15(w), self._up15(h)
        t15 = self._tokens(w15, h15, frames)
        two, use15 = False, False
        if upscale == "x2" or (upscale == "auto" and t2 <= budget):
            two = True
            why = "two passes x2 -> %dx%d (%.0fk pass-2 tokens" % (2 * w, 2 * h, t2 / 1e3)
            why += ", over the ~%.0fk this %.0f GB card fits - FORCED, expect streaming or an OOM)" % (budget / 1e3, total) if t2 > budget else ", of ~%.0fk this %.0f GB card fits)" % (budget / 1e3, total)
        elif upscale == "x1.5" or (upscale == "auto" and t15 <= budget):
            two, use15 = True, True
            why = "two passes x1.5 -> %dx%d (%.0fk pass-2 tokens" % (w15, h15, t15 / 1e3)
            why += ", over the ~%.0fk this %.0f GB card fits - FORCED, expect streaming or an OOM)" % (budget / 1e3, total) if t15 > budget else ", of ~%.0fk this %.0f GB card fits; x2 would need %.0fk)" % (budget / 1e3, total, t2 / 1e3)
        else:
            why = "ONE pass -> %dx%d (%.0fk tokens; x1.5 would need %.0fk, x2 %.0fk; ~%.0fk fits this %.0f GB card)" % (w, h, t1 / 1e3, t15 / 1e3, t2 / 1e3, budget / 1e3, total)
            if t1 > budget:
                why += " - even one pass is over budget: shorten the take or lower the size"
        summary = ("LTX TAKE: %.1f s -> %d frames in ONE generation | render %dx%d | %s | writer gets %d "
                   "beat(s) of %d frames" % (float(take_seconds), frames, w, h, why, beats, beat_frames))
        print("[H3LTXTakeControls] " + summary, flush=True)
        return (w, h, frames, two, use15, beats, beat_frames, summary)


NODE_CLASS_MAPPINGS["H3LTXTakeControls"] = H3LTXTakeControls
NODE_DISPLAY_NAME_MAPPINGS["H3LTXTakeControls"] = "LTX Take Controls (Joy-LTX 2.5)"
NODE_CLASS_MAPPINGS["H3AnySwitch"] = H3AnySwitch
NODE_CLASS_MAPPINGS["H3StudioSwitches"] = H3StudioSwitches
NODE_DISPLAY_NAME_MAPPINGS["H3AnySwitch"] = "H3 Any Switch (lazy A/B)"
NODE_DISPLAY_NAME_MAPPINGS["H3StudioSwitches"] = "H3 Studio Switches (feature toggles)"


class H3ReferenceVideo:
    """Trim a clip down to something sane to hand the sampler as a video reference.

    Read this before using it. H3 tells the model a video reference is "a clip
    from an earlier moment of this same continuous scene" and asks it to keep
    the framing, camera distance, room contents and colour temperature. That is
    SCENE and APPEARANCE conditioning. It is NOT motion transfer - H3 has no
    pose, depth or optical-flow path, so the subject will not copy the movement
    in your clip.

    What this node is for: reference frames are subsampled to 2 fps and then
    ride through every sampling step, so their cost is permanent. A 25-second
    clip is roughly 50 reference frames on every step. Trim to a few
    representative seconds and the reference does the same job for a fraction
    of the tokens.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE", {"tooltip": "Frames of the clip to reference."}),
            "start_seconds": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.5,
                "tooltip": "Where the reference window starts."}),
            "seconds": ("FLOAT", {
                "default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5,
                "tooltip": "How much of the clip to keep. 2-4s of a "
                           "representative moment carries the room and the "
                           "look; more mostly buys token cost."}),
            "source_fps": ("FLOAT", {
                "default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                "tooltip": "Frame rate of the incoming clip, used to convert "
                           "seconds to frames. Only affects the trim."}),
        }, "optional": {
            "audio": ("AUDIO", {
                "tooltip": "The clip's soundtrack. Trimmed to the same window. "
                           "Leave empty and the sampler pairs the video with "
                           "silence, which is fine when you only want the look."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("ref_frames", "ref_audio", "info")
    OUTPUT_TOOLTIPS = (
        "Wire to the sampler's reference_video.",
        "Wire to reference_video_audio (or leave it).",
        "What was kept and what it will cost.")
    FUNCTION = "trim"
    CATEGORY = "video/minimax"

    def trim(self, frames, start_seconds, seconds, source_fps, audio=None):
        n = int(frames.shape[0])
        fps = max(1e-6, float(source_fps))
        a = max(0, min(n - 1, int(round(start_seconds * fps))))
        b = max(a + 1, min(n, a + int(round(seconds * fps))))
        out = frames[a:b]
        kept = int(out.shape[0])

        out_audio = None
        if audio is not None and isinstance(audio, dict) and "waveform" in audio:
            try:
                sr = int(audio["sample_rate"])
                w = audio["waveform"]
                s0 = int(round(a / fps * sr))
                s1 = int(round(b / fps * sr))
                s1 = max(s0 + 1, min(int(w.shape[-1]), s1))
                w = w[..., s0:s1]
                if w.shape[-2] == 1:            # ref audio wants stereo
                    w = w.repeat_interleave(2, dim=-2)
                out_audio = {"waveform": w, "sample_rate": sr}
            except Exception as e:
                print("[H3ReferenceVideo] could not trim audio (%s); passing "
                      "it through untrimmed" % e, flush=True)
                out_audio = audio
        if out_audio is None:
            import torch
            sr = 32000
            out_audio = {"waveform": torch.zeros(1, 2, max(1, int(kept / fps * sr))),
                         "sample_rate": sr}

        # the sampler keeps every (FPS//2)-th frame; FPS is 24 in H3
        est = max(1, kept // 12)
        info = ("kept %d frame(s) (%.1fs-%.1fs of %.1fs) -> about %d reference "
                "frame(s) after the sampler's 2 fps subsample. Scene/appearance "
                "conditioning only; this does not transfer motion."
                % (kept, a / fps, b / fps, n / fps, est))
        print("[H3ReferenceVideo] " + info, flush=True)
        if est > 12:
            print("[H3ReferenceVideo] that is a lot of reference frames to "
                  "carry on every step - consider a shorter window.", flush=True)
        return (out, out_audio, info)


NODE_CLASS_MAPPINGS["H3ReferenceVideo"] = H3ReferenceVideo
NODE_DISPLAY_NAME_MAPPINGS["H3ReferenceVideo"] = "H3 Reference Video (trim for ref2va)"
