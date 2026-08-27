"""Rift Prompt Source - ONE dropdown for both prompt pipelines.

Lists LPFF-style brief/JSON-block .txt files (from the inspire-pack prompts
tree) AND passthrough .json scripts (from <ComfyUI>/input/rift_prompts/) in
a single combo, and always emits the same outputs:

  story_idea (STRING, list) -> JoyEcho_LLMEnhance.story_idea
  character  (STRING, list) -> RiftRefPicker.character
  count      (INT)

.txt files are parsed LPFF-style (blocks split on ---, positive:/negative:/
name: fields); each block becomes one queue item, so multi-block files fan out
exactly like LoadPromptsFromFile. .json files load as a single passthrough item
(the raw JSON text). Blocks without a name: line emit "" (RefPicker falls
through to prompt-scan / fallback), never the filename.

Replaces the LPFF -> UnzipPrompt chain and the Script Picker: wire once,
switch sources by picking a different file.
"""

import json
import re
from pathlib import Path

import folder_paths

# canonical folder; the pre-rename name is still read so existing
# script folders keep working without being moved
def _script_picker():
    """Import the sibling script-picker module, whatever the install layout.

    This pack ships two ways: as a package folder (relative import works) and
    as loose files in custom_nodes/ (it does not - ComfyUI loads each file
    under a private module name, so the bare name is not importable either).
    Falling through to a file-location load covers both. The two-branch
    version of this raised ModuleNotFoundError on every loose install
    (user-reported 2026-08-11).
    """
    try:
        from . import rift_script_picker as m
        return m
    except Exception:
        pass
    try:
        import rift_script_picker as m
        return m
    except Exception:
        pass
    import importlib.util as _ilu
    import os as _os
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "rift_script_picker.py")
    spec = _ilu.spec_from_file_location("rift_script_picker", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_JSON_SUBDIR = "rift_prompts"
_JSON_SUBDIR_LEGACY = "joyecho_prompts"
_TXT_PREFIX = "TXT: "
_JSON_PREFIX = "JSON: "
_EMPTY = "(no prompt files found)"

# folder -> pool size, so the ordered walk listing prints once per folder per
# session instead of ahead of every queued job. Re-prints if the folder's file
# count changes, which is the case where a stale listing would mislead.
_WALK_LISTED = {}

_BLOCK_SPLIT = re.compile(r"\n\s*-+\s*\n")
_BLOCK_PATTERN = re.compile(
    r"^(?:(?:name:(?P<name>.*?)|positive:(?P<positive>.*?)|negative:(?P<negative>.*?))\n*)+$",
    re.DOTALL,
)


def _txt_roots() -> list[Path]:
    """EVERY registered inspire_prompts root that exists.

    Taking roots[0] alone was wrong wherever more than one is registered: the Inspire pack
    registers its own prompts folder, so a corpus added through extra_model_paths.yaml sat at
    roots[1] and its thousands of .txt files never appeared in the picker (2026-08-20).
    """
    try:
        roots = folder_paths.get_folder_paths("inspire_prompts") or []
    except Exception:
        return []
    out, seen = [], set()
    for r in roots:
        rp = Path(r)
        k = str(rp).lower()
        if k not in seen and rp.is_dir():
            seen.add(k)
            out.append(rp)
    return out


def _txt_root() -> Path | None:
    """The root a relative TXT entry resolves against - the first one holding any .txt."""
    roots = _txt_roots()
    for r in roots:
        if next(r.rglob("*.txt"), None) is not None:
            return r
    return roots[0] if roots else None


def _json_root() -> Path:
    """The canonical scripts folder, created on demand."""
    d = Path(folder_paths.get_input_directory()) / _JSON_SUBDIR
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _json_roots() -> list[Path]:
    """Canonical folder first, then the pre-rename one if it still exists.

    Renaming the folder outright would hide every script an existing install
    already has filed, so the old location stays readable indefinitely.
    """
    roots = [_json_root()]
    legacy = Path(folder_paths.get_input_directory()) / _JSON_SUBDIR_LEGACY
    if legacy.is_dir():
        roots.append(legacy)
    return roots


def _list_files() -> list[str]:
    out = []
    seen_txt = set()
    for root in _txt_roots():
        for p in sorted(root.rglob("*.txt")):
            rel = str(p.relative_to(root))
            if rel.lower() in seen_txt:
                continue
            seen_txt.add(rel.lower())
            out.append(_TXT_PREFIX + rel)
    # rglob, matching the .txt branch above: a flat glob hides scripts filed in
    # subfolders, which is how anyone organises more than a handful of them.
    # _resolve() already joins the relative path, so nested names round-trip.
    seen = set()
    for jroot in _json_roots():
        for p in sorted(jroot.rglob("*.json")):
            rel = str(p.relative_to(jroot)).replace("\\", "/")
            if rel in seen:      # same name in both folders: canonical wins
                continue
            seen.add(rel)
            out.append(_JSON_PREFIX + rel)
    return out or [_EMPTY]


def _folder_of(entry: str) -> str:
    r"""Folder key of a dropdown entry; root-level files -> "<KIND>: (root)"."""
    for kind, prefix in (("TXT", _TXT_PREFIX), ("JSON", _JSON_PREFIX)):
        if entry.startswith(prefix):
            rest = entry[len(prefix):]
            ix = max(rest.rfind("\\"), rest.rfind("/"))
            return f"{kind}: (root)" if ix < 0 else f"{kind}: {rest[:ix]}"
    return "(all)"


def _folder_list() -> list[str]:
    return ["(all)"] + sorted({_folder_of(e) for e in _list_files()
                               if e != _EMPTY})


def _resolve(choice: str) -> Path | None:
    if choice.startswith(_TXT_PREFIX):
        rel = choice[len(_TXT_PREFIX):]
        for root in _txt_roots():          # the entry may live under any registered root
            if (root / rel).is_file():
                return root / rel
        root = _txt_root()
        return (root / rel) if root else None
    if choice.startswith(_JSON_PREFIX):
        rel = choice[len(_JSON_PREFIX):]
        for root in _json_roots():
            if (root / rel).is_file():
                return root / rel
        return _json_root() / rel
    return None


class RiftPromptSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_file": (_list_files(),),
            },
            "optional": {
                "load_cap": ("INT", {"default": 0, "min": 0,
                                     "tooltip": "TXT files only: 0 = all blocks, N = first N from start_index."}),
                "start_index": ("INT", {"default": 0, "min": 0,
                                        "tooltip": "TXT files only: first block index to load."}),
                "character_override": ("STRING", {"default": "",
                                                  "tooltip": "Force this character for EVERY emitted item "
                                                             "(useful for .json scripts, which carry no name field)."}),
                "manual_path": ("STRING", {"default": "",
                                           "tooltip": "Point at ANY .txt (LPFF blocks) or .json (passthrough script) "
                                                      "file by full path, ignoring the dropdown. Removes any dependency "
                                                      "on the inspire-pack prompts tree or input/rift_prompts/. "
                                                      "Leave empty to use source_file."}),
                "folder": (_folder_list(), {"default": "(all)", "tooltip":
                                            "Filter source_file to one folder (the pack's frontend "
                                            "script filters the dropdown live). Appended last so "
                                            "saved widget order is preserved."}),
                # NEW IN 2.2.4 - appended after `folder` for the same reason
                # that one was: widgets_values is positional and inserting
                # earlier shifts every saved graph.
                "walk_folder": ("BOOLEAN", {
                    "default": False,
                    "label_on": "index picks the FILE",
                    "label_off": "index picks the BLOCK",
                    "tooltip": "OFF (default): start_index selects a block "
                               "INSIDE source_file, as always. ON: start_index "
                               "selects WHICH FILE, walking the folder chosen "
                               "above in sorted order, and the whole file is "
                               "emitted. Wire start_index to a PrimitiveInt set "
                               "to 'increment' and queue N times to render a "
                               "folder of finished scripts unattended. Needed "
                               "because an H3 script uses --- for SHOT "
                               "boundaries, so scenes cannot be concatenated "
                               "into one file the way LPFF batches can."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("story_idea", "character", "count")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "load"
    CATEGORY = "H3/script"

    @classmethod
    def VALIDATE_INPUTS(cls, source_file=None, folder=None, **kwargs):
        # ComfyUI checks every combo value in the graph BEFORE it will queue
        # anything, so a saved source_file that no longer exists - a renamed
        # folder, a shared workflow, or simply the prompt lane switched off -
        # failed the whole prompt with "Value not in list" and nothing ran.
        # The file only matters if this node actually executes; accept any
        # value here and let load() raise a message that names the file.
        return True

    @classmethod
    def IS_CHANGED(cls, source_file, load_cap=0, start_index=0, character_override="",
                   manual_path="", folder="(all)", walk_folder=False):
        # walk_folder makes start_index choose the file, so it has to be part of
        # the cache key together with the folder - otherwise queueing with an
        # incrementing index re-serves the first file from cache.
        p = Path(manual_path.strip()) if manual_path.strip() else _resolve(source_file)
        key = manual_path.strip() or source_file
        if walk_folder:
            key = f"walk:{folder}:{start_index}:{key}"
        try:
            return f"{key}:{p.stat().st_mtime}:{load_cap}:{start_index}:{character_override}"
        except Exception:
            return key

    def load(self, source_file, load_cap=0, start_index=0, character_override="",
             manual_path="", folder="(all)", walk_folder=False):
        manual = manual_path.strip()
        if walk_folder and not manual:
            # start_index selects the FILE, not a block inside one. An H3 script
            # uses --- for shot boundaries, so a folder of finished scripts is
            # the only way to batch scenes; this walks it.
            pool = [e for e in _list_files() if e != _EMPTY]
            if folder and folder != "(all)":
                pool = [e for e in pool if _folder_of(e) == folder]
            if not pool:
                raise ValueError(
                    f"PromptSource: walk_folder is on but folder {folder!r} "
                    f"holds no prompt files. Pick a folder that has some, or "
                    f"turn walk_folder off.")
            _pos = int(start_index) % len(pool)
            source_file = pool[_pos]
            # Print the whole ordered list ONCE per folder per session. Without
            # it the index-to-file mapping is invisible until you have already
            # run the job, and after a cancel you are left doing off-by-one
            # arithmetic against a single log line to work out where to restart.
            _seen = _WALK_LISTED.get(folder)
            if _seen != len(pool):
                _WALK_LISTED[folder] = len(pool)
                print("[PromptSource] walk order for %s (%d file(s)) - "
                      "EPISODE INDEX is 0-based, so the number on the left is "
                      "what to set to start there:" % (folder, len(pool)),
                      flush=True)
                for _i, _e in enumerate(pool):
                    print("    %3d  %s" % (_i, _e), flush=True)
            print("[PromptSource] walk %d/%d (EPISODE INDEX %d) -> %s   "
                  "| to resume here set EPISODE INDEX = %d"
                  % (_pos + 1, len(pool), _pos, source_file, _pos), flush=True)
            if _pos == len(pool) - 1:
                print("[PromptSource] that was the LAST file in %s; the next "
                      "queued job wraps back to index 0." % folder, flush=True)
            start_index, load_cap = 0, 0     # the whole file is the unit now
        elif (folder and folder != "(all)" and not manual
                and _folder_of(source_file) != folder):
            # `folder` narrows the dropdown; it is not a veto on a file the user
            # deliberately selected. Outside walk mode it has no other job, so a
            # mismatch is a stale-filter annoyance rather than an error - and
            # failing the whole prompt over it wasted a queued run.
            print("[PromptSource] note: folder filter is %r but the chosen file "
                  "is in %r. Using the file you picked. (The filter only "
                  "selects the pool when walk_folder is ON.)"
                  % (folder, _folder_of(source_file)), flush=True)
        if manual:
            p = Path(manual).expanduser()
            is_json = p.suffix.lower() == ".json"
        else:
            if source_file == _EMPTY:
                # Name the CAUSE, not the symptom. The .txt root comes from
                # folder_paths.get_folder_paths("inspire_prompts"), a key that
                # comfyui-inspire-pack registers. Without that pack installed
                # the key does not exist, _txt_root() returns None and this
                # dropdown is empty however full the prompts folder is - which
                # reads as "the node is broken" rather than "a dependency is
                # missing". Cost a real debugging session on 2026-08-14.
                raise ValueError(
                    "PromptSource: no prompt files found.\n"
                    "  .txt files are located through the 'inspire_prompts' "
                    "folder path, which is registered by comfyui-inspire-pack. "
                    "%s\n"
                    "  Fix by EITHER installing comfyui-inspire-pack and putting "
                    "your prompts in its prompts/ folder, OR adding this to "
                    "extra_model_paths.yaml and restarting:\n"
                    "      my_prompts:\n"
                    "          base_path: D:/path/to/your/prompts/\n"
                    "          inspire_prompts: .\n"
                    "  OR just set manual_path to a file and ignore the dropdown."
                    % ("That pack does NOT appear to be installed."
                       if _txt_root() is None else
                       "It is registered, but no .txt files were found under it."))
            p = _resolve(source_file)
            is_json = source_file.startswith(_JSON_PREFIX)
        if p is None or not p.exists():
            raise FileNotFoundError(f"PromptSource: {manual or source_file} -> {p} not found. "
                                    "Refresh the node list (R) after adding files, or check manual_path.")
        text = p.read_text(encoding="utf-8")
        override = character_override.strip().lower()

        if is_json:
            data = json.loads(text)  # fail early with a clear error
            arr = data.get("prompts") or data.get("shots")
            if not isinstance(arr, list) or not arr:
                raise ValueError(f"{p.name} must contain a non-empty 'prompts' (or 'shots') array.")
            # Per-character audio memory: derive the speaker order from the
            # script (an explicit "speakers" array, else the "<ID> is talking"
            # attribution in each shot) and stash it for JoyEcho_Generate. The
            # derivation lives in rift_script_picker; both loaders feed the
            # same stash so it works regardless of which node the graph uses -
            # the first wiring of this feature went ONLY into ScriptPicker and
            # the live workflow loads through THIS node, so the 11:23 render
            # silently ran with per-character memory off.
            _sp = _script_picker()
            derive_speakers = _sp.derive_speakers
            set_last_speakers = _sp.set_last_speakers
            set_last_voice_refs = _sp.set_last_voice_refs
            speakers = derive_speakers(data, arr)
            set_last_speakers(speakers)
            vrefs = data.get("voice_refs") or {}
            set_last_voice_refs(vrefs)
            print(f"[H3 Multishot] PromptSource: {p.name} (json, {len(arr)} shots, 1 item)"
                  + (f", speakers: {' '.join(speakers)}" if speakers else ", no speaker tags")
                  + (f", voice anchors: {', '.join(vrefs)}" if vrefs else "")
                  + ".", flush=True)
            return ([text], [override], 1)

        # TXT: LPFF-style blocks. Clear the speaker stash - it holds whatever the
        # LAST json load derived, and stale speakers applied to an unrelated
        # script would filter the audio bank to the wrong characters.
        _sp = _script_picker()
        set_last_speakers = _sp.set_last_speakers
        set_last_voice_refs = _sp.set_last_voice_refs
        set_last_speakers([])
        set_last_voice_refs({})
        items, names = [], []
        for blk in _BLOCK_SPLIT.split(text):
            m = _BLOCK_PATTERN.search(blk)
            if m and m.group("positive") is not None:
                items.append(m.group("positive").strip())
                nm = (m.group("name") or "").strip().lower()
                names.append(override or nm)
                continue
            # PLAIN-PROSE FALLBACK: a --- separated file with no LPFF labels
            # (no positive:/name:/negative:) is a legitimate briefs file -
            # treat the whole block as the positive rather than yielding
            # zero blocks and failing the queue (MINIMAX.txt lesson).
            raw = blk.strip()
            if raw and not re.match(r"^(name|positive|negative)\s*:", raw,
                                    re.I | re.M):
                items.append(raw)
                names.append(override or "")
        total = len(items)
        items = items[start_index:]
        names = names[start_index:]
        if load_cap > 0:
            items = items[:load_cap]
            names = names[:load_cap]
        if not items:
            raise ValueError(f"PromptSource: {p.name} yielded no blocks "
                             f"(total {total}, start_index {start_index}, load_cap {load_cap}).")
        print(f"[H3 Multishot] PromptSource: {p.name} (txt, {len(items)}/{total} blocks).", flush=True)
        return (items, names, len(items))


NODE_CLASS_MAPPINGS = {"RiftPromptSource": RiftPromptSource}
NODE_DISPLAY_NAME_MAPPINGS = {"RiftPromptSource": "Prompt Source (txt briefs + json scripts)"}

# deprecated alias - saved workflows made before the rename still
# reference the old key; keep it resolving so they open.
NODE_CLASS_MAPPINGS["JoyEcho_PromptSource"] = RiftPromptSource
NODE_DISPLAY_NAME_MAPPINGS["JoyEcho_PromptSource"] = "[deprecated] " + NODE_DISPLAY_NAME_MAPPINGS["RiftPromptSource"]
