"""One script, two engines.

H3 and LTX-2.5 want opposite things from a multi-shot script.

H3 chains N separate generations. Each block is one generation, and the scene
description has to be restated WORD FOR WORD in every block or the room drifts
- continuity comes from the text agreeing with itself plus the latent pin.

LTX-2.5 generates all the shots in ONE pass and finds the cuts itself, holding
character, environment, lighting and voice across them natively. Feeding it ten
verbatim repetitions of the same paragraph does not help it and buries the part
that changes.

This node takes the H3 form - the one a writer actually maintains - and emits
both. The scene block is found as the longest common prefix across the blocks,
which is exactly what "restate it verbatim" produces, and is stated once for
LTX with the per-shot remainder following as a beat list.
"""

import re

# Match the SAMPLER's rule verbatim (h3_multishot_utils._parse_script, and
# h3_episode_tools._BLOCK_SPLIT). A literal "\n---\n" split was stricter than
# every other splitter in the pack, and its failure mode is silent: it returns
# ONE block, so an 8-shot script becomes a 1-shot render with no diagnostic.
# Measured on this box 2026-08-17:
#     unix 3 dashes    literal -> 2 blocks   sampler regex -> 2
#     CRLF             literal -> 1 block    sampler regex -> 2   <- collapse
#     trailing space   literal -> 1 block    sampler regex -> 2   <- collapse
# CRLF is the default for anything edited in Notepad on Windows, so this was
# live, not theoretical.
_SEP_RE = re.compile(r"(?m)^---\s*$")


def _blocks(script):
    # The prompt writer emits {"prompts": [...]} - accept that directly so
    # the writer can feed this node without a format hop (Joy-LTX 2.5 canvas).
    s = (script or "").strip()
    if s.startswith("{"):
        try:
            import json as _json
            d = _json.loads(s)
            arr = d.get("prompts") if isinstance(d, dict) else None
            if isinstance(arr, list) and arr:
                return [str(x).strip() for x in arr if str(x).strip()]
        except ValueError:
            pass
    parts = [b.strip() for b in _SEP_RE.split(script or "")]
    return [b for b in parts if b]


def _common_prefix(blocks, sentence_boundary=True):
    """Longest text shared by the start of every block.

    sentence_boundary=True cuts back to the end of a sentence, which is what
    the SCENE block wants - otherwise the LTX prompt opens with half a sentence
    and the per-shot beats start with the other half.

    It is wrong for the per-shot HOLD preamble, which by design ends mid
    sentence on "... says," - cutting back to a full stop finds none and
    silently returns nothing, leaving the preamble to repeat once per shot.
    """
    if len(blocks) < 2:
        return ""
    pre = blocks[0]
    for b in blocks[1:]:
        n = min(len(pre), len(b))
        i = 0
        while i < n and pre[i] == b[i]:
            i += 1
        pre = pre[:i]
        if not pre:
            return ""
    if not sentence_boundary:
        return pre.strip()
    cut = max(pre.rfind(". "), pre.rfind('." '), pre.rfind("? "), pre.rfind("! "))
    return pre[:cut + 1].strip() if cut > 40 else ""


class RiftEngineScript:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The H3 form: shots separated by --- on its own "
                               "line, with the scene block restated verbatim in "
                               "each. Write it once, here, for either engine."}),
            },
            "optional": {
                "ltx_shot_label": ("STRING", {
                    "default": "Then, without moving the camera,",
                    "tooltip": "Joins the per-shot beats in the LTX prompt. "
                               "LTX-2.5 decides its own cut points, so this is "
                               "how you tell it whether the shots are one "
                               "continuous take or separate setups. For hard "
                               "cuts try 'Cut to:'."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("h3_script", "ltx_prompt", "shot_count")
    FUNCTION = "split"
    CATEGORY = "H3/script"

    def split(self, script, ltx_shot_label="Then, without moving the camera,"):
        blocks = _blocks(script)
        if not blocks:
            return ("", "", 0)

        scene = _common_prefix(blocks)
        if scene:
            tails = [b[len(scene):].strip() if b.startswith(scene) else b
                     for b in blocks]
            # Second pass: shots 2..N usually share a "hold" preamble too -
            # "She sits exactly as she was, ... and the woman says," - which is
            # not a prefix of ALL blocks because shot 1 opens differently. Left
            # in, it repeats once per shot and buries the beats.
            hold = (_common_prefix(tails[1:], sentence_boundary=False)
                    if len(tails) > 2 else "")
            # The shared preamble runs right up to the opening quote of the
            # line - every block has one there, so the raw prefix swallows it
            # and each beat loses the quote that opens its dialogue.
            hold = hold.rstrip("“‘\"' \t")
            if len(hold) < 25:          # too short to be a real preamble
                hold = ""
            if hold:
                tails = [tails[0]] + [
                    (t[len(hold):].strip() if t.startswith(hold) else t)
                    for t in tails[1:]]
            label = ltx_shot_label.strip()
            joined = tails[0]
            for t in tails[1:]:
                if label:
                    # The beat continues the joining clause, so it must not
                    # arrive capitalised mid-sentence - UNLESS it opens with
                    # dialogue, where the capital belongs to the line and
                    # lowercasing it corrupts what the character says.
                    if (t[:1].isupper() and not t[:2].isupper()
                            and not t.startswith(("“", '"', "‘", "'"))):
                        t = t[0].lower() + t[1:]
                    joined += " " + label + " " + t
                else:
                    joined += " " + t
            ltx = scene + (" " + hold if hold else "") + " " + joined
        else:
            # no verbatim scene block - the writer is not using the H3
            # convention, so pass the blocks through in order rather than
            # inventing structure that is not there
            label = ltx_shot_label.strip()
            ltx = (" " + label + " ").join(blocks) if label else " ".join(blocks)

        print("[H3 Multishot] EngineScript: %d shot(s); scene block %s; "
              "H3 %d chars, LTX %d chars"
              % (len(blocks),
                 "%d chars, stated once for LTX" % len(scene) if scene
                 else "not detected (blocks passed through)",
                 len(script), len(ltx)), flush=True)
        return (script, ltx, len(blocks))


NODE_CLASS_MAPPINGS = {"RiftEngineScript": RiftEngineScript}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RiftEngineScript": "Script for Two Engines (H3 + LTX-2.5)"}
