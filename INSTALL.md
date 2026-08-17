# MiniMax-H3 Seamless Chain — Install

Requires **ComfyUI v0.30.0 or newer** (native MiniMax-H3 support).

Four workflows ship. **H3_Extend_Take** is the main one (one premise, one
continuous take for the length you ask). CORE needs nothing except this pack
if you want to try the engine with zero extras.

---

## 1. Install the node pack

Copy the `ComfyUI-H3-Multishot` folder into `ComfyUI/custom_nodes/`:

```
ComfyUI/custom_nodes/ComfyUI-H3-Multishot/
```

Or via ComfyUI-Manager → *Install via Git URL*:

```
https://github.com/jlucasmcrell/ComfyUI-H3-Multishot
```

Restart ComfyUI.

## 2. Get the models

| What | Where | Notes |
|---|---|---|
| **Checkpoint** | `huggingface.co/joeygambino/MiniMax-H3-GGUF` | Q8_0 for 32 GB, Q5_1 for 24–32 GB, Q4_0 for 16 GB. The workflows ship on **`ref2va`**; **`fl2va`** also chains and is lighter — see below. |
| **Text encoder** | `huggingface.co/Comfy-Org/MiniMax-H3` | Load with the pack's loader, `type = minimax`. |
| **Video + audio VAE** | `huggingface.co/Comfy-Org/MiniMax-H3` | Two separate VAEs; both are wired in the workflows. |

Using GGUF checkpoints? Install **ComfyUI-GGUF**. This pack teaches it the
`minimax_h3` architecture in memory at startup, so installing the pack is
normally the whole fix.

If you still get an architecture error, apply the on-disk fallback once from
inside the pack folder and restart:

```
python apply_gguf_arch_patch.py
```

It is idempotent — running it twice is harmless.

### Which checkpoint

Both H3 variants chain. They differ in what else they can carry:

- **`ref2va` — shipped default.** Adds *reference rows*, which is the
  mechanism behind voice anchoring (`voice_ref` / `self_anchor_voice`) and
  the identity bank. Reference tokens ride through every sampling step, so
  it is somewhat slower and wants a little more headroom.
- **`fl2va`** — the first/last-frame variant. Fewer tokens per step, no
  reference rows, so voice anchoring and the bank do nothing on it. Chains
  perfectly well; the voice is carried by the frame relay and the join's
  audio reference instead of being explicitly pinned.

Both files are **exactly the same size** at every quant level - fl2va and ref2va
are byte-identical in bytes, so there is no VRAM or disk saving in choosing one.
"Lighter" means fewer tokens per sampling step, because there are no reference
rows riding along; it has never meant a smaller file.

What you actually trade:

* **`ref2va`** carries reference rows - voice anchoring, the identity bank, and
  reference images. It holds a supplied keyframe only *softly*.
* **`fl2va`** has no reference rows at all, but it **lands on a supplied frame**.
  Measured against the same frame: **26.35 dB on fl2va versus 16.15 dB
  (turbo, 6 steps) and 16.81 dB (stock, 20 steps) on ref2va + keyframe.** The
  stock control rules out the sampler - the checkpoints genuinely differ. fl2va
  can also take a first *and* last frame and plan a camera move between them,
  which ref2va cannot do at all.

Rule of thumb: **ref2va when identity or voice must persist, fl2va when a shot
must start exactly where the last one ended.**

Blind review passed on the `fl2va` configuration. `ref2va` ships as the
default because it makes voice identity explicit rather than emergent — set
the checkpoint to `fl2va` and turn `self_anchor_voice` off if you prefer the
reviewed path.


### Prompt files and `comfyui-inspire-pack`

`RiftPromptSource` finds `.txt` prompt files through the `inspire_prompts`
folder path, which is registered by **comfyui-inspire-pack**. If that pack is
not installed the dropdown is empty however many prompt files you have, because
the path is never registered. Three ways to satisfy it:

* install **comfyui-inspire-pack** and put prompts in its `prompts/` folder, or
* register the path yourself in `extra_model_paths.yaml` and restart:

```yaml
my_prompts:
    base_path: D:/path/to/your/prompts/
    inspire_prompts: .
```

* or ignore the dropdown entirely and set `manual_path` to a file.

As of 2.2.4 the node says which of these applies instead of just coming back
empty.

## 3. Load a workflow

```
workflows/H3_Extend_Take.json           <- MAIN: one premise, one continuous take (start here)
workflows/H3_Seamless_Chain_v2.json     <- shot-by-shot chain, every option
workflows/H3_Seamless_Chain_CORE.json   <- same engine, zero extra packs
workflows/H3_Keyframes.json             <- single-clip keyframe anchoring
```

If you had an older H3 canvas open, close its tab before opening the new file:
ComfyUI restores old tabs and re-selects them when you click the same name.

**v2** supersedes the older `H3_Multishot_AIO` and `H3_Multishot_MEMORY`
graphs — every lane they had is in v2, behind gates that ship off. If you
have those open from a previous version they still work; there is just no
longer a reason to use them.

**Keyframes** is a different job, not a lesser one: a hand-built sampling
graph for anchoring one clip at chosen frame positions with per-anchor
condition strength. It is not multishot and does not chain.

The last three are carried forward unchanged from v1.5 — this release adds
workflows, it does not replace them.

Type your shots into the sampler's `script` box (one prompt per shot, `---`
on its own line between them) and queue once. Read `PROMPTING.md` before
writing — the boundary rules are what make the joins invisible.

---

## Packs the FULL workflow needs

**CORE needs none of this.** It runs on this pack plus stock ComfyUI, and that
is tested on a clean install.

The full workflow is different. ComfyUI validates **every** node class in a
graph before it will queue, so a missing pack is not a partial loss — the
workflow will not run at all until you install these or delete the affected
nodes. Two of them ship inside this zip.

| Pack | Needed for | Source |
|---|---|---|
| **ComfyUI_JoyAI_Echo_GGUF_Nodes** | the LLM prompt writer | **in this zip** — copy it in |
| **ComfyUI-H3-Motion-Context** (NikoDemon80) | `continuity = context_pin`, the shipped default | `github.com/NikoDemon80/ComfyUI-H3-Motion-Context` — **this exact pack**, see the note below |
| **RES4LYF** | the `beta57` scheduler the full workflow ships with | `github.com/ClownsharkBatwing/RES4LYF` |
| **ComfyUI-sol-attn** | the memory-efficient attention and chunked feed-forward switches | ComfyUI-sol-attn |
| **comfyui-minimax-h3-blockcache-T8** | the block-cache speed switch | comfyui-minimax-h3-blockcache-T8 |
| **ComfyUI-Custom-Scripts** | `ShowText`, the in-canvas script preview. **Ships bypassed** - install it and Ctrl+B the node if you want it | `github.com/pythongosssss/ComfyUI-Custom-Scripts` |

### A note on the Motion-Context fork

There is an active fork — **ethanfel/ComfyUI-MiniMaxH3-Contex-Loop** — that is
well ahead of upstream and adds a disk-backed chain/loop system. It is a
**complement, not a replacement**: by its own design it leaves the
`MiniMaxH3MotionContext` node id to NikoDemon80's pack and exports its own
`MiniMaxH3Chain*` / `MiniMaxH3LoopTrim` nodes instead.

So if you install the fork *instead of* upstream, `continuity = context_pin`
still fails — the node it calls is not there. Install **both**; they are built
to coexist, and this pack works with either one's runtime patches because all
three honour the same patch-ownership markers.

### If you would rather not install them

Every one of these can be removed instead:

- **RES4LYF** — set `scheduler` to `beta` on the sampler and on MASTER
  CONTROLS. Measured on an identical seed: `beta57` scored 10/10 for lip-sync
  against 8/10 for stock `beta` ("slight synthetic stiff quality to the mouth"),
  everything else equal. CORE ships `beta` for exactly this reason.
- **ComfyUI-H3-Motion-Context** — set `continuity` to `first_frame`. That is the
  model's own trained hand-off; it chains well and needs nothing extra.
- **ComfyUI-sol-attn** / **blockcache-T8** — nothing to do. All three patch
  nodes ship **bypassed**, so the graph queues without either pack. To actually
  use one you need BOTH steps, in this order: install the pack, restart, `Ctrl+B`
  the node to un-bypass it, **then** turn its switch on in VRAM / SPEED
  SWITCHES. Un-bypassing alone does nothing, and flipping the switch while the
  node is still bypassed does nothing either. Un-bypassing a node whose pack is
  NOT installed makes the whole workflow un-queueable — that is what the
  bypassing is protecting you from.
- **ComfyUI-Custom-Scripts** — nothing to do. SCRIPT PREVIEW ships bypassed, so
  a bypassed node is dropped from the prompt and the graph queues without the
  pack installed. It is a leaf anyway: the writer feeds the sampler directly, so
  the preview costs you the preview and nothing else. Install the pack and
  Ctrl+B the node to switch it on.

### The prompt writer pack — it is in this zip

Copy `ComfyUI_JoyAI_Echo_GGUF_Nodes/` from this zip into
`ComfyUI/custom_nodes/`, alongside `ComfyUI-H3-Multishot/`. If you already have
that pack installed, replace it with this one.

It is RealRebelAI's pack — <https://github.com/RealRebelAI/ComfyUI_JoyAI_Echo_GGUF_Nodes>
— modified. `NOTICE_RIFT_MODIFICATIONS.md` inside the folder lists every
change. The workflow drives the writer through inputs the upstream release does
not have, chiefly `join_style`, which appends the render-verified boundary rules
to the system prompt. With upstream the workflow still loads and renders, but
those values are dropped in silence and the join rules never reach the model.
(The workflow also carries the rules in its own `system_prompt` box, so it is
correct either way.)

### The prompt writer needs a model you actually have

`JoyEcho_LLMEnhance` calls an OpenAI-compatible endpoint. The workflows ship
pointed at a local Ollama; the model name is whatever the canvas was saved
with (H3_Extend_Take: an Ollama cloud tag, `deepseek-v4-pro:cloud`; v2:
`qwen3:14b`):

```
base_url    http://localhost:11434/v1
model_name  <pick one you have>
```

**Pick a model you have pulled (or a cloud/hosted one you are signed in for)
before the first queue, or the run stops on a 404** —
`LLM API error 404: model '...' not found`:

```
ollama pull qwen3:14b
```

Any OpenAI-compatible endpoint works; put its URL in `base_url` and the exact
model tag in `model_name`. `ollama list` prints the tags you have, and the tag
must match one of them character for character. A remote or hosted endpoint is
often the better choice: a local writer large enough to be good competes with
the H3 model for the same card, and on anything under 32 GB it will evict the
DiT mid-render.

If you do run a local writer, turn on **`unload_model_after`** on the writer
node. It frees that model from Ollama as soon as the script is written, so
the video model gets the card. Without it the model stays resident for the
server's default five minutes — the whole of your first shot. The switch is
added to the writer by this pack at startup; it is off by default.

**No LLM at all?** Set the master panel's `use_file_prompts` to manual entry,
delete the REWRITER, and wire your own shot script straight into the sampler's
`script` input — one prompt per shot, separated by `---` lines. `PROMPTING.md`
documents the boundary rules the writer would otherwise apply for you, and the
CORE workflow already works this way.

**Without the Motion Context pack**, set `continuity` to `first_frame` on
the sampler — that uses the model's own trained hand-off and needs no extra
pack. This is exactly what CORE does.

---

## Everything else the workflows use

These ship with ComfyUI; nothing to install:

`LoadImage` · `LoadAudio` · `SaveVideo` · `SaveAudio` · `CreateVideo` ·
`VAELoader` · `PrimitiveFloat` · `Note`

---

## Sanity check

A correct first run prints something like this to the console:

```
[H3Multishot] shot 1/3 (243f @ 1280x736)...
[H3AutoReserve] shape cells=...: reserving X GB (...)
[H3Multishot] shot 2/3 ...
[H3Multishot] done: 3 shots, 727 frames (~30.3s).
```

The progress bar sitting at `0/N` for a while at the start is the first pass
over the shot — not a hang.

### Automatic reference casting

If you keep character reference photos, `H3AutoRefs` can cast them from your
script - one subfolder per character under a `refs_root` (default
`input/h3_refs/`), matched against names in the prose. Full behaviour and every
widget is in `SETTINGS.md` under "Automatic reference casting". Optional; the
manual `LoadImage` reference slots work exactly as before.
