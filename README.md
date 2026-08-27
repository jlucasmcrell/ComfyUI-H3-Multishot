# ComfyUI-H3-Multishot

**Render a multi-shot MiniMax-H3 scene as one continuous take: no visible cuts, no colour shift between shots, unbroken audio.**

MiniMax-H3 natively generates blocks of roughly 10-15 seconds. This pack chains those blocks into a scene of arbitrary length and joins them so the result reads as a single unedited camera take rather than a cut sequence. It ships two independent chaining mechanisms, a complete single-purpose workflow (plus a variant with zero third-party dependencies), a dual-format model loader (safetensors + GGUF), and the GGUF architecture patch H3 needs.

Current release: **v2.6.5** - canvases redrawn left-to-right in numbered lanes, unique workflow ids, a leftover-VRAM sweep in the reserve planner, and a watchdog on the master assembly.

- Guides: [the 5-minute guide](https://civitai.com/articles/34047/make-talking-videos-with-minimax-h3-the-5-minute-guide-26) and [every setting explained](https://civitai.com/articles/34046/every-setting-explained-the-seamless-chain-deep-manual)
- GitHub: <https://github.com/jlucasmcrell/ComfyUI-H3-Multishot>
- Civitai: <https://civitai.com/models/2833322>

---


### Nodes added in 2.1

| Node | What it does |
|---|---|
| `RiftPromptSource` | One dropdown over LPFF-style `.txt` briefs and passthrough `.json` scripts. Emits `story_idea` / `character` / `count`. Reads `input/rift_prompts/`, and still reads the older `input/joyecho_prompts/` so existing folders keep working. |
| `RiftScriptPicker` | JSON script dropdown, and the speaker/voice stash `RiftPromptSource` feeds. |
| *(not a node)* `unload_model_after` | A switch **added to the LLM prompt writer** (`JoyEcho_LLMEnhance`). On, the writer frees its own model from Ollama once the script is written, so the video model gets the card. Uses the writer's existing `base_url` and `model_name` — nothing to keep in sync. Added in memory at startup by this pack, so the writer's own package is not modified; the switch simply appears on the node. Off by default. Ollama's OpenAI-compatible endpoint has no `keep_alive` field and its shim never sets one, so without this the model sits for the server default of five minutes — your whole first shot. |

`JoyEcho_PromptSource` and `JoyEcho_ScriptPicker` still resolve as deprecated
aliases, so graphs saved against 2.0 open unchanged. They were never published
under those names — that was the 2.0 bug.

### The prompt writer needs a model you actually have

The full workflow points at a local Ollama with `model_name = qwen3:14b`.
**Pull it before the first queue** or the run stops with
`LLM API error 404: model 'qwen3:14b' not found`:

```
ollama pull qwen3:14b
```

Any OpenAI-compatible endpoint works — its URL in `base_url`, its exact tag in
`model_name` (`ollama list` prints the tags you have). A remote endpoint is
often better: a local writer large enough to be good competes with H3 for the
same card and on under 32 GB will evict the model mid-render. When you do run
local, turn on `unload_model_after` on the writer — it frees the model as soon
as the script is written.

No LLM at all? Set `use_file_prompts` to manual entry, delete the writer, and
feed your own `---`-separated shot script into the sampler's `script` input.
The CORE workflow already works this way.

**Try a render with every switch off and the reserve at `0` before touching
any of this.** The activation reserve measures each shape and conditioning
payload as it renders and sizes the pool itself, and it holds on 24 GB cards
as well as 32 GB. A hand-set reserve *overrides* that measurement. These
switches are for digging out of a spill the console has already named.

## Quick start

Five steps to a rendering chain. This path uses the **CORE** workflow, which needs nothing except this pack and ComfyUI built-ins.

**1. Install the node pack**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jlucasmcrell/ComfyUI-H3-Multishot
```

Or search **H3 Multishot** in ComfyUI-Manager - the pack is on the Comfy
Registry as `comfyui-h3-multishot`. *Install via Git URL* also works.
Requires **ComfyUI v0.30.0+** (native MiniMax-H3 support). **v0.34+ is
recommended**: it places interior keyframe anchors natively. On an older core
those in-between anchors need the ComfyUI-H3-Motion-Context pack (already
required for `continuity=context_pin`); first/last anchors work everywhere.

**2. Put the models in place**

```
ComfyUI/models/diffusion_models/   <- MiniMax-H3 ref2va checkpoint (safetensors or GGUF)
ComfyUI/models/text_encoders/      <- H3 text encoder (+ its -mmproj sidecar if GGUF)
ComfyUI/models/vae/                <- video VAE + audio VAE
```

Download links are in [Models](#models).

**3. Restart ComfyUI and load the workflow**

Copy **both** node folders from the zip into `ComfyUI/custom_nodes/`,
restart, then load a workflow from the workflow menu. The full contents
are listed under [Files in the release zip](#files-in-the-release-zip).

**Three workflows, one reason each.** `v2` is everything with the optional
lanes gated off. `CORE` does the same job with zero third-party packs - start
there if you want a render before installing anything else. `Keyframes` is a
different job: a hand-built sampling graph for anchoring a single clip at
chosen frame positions with per-anchor condition strength, not multishot.

`H3_Multishot_AIO` and `H3_Multishot_MEMORY` from earlier versions are retired
- every lane they had is in v2 (the AIO's episode source, plate chain and audio
spine were folded in; MEMORY had nothing v2 lacks). Existing copies keep
working.

**4. Fill in the two panels**

- **MASTER CONTROLS** - resolution, frames per shot, steps. The shipped values (`1280x736`, `362`, `14`) are the verified recipe; leave them alone for your first render.
- **Script** - one prompt per shot, `---` between shots. Read [Prompting and boundary rules](#prompting-and-boundary-rules) before you write it; the join rules are the difference between a seamless take and a chain with clipped words at every seam.

**5. Queue**

`preview_first_shot` is ON by default, so shot 1 surfaces as soon as it is done and you can judge framing and voice before the rest of the chain commits. Output lands in `output/video/H3CHAIN/` as a 24fps video with a paired audio file.

**Then, for the FULL workflow** (`H3_Seamless_Chain_v2.json`), install the packs
it needs — ComfyUI validates **every** node class in a graph before it will
queue, so a missing pack stops the whole workflow, not just its own feature:

| Pack | Needed for |
| --- | --- |
| `ComfyUI_JoyAI_Echo_GGUF_Nodes` | the LLM prompt writer — **ships in this repo's release zip**, modified with attribution (see its NOTICE). Use that copy, not upstream: the workflow drives inputs upstream does not have. |
| [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | `continuity=context_pin`, the shipped default |
| [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) | the `beta57` scheduler the full workflow ships with |
| ComfyUI-sol-attn + comfyui-minimax-h3-blockcache-T8 | the VRAM/SPEED patch switches |
| [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) | the on-canvas script preview |

Every one of them can be removed instead — `INSTALL.md` in the zip gives the
one-widget change or node deletion for each. Highlights: no RES4LYF → set
`scheduler` to `beta` (measured cost: lip-sync 8/10 vs 10/10, all else equal);
no Motion-Context → `continuity=first_frame`.

---

## 2.7.2 - the actual reason Manager could not serve 2.7.0 or 2.7.1

- **Multi-statement lines removed (E702).** 2.7.1 dropped the `exec` that
  2.7.0 was flagged for, and the Registry flagged it anyway. Reading the
  publish log rather than guessing: the ONLY rule its scan reported was E702,
  50 multiple-statements-on-one-line, in 8 files. Both prior versions tripped
  it; 2.6.5 shipped the same style and predates the enforcement. All 33
  affected lines are now split, verified by comparing each file's AST before
  and after with positions stripped - the trees are identical, so the change
  cannot alter behaviour. `ruff --select E702,S102` is clean.
- No functional change of any kind in this release. If you are on 2.7.1 from
  GitHub or HuggingFace, you already have every fix; this exists purely so
  ComfyUI-Manager will serve it.

## 2.7.1 - installable from Manager again, and context_pin fixed on 0.34

- **2.7.0 was flagged by the Comfy Registry and never served.** Their security
  scan now rejects dynamic code execution in a published node, and
  `h3_interior_patch` recompiled a rewritten copy of `PackedLayout.__init__`
  to lift stock's first/last keyframe restriction. That mechanism is gone.
  The module now only *detects* where interior anchors come from: ComfyUI
  0.34+ places them natively, and on older cores the
  ComfyUI-H3-Motion-Context pack owns that job (it is already required for
  `continuity=context_pin`). If neither applies you get a message naming both
  remedies instead of a silent loss. First/last anchors are stock behaviour
  and were never affected. Nothing changes on 0.34, where the old path was
  already dead code.
- **context_pin: pinned blocks were collapsing onto frame 0 on ComfyUI 0.34.**
  Motion-Context's layout patch does two jobs - place interior video anchors,
  and translate the marked audio reference onto the target timeline. 0.34 made
  the first native, so 2.7.0's compatibility work stood the patch down
  entirely; but the node still emitted `resolved_frame_index=0` with the true
  position under a private key, and with the patch gone nothing applied it.
  Every pinned block landed on frame 0 while the console still reported them
  spread across the window. Joins lost their phase lock and the audio
  reference lost its end-alignment - quietly, with no error. **This fix lives
  in ComfyUI-H3-Motion-Context, not in this pack**: update that pack (a diff
  is attached to this release). Measured before/after on a 7-block pin:
  1 distinct position out of 7, versus 7 of 7 after.
- The 2.7.0 line "ComfyUI 0.34 ready" was true of this pack's own layout patch
  and wrong about `context_pin`. This entry is the correction.

## 2.7.0 - per-subject voices, flf_chain fixed, chain leveller, ComfyUI 0.34

- **Per-subject voice refs** — `voice_ref_2` / `voice_ref_3` on both samplers
  (appended last; saved graphs load unchanged). Each subject keeps their own
  voice across a chained scene instead of blending into one narrator. Verified
  blind across a full chain: no cross-speaker bleed. This was the most-asked
  request on the model page.
- **flf_chain fix — boundary plates no longer haunted by PLATE0.** With
  `continuity=flf_chain` the memory bank still ran, and its default
  (`bank_pinned=1`) pinned a reference clip of shot 1 into every later shot.
  Shot 1 opens on boundary plate 0, and in flf mode those bank clips are not
  named in the prompt — so the model treated shot-1 footage as content and
  mixed PLATE0 back in from shot 2 onward (user-reported: "it picks up PLATE0
  again"). The bank now stands down automatically in flf_chain, with a console
  line saying so. Plates alone carry the continuity in this mode, which is the
  whole point of the mode.
- **H3ChainNormalize** — post-chain texture and colour leveller. Addresses the
  2.6.0 known limit (the slow sharpening ratchet across long chains): run it
  after the chain and the take is levelled end to end.
- **refresh_pin splice alignment** — the re-encoded tail was landing 1-2
  latent frames early, so joins read as cuts and audio slipped. The splice is
  now correlation-aligned to the actual latent overlap.
- **x0 texture clamp dial** — `x0_clamp_window` (appended last), dose capped
  at 0.30. The eye-approved anti-dulling setting is the default.
- **Memory-sampler levers** — in-loop latent x2/x1.5 upscale,
  `refresh_renoise` (variance-matched splice), `pin_noise_ramp` (graded seam
  floor), `auto_chunk_ffn` (sol-attn chunking when VRAM is tight).
- **Schedule-split sampler (experimental)** — `sampler_2` + `sampler_2_at`
  on the memory sampler (appended last, `(off)` by default): one continuous
  sigma schedule, a second solver for the tail slice — e.g. euler for
  structure, then res_2s for the low-sigma refinement. Off = identical to
  before; being A/B'd, treat as experimental.
- **Motion-Context 0.4.0: not yet.** MC 0.4.0 (released 2026-08-26) moved its
  pinned audio from a patched reference block to an audio keyframe, and that
  does not yet compose with this sampler - context_pin fails at the shot-2
  boundary with a shape mismatch. The pre-flight now refuses 0.4.0 by name
  BEFORE any sampling and tells you to stay on 0.3.1; 0.4.0 support is the
  first item for the next update.
- **ComfyUI 0.34 ready** — 0.34 places interior keyframe anchors natively, so
  the pack's layout patch now probes the running core and stands down when it
  is not needed. Older cores keep the patch exactly as before.
- **Engine-aware writer** — the bundled JoyEcho writer now carries separate H3
  and LTX system prompts for every mode, selected by a new `engine` widget
  (appended last). H3 prompts use the `<d>[English] ...</d>` spoken-line spec;
  LTX prompts keep straight-double-quote dialogue for the TTS extractor.
- Widget range: `beat_seconds` max 15 -> 20 on H3LTXTakeControls.
- Fine-tooth fix batch: small correctness fixes across both samplers.

## 2.6.3 - one number for window length, and remote encoding that fails fast

### frames_per_shot now rules extend mode too

The MASTER CONTROLS panel had two silent authorities over window length: the
`frames_per_shot` widget in normal mode, and the `window` combo the moment
`take_seconds` was set - with the widget you had just typed sitting there
ignored. Now `window = auto` FOLLOWS the frames widget (snapped to the 17k+5
grid, and the console says so), an explicit `window` value still wins, and the
old VRAM-based sizing lives on as a new option, `fit this card (VRAM auto)` -
pick it when you want the fewest joins that will not thrash rather than a
specific length. One number, every mode. The shipped canvas is pinned to 243 so
existing setups render exactly what they rendered before.

### The remote encoder checks the address before the writer runs

A wrong or placeholder endpoint used to surface at the FIRST ENCODE - after the
LLM writer had already spent minutes producing a script that was then thrown
away. The node now checks the address the moment it executes: an empty box, the
untouched `OTHER-PC` placeholder, and an unreachable host each fail in about a
second with a message that says exactly what to fix.

### Remote encoding no longer pays a first-shot penalty

With the text encoder on another box, nothing ever evicted the models LEFT OVER
from the previous run - the reserve planner saw a nearly-full card, went to its
tight path, and the pool spilled into driver memory (measured: shot 1 at 65
s/it against 27 s/it once shot 2's normal eviction cleared a stale 15 GB
encoder). Both samplers now sweep leftovers once before the first DiT load
when the encoder is remote, and print what they cleared.

## 2.6.2 - the prompt the model was trained to read

MiniMax publishes the exact prompt format H3 was trained on (their
`VIDEO_PROMPT_WRITING_GUIDE`, and an agent skill carrying the full-reference
variant). This release closes the gap between that spec and what this pack
actually sent.

### Reference prompts are assembled in the documented order

`subject_definitions` has to establish `<Subject 1>` BEFORE the description
uses it. This pack appended the label sections to the END of the prose, so the
model read the description first and the definitions afterwards. The six
sections now go out in the guide's order, and two that were never emitted at
all have been added: `summary`, with the square-bracket task-type prefix
(`[reference generation + video continuation + audio reference]`), and
`non_diegetic_music: N/A` - nothing in the conditioning previously said "no
score". `H3_LEGACY_SECTION_ORDER=1` restores the old assembly.

### The classic sampler sends subject definitions at all

The tokenizer labels every reference `<Picture k>: ` / `<Audio j>: ` and then
appends your prompt. Only the memory sampler explained what those labels were;
the classic multishot sampler sent them bare, so references arrived unexplained
- the exact failure `subject_definitions` exists to prevent. It now emits them,
and distinguishes what they are: portraits are reference photographs, while the
chain frame carried between shots is declared as the shot's continuation
anchor instead of being described as a face reference.

### Keyframe modes send the alignment line

Keyframe modes do not use the reference sections - they use a documented
alignment instruction as the prompt's first line. Three paths sent none
(the classic sampler's chain frame, `first_frame` continuity, and the final
FFLF plate); the fourth had it with a hyphen where the trained text has an em
dash. All four now match the guide. `H3_NO_KF_ALIGN=1` disables them.

### Fixed: a negation in the conditioning, and an unresolved label

`retention_analysis` emitted "... and is never blended with `<Subject 1>`".
There is no negative branch at cfg 1.0, so that clause only put blending into
the conditioning - thirteen lines below a comment warning about exactly this
mechanism. The affirmative half already carried the meaning and is all that
ships now. Separately, `<Audio N>` claimed to be "the synchronized audio track
of `<Video 1>`" even when no reference video existed, which the guide
explicitly forbids; with no video it is now described on its own terms.

### Auto Refs: five characters no longer cast three

The prompt scan stopped at the first three characters it found. A five-character
script left two of them with no photographs while the writer still pointed all
five at "the reference photographs" - and a character pointed at photos that do
not exist renders as a random person. The scan now takes up to nine and splits
the model's nine reference slots across everyone who matched (two characters
keep three photos each, four get two, five or more get one), printing the split.
Fewer characters per run still holds a face more firmly, but every named
character now casts somebody.

### H3 Retake ships

`H3Retake` - redo one stretch of a finished clip and keep the rest - was written
but never included in the package. Load the clip's frames and audio, set a time
window, write a prompt for that moment; everything outside the window is frozen
as raw latents. Picture and sound are independent.

### Also in 2.6.2

- The prompt picker walks every registered `inspire_prompts` root, not just the
  first - a corpus added through `extra_model_paths.yaml` sat at `roots[1]` and
  never appeared.
- `rift_engine_script` accepts the writer's `{"prompts": [...]}` directly.
- Bundled writer rules: faces stay readable and each shot ends on the face
  (identity re-locks from a shot's closing frames, so a block that ends on the
  back of a head hands the next block a stranger); the strange thing never turns
  to the lens and the environment never reacts on cue; the model's own
  camera-motion vocabulary, with `Static Shot` as the affirmative way to hold a
  camera; the anti-merge rule rewritten affirmatively; "mood" dropped from the
  paragraph recipe and from the bundled example that was teaching it.
- Writer node: `num_ctx` is set for local Ollama endpoints (the 4096 default
  silently truncated long system prompts).

## 2.6.1 - hotfix for ComfyUI master

- **`AttributeError: MiniMaxH3ReferenceToVideo has no attribute '_encode_ref_audio'`**
  (GitHub issue #15). ComfyUI master moved that helper from a static method to a
  module-level function; the memory sampler's reference-video path called the old
  location. It now resolves whichever one this ComfyUI has. Nothing else changed.
- Also since 2.6.0's zip: AutoRefs runs before the writer, is gated by USE AUTO
  REFS, exposes `found` (drives the writer's `refs_attached`), and ships
  `on_no_match = no_reference` - a missing character folder warns and renders
  without photos instead of stopping the run.

## 2.6.0 - the extend take: one prompt, one continuous speech, as long as you want

Type a length. Get a take.

### The extend take

Set **`take_seconds`** on MASTER CONTROLS (or open the new
`H3_Extend_Take` workflow, which is v2 with it already set) and give the
writer ONE premise. The panel sizes a window for your card (`window = auto`
- the largest window whose activation pool fits with most of the weights
resident; wire the loader's MODEL into the panel for a real weight size) and
the number of windows that fills the time; the writer, in its new **extend
take** join style, writes ONE continuous speech and cuts it across the
windows at sentence boundaries; the memory sampler chains them under
`context_pin`. There is no airlock, no settle, no per-shot dialogue budget
to think about. H3 continues the speech across every join in its own voice
- no TTS.

Verified before shipping: seven writer-driven renders on a 32 GB card at
141/192/243-frame windows plus a 65-second, 7-window take, **every join
continued the speech** (zero repeats, zero clipped words), reviewed blind as
one uninterrupted take. Dead air at a join is a word-fill problem, not a
join problem: the writer is now steered to the upper half of its per-window
budget (measured: 11 words in a 10 s window left a 5 s hole; the steer took
the same config to 1.8 s).

- `take_seconds` = 0 (default) is the old behaviour exactly; saved
  workflows are unaffected (new widgets appended last).
- `audio_pin_frames` on the memory sampler: the audio reference window,
  independent of the picture pin (96 = the 4 s audio memory the JoyEcho
  ancestor of this sampler carried; measured neutral at n=1, ships 0).
- Standalone `H3ExtendTake` node for graphs that want the sizing outside
  the panel.

**Known limit (2.6.0):** the chain's texture ratchet is not fully solved for long takes - measured about +13% fine texture per join at 736x1280 with the anti-drift set on. Under ~4 windows (~30-40 s) it is slight; at 7 windows it is visible sharpening. Keep extend takes to ~4 windows for now; a pin-side fix is in progress for 2.6.1.

### Also in 2.6.0

- **Auto refs now reach the sampler on their own.** The REFERENCE IMAGES
  lane had two switches in series: USE AUTO REFS, and a REFERENCE gate
  downstream that silently discarded the auto refs unless you also flipped
  it - the log said `3 ref(s)` and the character rendered from prose alone.
  USE AUTO REFS alone is now enough; the gate is the MANUAL REFS gate and only
  guards the two LoadImage slots. `H3AutoRefs` gained a `refs_batch` output
  (every picked photo in one batch, so two or three matched characters no
  longer overflow a 3-slot chain), and `H3AnySwitch` passes an OFF gate's
  nothing through instead of erroring.
- **The writer no longer overrides your reference photographs.** Measured on
  the same seed: with photographs attached, the writer's identity sentence
  ("a woman in her thirties with dark hair tied back") rendered *that*
  person, not the one in the photographs; the same prompt with the sentence
  replaced by "looks exactly as in the reference photographs" rendered the
  referenced person. The writer has a `refs_attached` input (wired from USE
  AUTO REFS in the shipped canvases); when true its identity sentences point
  at the photographs and describe nothing about face, hair, age or build.
  Rules in `prompts/h3_refs_attached_rules.md`. If you use the writer with
  MANUAL refs, feed that input a true boolean.
- **Saved 2.5.x canvases keep working.** The VRAM/SPEED panel dropped its
  five dead toggles but keeps all eight output slots in the original order,
  because saved graphs link by slot index - the removed slots emit `False`.
- **Reserve planning fixes from the 24 GB test lab** - the bare-to-payload
  x1.6 no longer fires between the two samplers' differently-named payload
  signatures (it was inflating CORE-workflow reserves and leaving 129 MB of
  DiT resident); a first run at a new shape now borrows measurements only
  from the same quant family (GGUF pool measurements do not transfer to
  w4a8/int8, which run 1.5-2x the pool per token - a cross-family borrow
  under-reserved a first run straight into driver paging); the streamed
  master's embedded metadata is the API graph again, not the last shot's
  text.
- **Writer: silent shots can no longer echo the voice anchor.** A silent
  shot whose boilerplate said "visible lip movement clearly readable" while
  shot 1's voice rode along as `<Audio 1>` re-spoke shot 1's line word for
  word. The few-shot framing sentence no longer mentions lip movement, the
  silent-shot rule cites the render, and the writer warns by shot number if
  a silent shot's text still does.

---

## 2.5.5 - the memory release: renders that no longer gamble, RAM that no longer runs out

Four new memory systems, all measured, two of them automatic.

### Automatic: the driver-headroom rule (the random-slowdown fix)

The single biggest fix this pack has shipped. High-resolution renders would
randomly take anywhere from 27 minutes to 3 hours for identical work - same
seed, same settings, a lottery. The cause: when model weights plus working
memory fill the card past roughly 95%, the Windows driver starts demoting
GPU memory unpredictably, and whether your render crawled was luck.

The pack now detects that zone before sampling and deliberately streams a few
GB of weights instead of riding the ceiling - streamed weights are nearly
free on modern ComfyUI, the last few percent of VRAM are not. The lottery
render became 15 minutes, every time. Nothing to configure; you will see
"driver headroom" lines in the console when it saves you.

### low_ram_master: long chains without the system-RAM cliff

Chained renders used to hold every finished shot in system RAM until the
final join - tens of GB at the very last step, historically the point where
long renders killed whole machines. Switch `low_ram_master` ON and shots
stream to lossless disk staging as they finish; the final video is assembled
from disk through the exact same levelling math (verified: 42.8 dB against
the RAM path - the difference is codec noise). Peak RAM becomes about two
shots. The finished file's path comes out of the new `master_path` output.

### Remote text encoder: the 15 GB the render card never needed

The text encoder runs for seconds per shot and occupies 15+ GB the rest of
the time. The new **H3 Remote Text Encoder** node runs it on any second PC
with ComfyUI and this pack: encodes travel over the LAN, identical results
(verified across machines to float precision), and repeated scene text is
answered from a local cache without any network at all. Turn `remote_encoder` ON in the VRAM / SPEED
SWITCHES panel to use it; it ships OFF, so single-PC users see zero change.

### H3 TAE Decode: 2-second draft previews

A 9 MB tiny decoder turns latents into full-resolution draft frames in about
2 seconds, against roughly a minute per shot through the real VAE. Drafts
smear texture but composition and motion read clearly - built for seed hunts
and batch triage, never for finals.

### Also in 2.5.5

- **Speed Boosters panel** - Spectrum, TeaCache, block cache and ComfyUI's
  own EasyCache behind switches, each measured (-11% to -29%) and eye-tested,
  with honest per-switch notes about which ones can distort people. Block
  cache turned out inert at 14 steps (0 hits measured on every run, both
  cards) and ships OFF as of 2.6.0. Missing packs print an install
  link instead of breaking the graph.
- **New defaults for 16-24 GB cards** - 736x1280 (the model distorts faces
  below ~1 MP), 192 frames, 14 steps, curve-Q5_1 model, the full anti-drift
  dial set including chain_gain_control=flatten.
- **Dialogue that fits** - the writer is told each shot's real speaking time;
  overruns and silent scripts print warnings with numbers at generation time.
  Measured: lines over budget garble, lines under budget drag - matched
  budgets reviewed as "natural, fully intelligible."
- **Writer craft rules** from failed renders: silent shots with visible
  people state what mouths are doing (stops invented mumbling); revealed
  things are written as already present (stops mid-shot pop-ins).
- **Clickable title blocks** in every workflow with version and links, a
  plain-language notes pass, and two companion articles (quick start + the
  full settings manual).

---

## 2.2.5 - a chain that could refuse to start, an upscale that could eat the host, and one-click install from the Registry

### Fixed: chained shots could stall before the first sampling step

The auto-reserve learns how much activation pool a given shape needs by
measuring it. On a chained render the later shots often load *partially*
- the weights are already resident, so only part of them streams in. The
old code discarded every partial measurement as untrustworthy, which meant a
chained shot could never contribute what it learned. If the very first shot was
also partial, the cache stayed empty forever, the fallback reserve was too small
for the shape, and every subsequent attempt failed the same way. A render could
sit there refusing to start with no error to point at.

Partial measurements are now kept when the pool is genuinely large, and
clamped by a named floor (`_auto_cache_floor`) that only ever ratchets
upward, never down. Verified live on a six-shot chain: offload dropped from
4308 MB to 638 MB once the cache started learning.

The warning that fires when a shot asks for far more pool than is available
now distinguishes the two cases it was conflating. If this shape has been
measured before, it says so and gives you the number. If it is the first run at
this shape, it says *that* instead of implying something is broken.

### Fixed: `output_scale` was documented backwards

With an upscale model selected, `output_scale` is not a multiplier
on top of it - the model applies its own fixed factor (4x for the
ESRGAN family) and `output_scale` is the **final** size
you want, not an extra step. The tooltip said otherwise, in both samplers. Both
are corrected, and `upscale_model_name` now explains the interaction
rather than leaving you to discover it.

### New: the upscale says what it will cost before it runs

Decoded frames accumulate in host RAM until the master is joined, so the bill
is per-shot multiplied by shot count. A 1344x768 chain through a 4x model is
5376x3072 per frame, and an `output_scale` left at a value that
looked harmless has taken machines down at the join - after all the
sampling was already paid for.

Before sampling starts you now get the projected dimensions, MB per frame, GB
per shot and GB total, measured against actual free RAM:

```
[H3Memory] upscale will produce 5376x3072 (4.0x): 198 MB per frame
[H3Memory] WARNING: that will not fit. Frames are held in host RAM
until the master is joined - lower output_scale to 1.60
or lower, or turn the upscaler off.
```

### New: the folder walk prints which index is which file

Queueing a folder told you nothing about the mapping between
`EPISODE INDEX` and the file it would pick, so a cancelled run left
you doing off-by-one arithmetic against a single log line to work out where to
restart. The ordered list now prints once per folder per session, and each job
reports its own position and the index to set to resume there. It re-prints if
the folder's file count changes, which is the one case where a remembered
listing would mislead.

### Fixed: garbled speech - the writer was never told how long a shot is

`num_frames` on the LLM Enhance node defaults to `0`,
and it shipped as `0` in the bundled workflow. At zero, the writer is
told **nothing** about clip length, so it sizes every spoken line
for the "~10 second clip" assumed in the system prompt regardless of what you
actually render. Too long for a short shot is crammed, garbled speech; too short
for a long one is dead air.

Three things now close that:

- `num_frames = 0` prints a warning naming the widget and telling
you to match it to the sampler's `frames_per_shot`. It used to pass
in complete silence.

- The bundled workflow ships with `num_frames` set to
**243**, the same value its `frames_per_shot` already
uses, so a fresh install is correct out of the box.

- Shots whose dialogue overruns the budget are reported at generation time
with the numbers, instead of being discovered forty minutes later by ear. The
check is symmetric: a script that comes back with **no dialogue at
all** is also flagged, because that renders as a silent slideshow.

### Fixed: `revise` asked for lines that could not fit

Chained styles spend about five seconds of every block on the replay, airlock
and settle, so the speakable span is shorter than the clip. The main writer
subtracted that. `revise` did not - it sized lines against the
raw clip length, from its own separate copy of the arithmetic.

On a 243-frame chained shot that meant `revise` asked for
**12-20 words where only 6-10 fit**: double the budget,
every line, which is exactly the overrun that renders as garble. It also never
checked its own output, because it returns long before the check the main path
runs.

Both paths now call one `_speakable_budget()` and both check their
result, so they cannot drift apart again. The duplicate arithmetic is why they
had.

### Changed: the story writer stops prescribing creativity

The system prompts had accumulated a five-beat dramatic arc, a banned-cliche
list, a "land near seven shots" target and a dialogue quota. None of those are
properties of the model - they are one person's taste, hardcoded, and they
were flattening every brief toward the same shape.

They are gone. What remains is what the renderer actually needs: valid JSON,
identity restated verbatim per shot so the room does not drift, literal physical
description over mood language because abstract adjectives have nothing to
render, and the note that audio is half the model. The story, its length, its
tone and whether any given shot speaks are the model's call now.

### Changed: the node menu no longer says "Rift"

Four nodes sat in a category called `Rift`, which means nothing to
anyone who is not me. They are in `H3/script` now, the labels drop the
prefix, and ten console messages say `[H3 Multishot]` - the name
you actually installed - instead of `[Rift]`.

> **Saved workflows are unaffected.** Categories are
never written into a graph and display names resolve fresh on load. The
underlying class names are deliberately *unchanged*, because those strings
*are* written into every saved `.json` and renaming them would
turn your nodes into red missing-node boxes.

### New: installable from the ComfyUI Registry

The pack is published as `comfyui-h3-multishot`, so ComfyUI-Manager
can find and update it without a manual clone. The zip on this page stays the
complete bundle - both node packs, the workflows and the docs - and
remains the right choice if you want everything in one drop.

The dependency list is deliberately empty. Every import this pack uses
(`torch`, `torchaudio`, `psutil`,
`av`, `numpy`, `PIL`) is already guaranteed by
ComfyUI itself, and a custom node that declares `torch` can talk pip
into replacing a working CUDA build with a CPU wheel.

### Documentation: drift on long chains, and the dials that fight it

`SETTINGS.md` gains a measured section on why chained shots accrete
detail - each conditions on the previous shot's own output, so invented
texture compounds - and which controls actually counter it. Measured over
ten shots at 960x544:

- `memory_frames = 0` is the big one. The bank's RECENT slots feed
accreted output forward on top of the pin.

- `master_normalize = luma+contrast` levels brightness
**and** contrast of the finished chain.

- `pin_renorm = on` holds each pinned latent at shot 1's sigma.

- `chain_gain_control = flatten` on chains longer than about five
shots - texture ratchets roughly 1.3x per join.

Residual with all of those on is about **1.02 per hop**. Not
zero. Long chains still drift, slowly. `pin_noise` is small,
scene-dependent and gets worse above `0.10`; it is not the fix it was
described as in 2.1.5.

---


## 2.2.4 - references for more than one person, a folder you can queue, and video references

Four things people asked for or tripped over, plus the documentation that
should have prevented two of the questions.

### Fixed: reference images of different people blended into one face

If you gave the sampler reference pictures of two characters, it rendered
something that resembled neither. That was not you holding it wrong. Internally
every reference picture was declared to the model as
`<Picture N> is a reference photograph of <Subject 1>` -
*the same* Subject 1, every time. With one character that is correct and
it is why single-character references work well. With several you were telling
the model that photographs of different people all showed one individual, and
it produced the average.

There is a new `reference_subjects` field on the sampler. Leave it
empty and nothing changes - every picture is one person, exactly as before.
Fill in comma counts in picture order to group them:

```
reference_subjects: 3,3 pictures 1-3 are person A, 4-6 are person B
reference_subjects: 2,2,2 three people, two pictures each
```

Each group is then declared its own subject, with an explicit instruction that
it is never blended with the others. Only Subject 1 is described as speaking,
because H3's voice conditioning is single-speaker.

### New: point the prompt source at a folder and queue the whole thing

`start_index` has always selected a block *inside* one file.
That works for LPFF-style prompt batches, where one file holds many prompts. It
does not work for H3 scripts, because in an H3 script `---` separates
**shots within one scene** - concatenating scenes into one file
would fuse them into a single enormous take.

Turn on the new `walk_folder` switch and `start_index`
selects **which file** instead, walking the chosen folder in sorted
order and emitting the whole script. Wire it to a Primitive set to
`increment`, queue thirty times, and thirty finished scenes render
unattended.

### New: hand H3 an existing clip as a video reference

There is now a `V2V REFERENCE` lane in the workflow - Load
Video into Get Video Components into the new **H3 Reference Video**
node, then into two new sampler inputs. It ships muted; un-mute the three nodes
to use it.

**Read what this is before you wire it.** H3 is told a video
reference is "a clip from an earlier moment of this same continuous
scene" and asked to keep its framing, camera distance, room contents and
colour temperature. It is **scene and appearance conditioning**. It
is **not motion control** - there is no pose, depth or optical
flow path in H3, so your subject will not copy the movement in the clip. It
borrows the place and the look, not the action.

Keep the window short. References are subsampled to 2 fps and then ride
through *every* sampling step, so a 25-second clip is roughly 50 reference
frames of permanent per-step cost. The H3 Reference Video node trims to a window
and prints what that window will cost before you commit to it. Requires ref2va;
fl2va has no reference rows and ignores video references entirely.

### Fixed: an empty prompt dropdown that blamed the wrong thing

The prompt source finds `.txt` files through the
`inspire_prompts` folder path, which is registered by
**comfyui-inspire-pack**. Without that pack installed the path does
not exist, so the dropdown came back empty however many prompt files you had,
and the error blamed the prompts folder. It now tells you which of the three
fixes applies: install the pack, register the path yourself in
`extra_model_paths.yaml`, or set `manual_path` and ignore
the dropdown. The dependency is written into INSTALL.md as well.

### Documentation: what has to change between shots

Several people have hit chains where shot 3 comes back as a near-copy of shot
2. The prompting guide is partly responsible: rule 5 says to repeat descriptions
word-for-word, and never said what must *differ*. Rule 5 is only about
appearance and the room. It is not an instruction to restate the action, and
when it gets read that way most of shot 3 is byte-identical to shot 2 - and
the model is separately instructed to preserve the subject, the room and the
colour temperature, so "keep everything the same" wins.

There is now a sixth rule covering it. The short version: each shot's action
must leave the world in a state the previous shot's world was not in, physical
and irreversible, not a mood or a camera move. If you can swap two shots' action
lines and the script still reads correctly, the model cannot tell them apart
either.

### Documentation: fl2va is not a smaller file

Both this guide and INSTALL.md described fl2va as "lighter and
faster", which everyone reasonably read as a size claim while choosing a
file for a 24 GB card. It is not. **fl2va and ref2va are exactly the
same size at every quant level.** "Lighter" only ever meant
fewer tokens per sampling step, because there are no reference rows riding
along.

What actually separates them was never written down: **fl2va lands on a
supplied frame and ref2va only nudges toward one.** Measured against the
same frame, 26.35 dB on fl2va versus 16.15 dB on ref2va with a keyframe
at 6 turbo steps - and 16.81 dB at 20 stock steps, which rules out the
sampler and leaves the checkpoint. fl2va can also take a first *and* a
last frame and plan a camera move between them, which ref2va cannot do at all.
So: ref2va when identity or voice must persist, fl2va when a shot must start
exactly where the last one ended.

### Improved: the story writer, for thin briefs and long shot counts

Give the writer a couple of characters and a genre with no plot, ask for
fifteen shots, and it tended to return atmospheric moments rather than a story.
That was the system prompt's fault in four specific ways, all now fixed.

It was **calibrated to about seven shots and argued against more**
- "most scenes land between 4 and 10", "go higher only when
the story clearly has that many real beats", plus a padding test. Asked for
fifteen, the model was simultaneously told to produce exactly that count and that
this many was probably padding, and it resolved the contradiction by rationing one
thin premise across the shots. It is now told the opposite: **a high count
is an instruction to invent more story** - another location, a second
complication, a character who arrives partway through, a reversal that changes
what the earlier shots meant. The no-padding rule is unchanged and is exactly the
point; the way to satisfy a high count is to invent enough real plot that no shot
is padding.

When the brief names **only characters and a genre**, the writer is
now told plainly that it is the author: decide who wants what, the incident that
starts it, the complication, the turn and the ending *before* writing any
shot. Returning a series of moods involving the named characters is called out as
the most common failure on a thin brief.

**Dialogue density** now has a floor. Speech was marked optional
per shot and non-speaking shots were actively encouraged, with nothing on the
other side, so long pieces came back nearly silent. It now aims for two thirds or
more of shots to carry speech, with silent shots as punctuation.

And a new **FIT THE SHOT** section covering *both* dialogue
and action. Overrunning the clip is what produces crammed, garbled speech and
distorted motion, so the writer now budgets each shot - settle, action, line
at an unhurried pace, a beat to land on - and cuts the action rather than
speeding up the speech. One clear physical action per shot; if more than one is
written, split the shot.

### Note on saved workflows

Every new control in this release is appended at the *end* of the
node's input list. Widget values are stored positionally, so inserting a control
anywhere else silently shifts every value after it in workflows you have already
saved. Your existing graphs load unchanged.

---


## 2.2.3 - three ways a long render could waste itself, and a switch that did nothing

Everything here is a fix. Nothing changed about how you use the workflow.

### Fixed: long multi-shot renders died at the very last step

A long job would sample every shot, upscale every shot, finish master
normalize - and then die while assembling the result, with nothing
written. Every expensive stage had already succeeded.

```
RuntimeError: DefaultCPUAllocator: not enough memory:
you tried to allocate 33791016960 bytes.
```

That is **host RAM, not VRAM**, and it is one contiguous request.
Upscaled frames were held on the host as fp32 for the whole run, and the final
assembly then allocated a *second* complete timeline while the first was
still alive. Peak was twice the timeline, and the timeline grows as
`shots × frames × height × width × 3 × 4 bytes`
with the upscale factor squared. Six shots of 243 frames upscaled is a 36.5 GB
timeline and a ~70 GB peak, which no 64 GB machine can satisfy.

Three changes: the timeline is now assembled into one preallocated buffer,
releasing each shot as it is copied (bit-exact - the same bytes as before);
frames are parked as fp16, which is well clear of the 8-bit the encoder writes
anyway; and master normalize upcasts to fp32 for its arithmetic and writes back,
so the memory saving costs no precision.

**Effect:** six shots of 243 frames upscaled drops from ~70 GB
peak to ~21 GB. Twelve shots of 192 frames drops from ~58 GB to ~31 GB.
Both now fit in 64 GB with the per-shot upscale unchanged. Diagnosed by a
second operator on a 3090 box; thank you.

### Fixed: the auto-reserve inflated its own estimate until the render spilled

If your chains got slower as they went - shot 1 fine, shot 3 sluggish,
shot 4 crawling - this is why. After each shot the pack measures how much
activation memory that shape needed. It subtracted the weight bytes
*currently resident* rather than the model's full size, so whenever the
model was only partly loaded the measurement came out too large by exactly the
amount that had been offloaded. That number then raised the next shot's reserve,
which left less room for the weights, which offloaded more.

```
shot 1 recorded 6.4 GB true 6.4 (full load - correct)
shot 2 recorded 8.0 GB true 5.3 (2749 MB offloaded)
shot 3 recorded 8.9 GB true 4.3 (4784 MB offloaded)
shot 4 asked for 19.9 GB -> clamped -> 262 s/step
```

The overstatement equals the offload every time. The measurement now uses the
model's full size, and a shot that offloaded weights records nothing at all
- it measured a spill, not an activation pool.

The cache also only ever grew, so a single bad shot poisoned a shape
permanently. Entries written by earlier versions are **dropped once**
on first load; you will see a line saying so, and those shapes re-measure on
their next run.

This does not make an over-committed render fit. If a shot
honestly needs more memory than the card has you still have to lower
`frames_per_shot` or resolution - the difference is that the
warning now fires on true numbers instead of the reserve quietly climbing.

### Fixed: the sol_attn and chunk_ffn switches did nothing

Reported by **sdktertiaire2**. Those switches shipped ON, but the
nodes they gate ship *bypassed* - they need third-party packs that
cannot be bundled (`ComfyUI-sol-attn`,
`comfyui-minimax-h3-blockcache-T8`). A toggle routing into a disabled
node changes nothing and warns about nothing. That contradictory default was
mine.

Both now ship OFF so the panel states what the canvas actually does, and the
panel is labelled with the fix: **install the pack, then select the node
and press Ctrl+B to un-bypass it.** The switch only routes; it cannot
enable a bypassed node. If you do not want those packs, leave the switches off
and lose nothing - they are speed and memory optimisations, not quality
features. Everything renders identically without them, just slower.

The nodes stay bypassed on purpose. Enabling them by default would hard-fail
every install that lacks the optimizer packs.

---


## New in v2.2.2

One fix, reported by a user against 2.2.1.

### Fixed: reference renders forced glasses onto the subject

Reported on Civitai: a reference image produced a character wearing thick black
frames in every shot, and prompting to remove them changed nothing.

The pack injects a `retention_analysis` block alongside reference images so that
identity and voice stop drifting across a chain. That block was hardcoded to say
the subject *"retains the same face, skin, hair, **glasses** and wardrobe"* - so
every ref2va render was instructed to keep glasses regardless of the prompt.

It could not be argued with either, because the sampler runs on a `BasicGuider`:
**cfg 1.0, no negative branch**. There is nothing for a negative statement to
subtract from, so "remove the glasses" simply put the word into the conditioning
a second time.

The block now says "the same face, skin and hair" and names no accessories.
Eyewear, hats and jewellery are wardrobe choices that belong to the prompt.

**The general rule this exposes:** anything hardcoded into unconditional
conditioning is permanent from the user's side. At cfg 1.0 a default that names
a thing can never be prompted away. Phrase everything positively - "clear
unobstructed eyes", never "no glasses".

---

## New in v2.2.1

A bug-fix release. Four defects, three of them reported from real renders on
real machines rather than found in review.

### Fixed: two crashes that only appear on long or high-resolution chains

Both were `DefaultCPUAllocator: not enough memory` - system RAM, not VRAM - and
both killed the job after it had already done the expensive part.

- **`master_normalize` allocated the entire finished timeline four times over.**
  A 12-shot 1088x1920 chain tried for a single **31.2 GB** block and died on a
  24 GB machine after 81 minutes. Worse, one of those four copies was pure
  waste: the code rebuilt a tensor byte-for-byte identical to one already in
  memory purely to compute two numbers for a log line, then discarded it. Every
  statistic the function needs is one-dimensional. It now measures per shot and
  releases each input as it is consumed, so peak memory is one finished timeline
  plus one shot instead of four timelines. Output is **bit-identical** - verified
  against the old implementation across every colour mode and median width, on
  the pixels and on the log strings.
- **The upscaler was handed every frame at once.** 243 frames at 1472x2560 is an
  **11.0 GB** float32 allocation on top of the input. It now runs in chunks into
  a preallocated output, so peak is one chunk rather than the whole batch.
  Bit-identical, verified at two chunk sizes. Tune with `H3_UPSCALE_CHUNK` if you
  want to trade memory for a little speed; default 16.

If you have been unable to finish long chains, this is why.

### Fixed: auto-reserve could clamp itself into a hard crash

On a 12-shot 1280x736 run, shot 1 loaded completely and rendered. Shot 2 carries
the context pin **and** the reference rows, so its activation pool requirement
roughly doubles - the node correctly costed it at 18.2 GB. It then clamped the
pool to 9.4 GB "so the weights still load completely", and the weights loaded
**partially anyway**, 401 MB offloaded. Neither constraint was met and the render
aborted inside a CUDA kernel, taking the whole ComfyUI process with it.

Two causes, both fixed:

- The keepout was **384 MB**, which did not cover ComfyUI's own buffer plus the
  difference between `get_free_memory` and what it treats as usable - about
  400 MB short. Now **1 GB**.
- The "this is tight" warning compared the clamped reserve against a *sibling
  shot's* measurement rather than against what this shot had just asked for. Shot
  1's 9.1 GB made a 9.4 GB clamp look fine while the payload needed 18.2 GB, so
  nothing was printed. It now compares against the actual request and says
  plainly that this usually dies inside a kernel and takes the server with it,
  and that raising the reserve cannot help because the memory is not there.

**The trap worth knowing regardless of this fix: shot 1 succeeding tells you
nothing about shot 2.** Shot 1 has no pin and no references. If shot 2 will not
fit, lower `frames_per_shot` or the resolution, or load a smaller quantisation.

### Fixed: `start_image` on the memory sampler silently did nothing

Reported from the field. `start_image` on `H3MultishotMemorySampler` is an
identity **reference row**, not a first frame - and on an fl2va checkpoint, which
has no reference rows, it is built and then ignored entirely. Meanwhile the
sibling node `H3MultishotSampler` has an input with the **same name** that really
is an I2V first frame, which is where the reasonable expectation comes from. The
combination produced no warning at all. It now prints one naming both ways to
actually start shot 1 on a picture: use `H3MultishotSampler`, or set
`continuity=flf_chain` here and feed `keyframe_images`.

Note that `continuity=first_frame` is also not about your image - it hands over
the *previous shot's* last frame, and does nothing on shot 1.

---

## New in v2.2.0

Everything since **2.1.2**, which is where most people still are. Seven point
releases in one: five separate defects that stopped the workflow running, a
measured campaign against chain drift that changed the shipped defaults, and a
whole ComfyUI version this pack could not previously run on.

*(2.1.9 on GitHub and HuggingFace is this same content under a smaller number,
tagged an hour earlier.)*

**Run-blocking, all user-reported or found by finally testing what we ship:**

| | |
|---|---|
| the Audio Spine produced static on ComfyUI 0.32.0 | fixed in 2.1.7 |
| naming an mmproj file broke GGUF text encoders | fixed in 2.1.7 |
| `The value 1 for reference_image_size is not available` | fixed in 2.1.6 |
| the full workflow needed a node from a pack not in the zip | fixed in 2.1.8 |
| **`H3_Seamless_Chain_CORE` could not be queued at all** | fixed in 2.1.9 |

**New controls:** `pin_frames`, `pin_noise`, `pin_renorm`, and
`master_normalize`'s `luma+contrast` mode.

**Changed defaults that change your output:** `memory_frames` 2 → **0**, and
`join_anchor_noise` / `handoff_release` to 0 because they are inert under
`context_pin`.

**New runtime:** ComfyUI **0.32.0**, which needed real work — its
`ModelSamplingAV` carries the audio half of the latent on a different scale.
0.30.0 is unchanged and still supported.

Every section below is the original release note for each of those versions,
newest first.

---

## New in v2.1.9

### Fixed: `H3_Seamless_Chain_CORE` could not be queued at all

```
value_smaller_than_min: Value 0.0 smaller than min of 1.0 - output_scale
```

CORE's sampler still carried the widget array it was saved with **before 2.1.3
removed the four `two_pass_upscale` dials**. Nineteen saved values against the
class's fourteen live widgets, so the frontend put `false` into `output_scale`
(minimum 1.0) and `1.5` into `save_every_shot`, and the server rejected the whole
prompt. The workflow advertised as the one with no third-party dependencies — the
safest thing for a new user to open — has been un-runnable since 2.1.3.

It was never caught because CORE had **never been rendered end to end**. Our own
release checklist said "one CORE render before posting" and that step had been
carried, unticked, through six releases.

The array is now generated from a name→value map resolved against the server's
schema, and the map is stored on the node so the pack's JS can re-apply it by
name — the same treatment the full workflow's sampler got in 2.1.6. CORE has now
rendered: three chained shots at 960x544, 370 frames, picture and audio, holding
framing, wardrobe and lighting across both joins.

All three bundled workflows are now submit-tested against a live server as part
of the release routine, not read.

### Bypassed nodes now say what they need

Four nodes ship bypassed because their packs are not in the zip, and a missing
class fails the entire prompt. That is the right default — but a bypassed node
sitting on the canvas invites you to un-bypass it, and doing that without the
pack installed breaks the workflow with no explanation. The reporter's point, and they are
right.

Now it says so in four places, in order of how hard they are to miss:

| | |
|---|---|
| the group title | `VRAM PATCHES - bypassed: install the pack named in each title BEFORE Ctrl+B` |
| each node title | `attention patch (bypassed - needs ComfyUI-sol-attn)`, and so on |
| a note above the cluster | install first, then `Ctrl+B`; without it the whole workflow stops queueing |
| the VRAM / SPEED note | the full table with repository URLs, and the two-step rule |

The two-step rule is the part people lose: un-bypassing a patch node changes
nothing on its own, and flipping its switch while the node is still bypassed
changes nothing either. Both, in that order, after installing the pack.

`SCRIPT PREVIEW` is the odd one out — it is a leaf, so leaving it bypassed costs
you the on-canvas preview and nothing else.

**Why not just bundle the packs?** The writer pack is bundled because it is
modified and pinned. These three are not: shipping a second copy of
Custom-Scripts, sol-attn or blockcache-T8 inside the zip would shadow whatever
the user already has installed and freeze it at the version we happened to
vendor. Naming them and linking them is the honest version.

---

## New in v2.1.8

### Fixed: one node in the workflow came from a pack that is not in the zip

`SCRIPT PREVIEW` is `ShowText` from **ComfyUI-Custom-Scripts**, and it shipped
*active*. A node whose class is missing serialises as `class_type: null` and the
server rejects the **whole prompt**, so without that pack installed the workflow
could not be queued — for the sake of an on-canvas text box. INSTALL.md said to
delete the node, which only helps someone who reads it before pressing Queue.

It now ships **bypassed**, the same remedy the three accelerator nodes got in
2.1.4. A bypassed node is dropped from the prompt entirely. Install
Custom-Scripts and `Ctrl+B` the node if you want the preview; the writer feeds
the sampler either way.

Found by mapping every node type in all three bundled workflows to its owning
Python module against a running server. That check is now part of the release
routine instead of something I do after a user tells me. The other two workflows
were already clean, and everything else in the full one is either core ComfyUI,
this pack, the writer pack **that is in the zip**, or one of the three bypassed
accelerators.

### Verified end-to-end on ComfyUI 0.32.0

Not by reading the graph — by loading the shipped file in a browser on a 0.32.0
install and pressing Queue. Twice: once through the prompt-writer lane exactly
as shipped, and once with a hand-written script.

- loads with no missing node types, and every sampler widget lands on its own
  name (`memory_frames` 0, `master_normalize` luma+contrast, `pin_renorm` on) —
  the shift that caused `The value 1 for reference_image_size is not available`
  cannot reproduce
- no null `class_type`: the bypassed preview and the three bypassed
  accelerators are all dropped cleanly
- three chained shots at 960x544, 124 frames each → **328 frames**, exactly 372
  minus the two 22-frame `context_pin` head trims
- a reviewer given the clip cold, with no idea how it was made, read it as
  **one continuous static take**, found no cut or jump anywhere, transcribed all
  three lines of dialogue, reported the lip-sync as matching, and found no shift
  in framing, colour, brightness or wardrobe and no hiss, dropout or click in
  the audio

0.30.0 remains the version everything else here was measured on; both are now
tested before release.

---

## New in v2.1.7

Three things that stopped the workflow running. All user-reported, all
reproduced, all fixed and verified by render rather than by reading the code.

**Everything below was verified on ComfyUI 0.32.0**, not just 0.30.0. Two of
these three only ever appear on 0.32, which is why they survived several
releases.

### Fixed: the Audio Spine produced static on ComfyUI 0.32.0

`guide_audio` came out as hiss while the same file through `voice_ref` or the
native node was perfect. Reported against 2.1.2, still present through 2.1.6.

ComfyUI 0.32.0 introduced `ModelSamplingAV`, which carries the **audio half of
the packed AV latent scaled onto the video schedule** — `process_latent_in`
multiplies it by `shift / audio_shift` (12/3 = **4** for H3) and
`process_latent_out` divides it back. Everything this pack injects into the
sampler's latent is in the stream's *native* domain, so on 0.32 it landed 4x
too small and decoded as broadband noise. On 0.30.0 there is no such scaling,
so the same code was correct — which is why it never reproduced here until a
0.32 rig existed.

Three injection sites had it, not one:

| source | used by |
|---|---|
| the encoded spine | Audio Spine |
| the previous shot's audio tail | `audio_lock`, latent handoff |
| the encoded room tone | onset guard |

All three now scale at the point of use, reading the factor off the live
`model_sampling` object so a changed sigma shift stays correct. `getattr`'s 1.0
default leaves 0.30.0 byte-identical.

Verified on 0.32.0 with a real 44.1 kHz stereo voice track: loudness-envelope
correlation against the guide **+0.965**, speech-band energy 34.2% against the
guide's 39.3%, and a blind listener transcribing the guide's words with "clean,
no background hiss". Before the fix the same render was hiss.

### Fixed: naming an mmproj file broke GGUF encoders

    mat1 and mat2 shapes cannot be multiplied (3680x1152 and 3456x1152)

thrown at the handoff into shot 2, GGUF encoders only, unaffected by resolution
or by turning every image input off.

Setting `mmproj_name` explicitly took a different code path than `(auto)` and
skipped the vision key-renaming step entirely — so the vision tower loaded
under raw llama.cpp names (`v.blk.*`, `mm.*`) that nothing downstream reads,
the merger was never populated, and the first matmul touching vision features
had the wrong width. Same file, two loaders, **19 key names in common out of
351**.

The bitter part: `mmproj_name` is documented as the reliable escape hatch for
when filename pairing fails, and it was the broken path.

It now runs the same post-processing as `(auto)`, using ComfyUI-GGUF's own key
map rather than a copy, so it follows their changes. Verified: an explicitly
named file now yields a state dict identical to `(auto)` — 351 tensors, every
shape matching. A new guard also fails by name if a chosen mmproj produces no
`visual.*` tensors, instead of dying in a matmul twenty minutes later.

### Fixed: `The value 1 for reference_image_size is not available`

The shipped workflow's saved widget values were written for a layout that did
not present `sampler_override` and `scheduler_override` as widgets. The current
schema does, so everything from index 27 read two slots early and
`output_scale`'s `1.0` landed in `reference_image_size`, a combo of
`match`/`max`. This affected **2.1.3, 2.1.4 and 2.1.5**.

The array is now generated from a name→value map resolved against
`/object_info`, and the same map is stored in the node's properties so the
pack's own JS can re-apply by name if a future schema change shifts anything.

### Also: `memory_frames` now defaults to 0

The bank's *recent* slots hand each shot's accreted output forward as reference
images on top of the latent pin, so invented detail compounds. Measured over ten
shots at 960x544, moving 2 → 0:

| | 2 | 0 |
|---|---|---|
| texture per hop | 1.055 | **1.022** |
| chroma per hop | 1.086 | **1.039** |
| framing correlation at shot 10 | 0.976 | **0.995** |
| drift acceleration | 4.2% → 6.7%/hop | **2.3% → 2.0%/hop** |

The last row matters most: at 2 the drift *accelerates*, which is what a runaway
loop looks like. At 0 it holds flat.

The obvious worry was motion continuity, since the recency slots exist to carry
it. Tested on a scene with continuous large-amplitude movement: anchor-only
retained motion **slightly better** (−5.9% vs −6.5% over four shots) with better
framing (0.983 vs 0.971). The cost does not exist. If a busy scene ever does
lose continuity between shots, raise it to 1.

`join_anchor_noise` and `handoff_release` now ship at 0 — both are **inert**
under `context_pin` (one noises keyframes the mode never creates, the other
belongs to `latent_handoff`) and non-zero values read as tuned settings while
doing nothing.

---

## New in v2.1.6

**Chained shots stop getting brighter-edged every hop.** Not by the route
2.1.5 claimed - see the correction below.

`master_normalize` matched every frame's MEAN to one global target, which is
why a chain shows no brightness step. It was also masking a second drift it
never touched. Measured on a 3-shot chain at 960x544: mean held flat, 27.31 ->
27.43, while the DISTRIBUTION stretched - p25 fell 6 -> 2 and p95 rose 85 -> 96.
Contrast climbing every hop, re-centred each time, and handed to the next shot
as a higher-contrast starting point.

**New: `master_normalize=luma+contrast`** (now the default) matches the spread
as well as the mean. Rescaling amplitude about each frame's own mean is an
affine remap: it moves no edges, so it is not the blur this pack has always
ruled out for texture drift.

Texture growth per hop, from a log fit across all shots. 1.000 is no accretion:

| where | `luma` | `luma+contrast` | contrast spread |
|---|---|---|---|
| 960x544, in-render, 124f x 4 | 1.126 | **1.047** | 11.2% -> 0.3% |
| 960x544, 243f x 3 | 1.199 | **1.064** | 12.2% -> 0.4% |
| 640x352, 243f x 3 | 1.130 | **1.055** | 7.0% -> 0.2% |

The baseline ratchet scales with canvas (1.130 at 640x352, 1.199 at 960x544);
after normalising it stops caring (1.055 vs 1.064). There is nothing in it to
tune per resolution - it works on decoded frames with per-frame statistics
against one global target.

The target anchors to **shot 1**, not the timeline median. Contrast only
ratchets upward, so shot 1 is the one frame-set with nothing accreted onto it;
a median target pulls shot 1 UP to meet the drift (+11.7% texture, for no
benefit) where anchoring to shot 1 leaves it untouched (+0.1%) and only ever
pulls later shots down.

1:1 crops of the last shot show no loss of real detail - lamp vent slots, hinge
rivets, hair strands and knit weave all survive. What leaves is the invented
crispness.

### What is left, honestly

About **1.05 per hop**. That residual is spatial accretion, and this pass
cannot reach it: `master_normalize` runs on the finished master, outside the
feedback loop, so it cleans what you see while the next shot is still handed
the inflated pin. Four shots is slight. Ten shots is roughly +50%. If you are
chaining long, expect it.

### Correction to v2.1.5

v2.1.5 said `pin_noise=0.05` fixed this. **It does not.** That was measured on
two seeds of a single scene at 640x352 - a scene whose background was nearly
black and which barely ratcheted to begin with - and the control that would
have caught it, `pin_noise=0.00` at the reporting user's own resolution, had
never been run. With it run:

| canvas | 0.00 | 0.05 | change |
|---|---|---|---|
| 640x352 | 1.131 | 1.111 | -1.8% |
| 960x544 | 1.211 | 1.201 | -0.9% |

Both on a detail-heavy scene. The dial is small and scene-dependent, it cannot
touch the dominant drift in a busy frame, and above 0.10 it gets **worse** (0.20
measured 1.228 against a 1.211 control). Its range now stops at 0.10 and its
tooltip says all of this. It stays in the pack because it costs nothing and
does help where the ratchet is already small; it is not the fix.

### Not tested

Portrait canvases. Everything above is landscape - 640x352 and 960x544. The
mechanism is resolution-independent by construction and the two landscape sizes
agree, but 768x1344 and 736x1280 have not been measured and are not claimed.

### Measuring this yourself

Texture comparisons are only meaningful **while the framing holds**. If the
model cuts to a different setup, texture reflects content and the number is
meaningless - one portrait run here scored a flattering 0.878 per hop purely
because shot 3 cut to a close-up of a film reel. Correlate each shot's mean
frame against shot 1's before trusting any of it; a held framing sits above
0.95.

That cut is worth a writing rule of its own: **do not name a nearby object in a
shot's closing beat.** *"She glances down at the reel"* reads as a request for
a shot of the reel. Keep closing beats on the speaker's own body.

---

## Fixed in v2.1.4

Both of these are user-reported, both reproduced, both fixed and verified.

- **The main workflow would not queue without two third-party packs.**
  `H3_Seamless_Chain_v2.json` ships three optional accelerator nodes -
  `sol_attn`, `chunk_ffn` (from **ComfyUI-sol-attn**) and `block_cache` (from
  **comfyui-minimax-h3-blockcache-T8**). They were saved **active**, so a clean
  install hit `missing_node_type: Node 'attention patch (gated)' has no
  class_type` and nothing ran - even though the shipped recipe has all three
  gates OFF and never touches them.
  They now ship **bypassed** (purple). Bypassed nodes are dropped from the
  prompt entirely and the model passes straight through, so the workflow queues
  on an install that has neither pack. To use one: install its pack, select the
  node, `Ctrl+B` to un-bypass, **then** turn its gate switch on. Both steps.

- **`Value 4 bigger than max of 3: memory_frames` on a workflow you never
  edited.** v1.2 inserted `seed_per_shot` into the middle of the sampler's
  input list, ahead of `memory_frames`. ComfyUI stores widget values as a
  **positional array**, so every dial after the insertion point shifted by one
  in any workflow saved on v1.0/v1.1: your old `anchor_frames` was being read
  as `memory_frames`, your old `memory_frames` as `seed_per_shot`, and so on
  down the node. The error named a dial you never set.
  Two fixes, one backward and one forward:
  1. **Repair.** Pre-v1.2 workflows are detected on load (index 8 holds an int
     where a boolean belongs) and the missing value is spliced back in, so
     every dial lands where it belongs. Save the workflow to make it stick.
  2. **Never again.** The sampler now also stores its values **by name** in the
     node's properties and re-applies them by name on load, so no future change
     to the input list can shift anything. (Editing workflow JSON by hand? The
     values live in two places now - patch `h3_widget_values` as well as
     `widgets_values`.)
  If a bad value still reaches the queue - custom frontend extensions disabled,
  say - the validation error now explains the shift instead of naming the dial.

---

## New in v2.1.3


> **Correction, 2026-08-12 — read this before turning any drift dial on.**
> A render with `chain_gain_control=flatten`, `color_level=mvgd` and the
> per-shot audio leveller all ON came out **+142% texture and +18% brighter**
> over three shots, with a visible brightness step at each join. The per-shot
> approach cannot work, for a reason the code already knew: under
> `context_pin` the drift is carried by the **raw latent pin**, and every one
> of those dials operates on decoded frames *after* the pin has been stored.
> They correct what you see and not what feeds forward. `audio_tone_control`
> has been **removed**. `color_level=mvgd` is **deprecated** — its own source
> comment records a 29% warmth step at every join. Use `color_level=scene`
> (one target for the whole piece, applied per frame at the end) and the new
> `master_normalize=luma`, both of which run outside the feedback loop and
> land every frame on the same number, so they cannot create a seam.
> Texture drift is **not** fixable after the fact: the only lever is blur, and
> blur removes real detail along with the invented kind.

- **Picture darkening over chains: measured, and the existing dial verified.**
  User-reported (−1.5 luma/shot, monotonic). Same autoregressive mechanism as
  the audio dulling; the raw-latent pin carries it directly. `color_level=mvgd`
  - shipped since 2.1, never verified - holds an 8-shot chain to −1.0 total
  luma where uncorrected loses −10.5 (both seeds). On long chains turn on all
  three drift dials: 
- **Audio dulling over long chains: measured, mechanism found, countered.**
  Five-arm A/B on 8-shot chains: with `bank_pinned=0` (pure recency
  conditioning) the voice band collapses - 84-92% of 4-10 kHz energy gone by
  shot 8; with the default pinned slot, 8-50% depending on seed. It is the audio twin of the
  seam sharpening ratchet, running the other way, and continuity mode is
  irrelevant - the bank decides. Two counters ship: a console warning when
  `bank_pinned=0` on a chain past 4 shots (there is no true "bank off" - 0/0
  leaves one recency slot, the worst configuration), and
  **`audio_tone_control=flatten`** - the audio twin of
  `chain_gain_control=flatten`, EQ-matching every shot's long-term spectral
  envelope to shot 1's before the weld. Constant per-shot gains, clamped
  +/-9 dB, half-strength in the top band so it cannot manufacture hiss.
  Paired A/B on the worst seed: HF loss halved (-49.5% -> -23.7%), rolloff
  drift cut to a third. It reduces the drift rather than eliminating it (the
  context_pin replay carries raw latents the EQ cannot reach), and it ships
  OFF until ears, not spectra, have judged it.
- **The Audio Spine produced static with real-world audio files (ref2va,
  user-reported).** The spine encoded `guide_audio` at whatever sample rate the
  file arrived in, while the audio VAE expects its own rate (32 kHz) - the
  native node resamples, the spine path did not. Nearly every real voice or
  music file is 44.1/48 kHz, so the encoded latent was garbage, and because
  the spine LOCKS the audio stream to that latent at every sampling step, the
  render came out as noise. The same file worked through the native
  `MiniMaxH3ReferenceToVideo` node, which is exactly what the reporter
  observed. The spine now resamples to the VAE's rate and upmixes mono to
  stereo, and the console says so. Measured: guide-to-output correlation went
  from **0.06** (unrelated noise) to **0.97** on a 48 kHz voice track -
  identical to a native-rate control. Also verified at 44.1 kHz mono.
- **The spine's tooltip claimed `latent_handoff` only - wrong.** It works with
  every continuity mode; the per-shot stride table has carried each mode's
  seam trim all along, and the fix above was render-verified on `context_pin`.
  This is the locked-audio music-video path, now documented as such.
- **`two_pass_upscale` is removed.** It spatially interpolated the raw latent
  between passes. H3's latent is not a spatially smooth representation, so the
  interpolated values landed off-manifold and pass 2, running at low sigma, had
  no room to pull them back. Every arm tested came back as colour-noise mush
  against clean single-pass controls - including at 14 steps / `beta57`, the
  recipe `SETTINGS.md` previously called render-verified, and including shot 1,
  which carries no pin at all. It was never a `context_pin` incompatibility;
  it did not work in any mode. The guard around it is gone with it.
- **`output_scale`** replaces it: a lanczos resize of each shot's finished
  frames, after decode, so it cannot leave the latent manifold and works with
  every continuity mode. It adds resolution, not detail - measured at
  **1.78x faster** than rendering the same output size natively (45.5s vs
  80.9s at 672x384) and visibly softer. Applied **per shot**, so a long chain
  never holds a full upscaled master in memory at once.
- **`upscale_model`**: optional `UPSCALE_MODEL` input for real detail synthesis
  (ESRGAN and friends via ComfyUI's own loader), per shot, at the model's own
  factor. Render-verified with RealESRGAN x2plus: 448x256 -> 896x512, and
  combined with `output_scale` it lands exactly on the requested size.
- **`video_latents` / `audio_latents` / `head_frames` outputs** on the memory
  sampler (issue #12). Every shot's latent exactly as sampled, batched along
  dim 0, untrimmed. Shots after the first open with `head_frames` of replayed
  material that is only removed at decode, so they do not line up with the
  master until you trim it - the outputs are deliberately raw rather than
  trimmed on your behalf, because the pin material cannot be recovered later.
  Verified: 124-frame shots return 37 latent rows (`5*((F-5)//17)+2`), and
  124 + 124 - 22 is exactly the 226-frame master.

Both upscales are applied after the memory bank has taken its base-resolution
reference clip, so conditioning and VRAM are unchanged from an un-upscaled run,
and the returned latents stay base-resolution.

**If you saved your own copy of a v2.1.2 graph**, reload the shipped workflow:
removing four widgets shifts the saved widget order on that node.

---

## Fixed in v2.1.2

All four of these are in the writer half of the pack
(`ComfyUI_JoyAI_Echo_GGUF_Nodes`), so they only matter if you let the LLM write
the shots. If you paste your own prompt list, nothing here changes for you.

- **Every shot can now be saved as it renders.** A chain only became a file at
  the very end, so anything that failed after the last shot destroyed the whole
  run — one report was three hours lost to an OOM at the mux, *after* every shot
  had rendered successfully. `save_every_shot` (both samplers) writes each shot
  to `output/video/H3_SHOTS/` the moment it decodes. Written before the seam
  trim, so consecutive files overlap ~1s and the master is still the clean join.
  Requested in issue #13.
- **Custom sigma schedules.** The samplers built the schedule themselves from
  `steps` + `scheduler` with no way to supply your own, so a turbo LoRA that
  ships the curve it needs simply ran wrong rather than refusing. Both samplers
  now take an optional `SIGMAS` input; connect one and it replaces the schedule,
  `steps` rebinds to `len(sigmas)-1` so the two-pass split rides your curve, and
  the console says the widgets are being ignored instead of silently overriding
  you. It is a link-only input, so saved graphs are unaffected. Issue #14.
- **`---` separators were ignored in passthrough mode.** `example_script.txt`
  ships `---` separated and every doc tells you to write scripts that way, but
  the writer's passthrough path returned the whole file as ONE shot — which the
  sampler then repeated to fill `shot_count`. Pasting a finished multi-shot
  script rendered the entire text as shot 1, four times. It now splits on the
  same rule the sampler uses. A single paragraph is still one shot, so `.txt`
  batches are unaffected.
- **Reference images had no way in.** The sampler's `reference_images`
  input has always existed, and `SETTINGS.md` documented it — as `unwired`,
  because nothing in the workflow was connected to it. There is now a
  **REFERENCE** lane in the anchors column (two image loaders → `ImageBatch` →
  a gate), shipped with the gate **off** so nothing changes until you turn it
  on. This is the one item here that is a new capability rather than a repair.
- **A stale prompt-set filename blocked the whole queue.** ComfyUI validates
  every combo value in a graph before it will run anything, so if
  `RiftPromptSource`'s saved `source_file` no longer existed — a renamed
  folder, a workflow shared from another machine, or simply the prompt lane
  switched to manual — the run died with `Value not in list` and *nothing*
  executed, including the lanes that were fine. The node now declares
  `VALIDATE_INPUTS`, so the filename is only resolved if the node actually
  runs; switching to manual genuinely disables it. If it does run and the file
  is missing, the error names the file.
- **Every story came out 15 shots.** The system prompt ordered exactly 15 when
  the brief didn't ask for a count, so the LLM never got to decide. It now
  counts the story's beats and lands where the story lands — measured 4–7 on
  ordinary briefs, ~7 when the brief gives no signal at all (7 chained shots is
  about 65 s at 243 frames, just over the 1-minute mark that platforms pay on).
  Ask for a count and you still get exactly that count.
- **`short_story` mode did nothing.** The workflow's `system_prompt` widget held
  a frozen copy of the long-story prompt, and a filled widget overrides the
  per-mode file — so every mode ran the long prompt, and pack prompt updates
  never reached anyone who loaded the shipped workflow. The widget now ships
  empty and the dropdown works. **If you saved your own copy of the v2.0/v2.1
  workflow, clear that widget by hand.**
- **`short_story` is 1–3 shots** instead of always exactly 1 — one is still the
  common case, but a second or third is allowed when there's a named reason for
  it.
- **A messy LLM answer no longer kills the render.** Three separate real-world
  failures, all fixed: a markdown fence sharing a line with the JSON used to
  destroy the payload; a reply truncated at the token ceiling now has its
  complete shots salvaged and the partial tail dropped; and parsing sat *outside*
  the retry loop, so one malformed answer ended the run even though a re-ask
  usually fixes it. Order is now clean → parse → retry ×3 → salvage → fail, and
  the final error points you at `passthrough` mode.

---

## Fixed in v2.1.1

- **`context_pin` + Motion-Context coexistence.** Both packs patched
  `MiniMaxH3.extra_conds` and Motion-Context refuses to stack on an unknown
  wrapper, so with both installed `context_pin` errored out. This pack's
  wrapper is now a superset of theirs and declares their compatibility marker
  (`_h3_motion_context_payload_patch`), so whichever loads first owns the site
  and the other stands down. Load order no longer matters.
- **`seamless_tail` fails fast.** It needs interior keyframe anchors, which
  conflict with Motion-Context; it used to crash *mid-chain* after shot 1 had
  already rendered. It now stops before any sampling with the alternatives
  named (`context_pin`, `first_frame`, or remove that pack).
- **`seamless` is labeled the legacy soft pin it is.** Latent-only, no vision
  tokens, often still reads as a cut. Use `context_pin` or `first_frame` for a
  real join.
- The dev-machine blind spot that hid the first two (an install-layout bug in
  the conflict detection) is fixed, and releases are now tested on a packaged
  clean install.

## What is new in v2.0

- **A complete single-purpose workflow.** `H3_Seamless_Chain_v2.json` - 42 nodes in 9 grouped lanes with 8 on-canvas notes - built for one job instead of exposing every knob in the pack. `H3_Seamless_Chain_CORE.json` is the same graph with every third-party node removed.
- **MASTER CONTROLS panel** (`H3StudioControls`). One node drives resolution, frames per shot and steps for the sampler *and* for the prompt writer's dialogue pacing, so the writer's line lengths stay inside the shot length you actually set.
- **VRAM/SPEED panel** (`H3StudioSwitches` plus a reserve control). Three lazily gated model patches: memory-efficient attention, chunked feed-forward, block cache. **All OFF by default reproduces the verified recipe exactly**, and the gates are lazy, so an OFF patch never executes.
- **Energy-aware seam audio ("smart weld").** The boundary audio cut now lands in the quietest gap inside the incoming shot's first 0.75s rather than blindly at sample 0. A word placed at a shot head is no longer clipped.
- **Rewritten activation-reserve heuristic.** Cache keys now include a conditioning-*payload* signature (keyframes, audio references, two-pass), so a bare shot 1 and a reference-laden shot 2 get their own memory measurements instead of sharing one. Measured pools are no longer overridden by a fixed floor, a first run of a new payload variant estimates from a measured sibling, and **a VRAM spill into system RAM is now detected and named in the console** - previously it presented only as an unexplained ~5x slowdown.
- **`join_style` on the prompt writer.** Appends the render-verified boundary rules to the system prompt so generated scripts obey them automatically.
- **`flf_chain` fails loudly.** Selecting it with no boundary plates raises a clear error instead of silently rendering an unanchored chain.

---

## Requirements

### Required

| Component | Version | Why |
| --- | --- | --- |
| ComfyUI | **v0.30.0+** | Native MiniMax-H3 support. Older builds do not have the model. |
| This pack | v2.0 | Samplers, loaders, studio controls, gates. |
| MiniMax-H3 checkpoint (`ref2va` shipped, `fl2va` also works) | - | The generator. See [Models](#models). |
| H3 text encoder + video VAE + audio VAE | - | See [Models](#models). |

The **CORE** workflow needs nothing beyond the above. It is built entirely from this pack plus ComfyUI built-ins (`LoadImage`, `LoadAudio`, `SaveVideo`, `SaveAudio`, `CreateVideo`, `VAELoader`, `PrimitiveFloat`, `Note`).

### Full workflow

These serve the **FULL** workflow. A missing one does not degrade a single
feature — ComfyUI refuses to queue a graph containing any unknown node class,
so the workflow will not run until the pack is installed **or its nodes are
removed** (each has a documented removal, see `INSTALL.md`).

| Pack | Author | Unlocks | Needed when |
| --- | --- | --- | --- |
| [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | NikoDemon80 | `continuity=context_pin`, the raw-latent join (the shipped default) | Remove by setting `continuity=first_frame`. |
| [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) | ClownsharkBatwing | the `beta57` scheduler | Remove by setting `scheduler=beta` (CORE ships `beta` already). |
| ComfyUI_JoyAI_Echo_GGUF_Nodes (`JoyEcho_LLMEnhance`) | RealRebelAI (modified copy in the release zip) | The automatic prompt-writing lane | Use the zip's copy. Hand-written scripts can delete the writer instead. |
| [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) | pythongosssss | `ShowText` script preview on canvas | You want to read the generated script in the graph. |
| ComfyUI-sol-attn | sol-attn | Memory-efficient attention, chunked feed-forward | Only if you enable those two VRAM/SPEED switches. |
| comfyui-minimax-h3-blockcache-T8 | T8 | Block cache | Only if you enable that VRAM/SPEED switch. |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | city96 | GGUF checkpoints and encoders | Only if you run quantised models. |

**GGUF users:** install ComfyUI-GGUF, then run the arch patch once:

```bash
cd ComfyUI/custom_nodes/ComfyUI-H3-Multishot
python apply_gguf_arch_patch.py
```

This teaches ComfyUI-GGUF the `minimax_h3` architecture. See [Troubleshooting](#troubleshooting) if you still get an architecture error.

---

## Models

**Diffusion checkpoint (required).** MiniMax-H3. The workflows ship on **`ref2va`**; **`fl2va`** chains equally well and is lighter — see [Which checkpoint](#which-checkpoint). Either way it is the checkpoint's *trained continuation task* that makes the joins work.

### Which checkpoint

Both H3 variants chain, and they differ in what else they can carry.

- **`ref2va` — the shipped default.** Carries *reference rows*, which is the
  mechanism behind voice anchoring (`voice_ref`, `self_anchor_voice`) and the
  identity bank. Reference tokens ride through every sampling step, so it is
  somewhat slower and wants a little more headroom.
- **`fl2va`.** The first/last-frame variant. Lighter and faster, no reference
  rows — so voice anchoring and the bank do nothing on it, and turning them on
  only costs tokens. It chains just as well; the voice rides the frame relay
  and the join's audio reference instead of being explicitly pinned.

**Blind review passed on the `fl2va` configuration.** `ref2va` ships as the
default because it makes voice and character identity explicit rather than
emergent. To take the reviewed path: set the checkpoint to `fl2va` and turn
`self_anchor_voice` off.



GGUF quants: <https://huggingface.co/joeygambino/MiniMax-H3-GGUF>

| Quant | Card size it targets |
| --- | --- |
| `Q8_0` | 32GB |
| `Q5_1` | 24-32GB |
| `Q4_0` | 16GB |

The `curve` variants on that repo are pruned-form requants of the same weights.

**Text encoder and VAEs (required).** <https://huggingface.co/Comfy-Org/MiniMax-H3>

If you use a **GGUF text encoder** it needs its `-mmproj-*.gguf` vision sidecar — that path is what carries frames between shots. Load it with this pack's **H3 CLIP Loader (safetensors + GGUF)**, not the stock `CLIPLoaderGGUF`. Encoder quants: <https://huggingface.co/joeygambino/MiniMax-H3-encoder-GGUF>

ComfyUI-GGUF pairs the sidecar by **filename**, scanning only the encoder's own folder for a `.gguf` containing both `mmproj` and the encoder's stem. Rename either file, or split them across folders, and the match fails — upstream then logs an error and continues *without* the vision tower, which presents as the model ignoring your reference image. This pack's loader never does that:

- it **raises** rather than continuing blind;
- if the name match fails but exactly one mmproj sits beside the encoder, it uses that one and says so;
- and the `mmproj_name` widget lets you point straight at any sidecar, after which **names and folders stop mattering**.

**ref2va (optional).** Only needed for the reference/bank workflows. Seamless chaining does not use it.

---
- Curve-form GGUFs (smaller at the same quant, ComfyUI 0.30+): [HF](https://huggingface.co/joeygambino/MiniMax-H3-curve-GGUF) | [Civitai](https://civitai.com/models/2835678/minimax-h3-curve-form-gguf-fl2va-ref2va-low-vram)

## How the chaining works

H3 renders a block. Two blocks placed end to end normally read as a cut: the second block re-imagines the scene from text, so faces, wardrobe, framing and colour all shift at the boundary. Both mechanisms below attack that by carrying real rendered state across the join instead of re-describing it.

### 1. `first_frame` - the frame relay

Classic `H3MultishotSampler`. No third-party dependency.

Each shot's **last frame** is handed to the next shot as its **first frame**, through fl2va's *trained continuation task* - the same conditioning path the checkpoint was trained on for image-to-video, not a bolted-on hack. Because the next shot is generated *from* that picture rather than from a fresh reading of the prompt, the pixels at the boundary are continuous by construction: same face, same wardrobe, same lighting, same colour balance.

Two cleanups make the join invisible:

- The **duplicated boundary frame** is trimmed. Shot N's last frame and shot N+1's first frame are the same picture; leaving both in produces a one-frame stutter.
- The **seam audio** gets a **40ms equal-power weld**. Equal-power (rather than linear) crossfade keeps perceived loudness constant through the overlap, so the join does not dip.

### 2. `context_pin` - the raw-latent pin

`H3MultishotMemorySampler` with `continuity=context_pin`. Requires ComfyUI-H3-Motion-Context.

Instead of one frame, the previous shot's **last 22 frames** ride into the next shot **as raw latents** - bit-identical, with no VAE round trip - placed at *interior keyframe coordinates*, alongside a timeline-placed audio reference.

Why this is stronger than the frame relay:

- **No VAE round trip.** A decoded-and-re-encoded frame is not the frame the model produced; the codec loss at exactly the boundary is where colour and micro-texture drift enters. Passing latents keeps the handoff bit-identical.
- **22 frames of motion, not one still.** A single frame tells the model where things *are*. Twenty-two frames tell it where things are *going* - velocity, gesture direction, head turn, camera drift - so motion carries through the join instead of restarting from rest.
- **Interior coordinates.** The pinned latents sit inside the new shot's timeline rather than only at frame 0, so the model regenerates *through* the shared window and matches it, rather than departing from it.

The regenerated **0.92s head** (22 frames at 24fps) overlaps material you already have, so it is trimmed on decode. This is why the first second of every chained shot is discarded replay - see the boundary rules.

### Which one to use

`context_pin` is the shipped default and the tighter join. `first_frame` is the zero-dependency path and is verified across multi-shot chains in its own right. Both are real seamless mechanisms; the choice is dependency tolerance versus join tightness.

---

## Identity without reference images

Most chaining approaches hold a character by feeding reference images into every shot. This pack does not need them, and that is the point: **a 40-second two-character scene was rendered with zero reference images supplied**, and both characters held.

Two stacked mechanisms do it:

1. **The frame relay pins the instance.** Every shot after the first begins from an *actual rendered picture* of the character. The face and wardrobe propagate as pixels, not as a text description the model re-interprets. There is nothing to re-imagine, because the starting state is already correct.
2. **Byte-identical text pins the category.** The prompt writer repeats each character's appearance block **verbatim** in every shot - same words, same order, no paraphrase. Sampling noise around a slightly reworded description is exactly how a face drifts between shots; identical tokens remove that degree of freedom.

Frame pins the instance, text pins the category. Neither is sufficient alone: the relay without stable text lets the model gradually re-interpret the person over many hops, and stable text without the relay just gives you a well-described stranger every shot.

This is also why `seed_per_shot` should stay ON - see the settings table.

---

## Settings reference

### MASTER CONTROLS (`H3StudioControls`)

| Setting | Shipped | Notes |
| --- | --- | --- |
| Resolution | `1280x736` | Drives the sampler and is reported to the prompt writer. |
| Frames per shot | `362` | ~10.1s at 24fps. Sits on H3's frame grid. Also sets the writer's dialogue budget. |
| Steps | `14` | Part of the verified recipe. |

### Sampler dials

| Dial | Shipped | What it does |
| --- | --- | --- |
| `shot_count` | `0` | The TOTAL number of shots, **not** shots per prompt. `0` = one shot per `---` block in the script. A number forces that total: extra blocks are dropped, a short script repeats its last block. |
| `seed_per_shot` | `ON` | Derives a distinct seed per shot. **Leave this ON.** Measured: per-shot seeds *hold* the face; reusing one seed for every shot drifted both face and voice. |
| `continuity` | `context_pin` | `first_frame` (no deps), `context_pin` (needs Motion-Context), `flf_chain` (hard boundary-plate mode; requires plates, and raises an error without them). `seamless`/`seamless_tail` are legacy comparison modes: `seamless` is a soft latent-only pin that often reads as a cut; `seamless_tail` conflicts with Motion-Context and stops up front when that pack is installed. |
| `chain_gain_control` | - | Set to `flatten` on chains past about 5 shots. Seam texture ratchets roughly 1.3x per join - each shot sharpens the one after it - and `flatten` stops the compounding. |
| `color_level` | `off` | `off` / `mvgd` / `scene`. Levels each shot's colour and exposure statistics to shot 1's *settled tail* - a fixed reference, because matching each shot to its neighbour re-accumulates drift. Not needed when chaining by latents; reach for it if a long chain drifts warm or cool. |
| `self_anchor_voice` | - | Feeds shot 1's own rendered audio forward as the voice reference for later shots, so the voice identity established in shot 1 is what later shots match. |
| `voice_ref` | - | An external audio clip used as the voice reference instead. |
| `output_scale` | `1.0` (off) | Lanczos resize of each shot's finished frames, AFTER decode - resolution, not detail. Works with every continuity mode including `context_pin`. Applied per shot and after the bank takes its clip, so conditioning and VRAM are unchanged and a long chain never holds a full upscaled master in memory. Measured 1.78x faster than rendering the same output size natively, and visibly softer. |
| `upscale_model` | unwired | Optional `UPSCALE_MODEL` link (ComfyUI's Load Upscale Model - ESRGAN and friends). This one *synthesizes* detail rather than resizing. Per shot, at the model's own factor; combine with `output_scale` to land on an exact size. Its invented texture never reaches the memory bank, so it cannot feed the sharpening ratchet. |
| `reference_image_size` | - | `match` (use the render resolution) or `max`. |
| `preview_first_shot` | `ON` | Surfaces shot 1 as soon as it finishes so you can check framing and voice before the rest of the chain renders. |

### VRAM/SPEED switches (`H3StudioSwitches`)

| Switch | Shipped | Requires |
| --- | --- | --- |
| `sol_attn` (memory-efficient attention) | `OFF` | ComfyUI-sol-attn, un-bypass its node |
| `chunk_ffn` (chunked feed-forward) | `OFF` | ComfyUI-sol-attn, un-bypass its node |
| `remote_encoder` (encode prompts on a second PC) | `OFF` | the H3 Remote Text Encoder node filled in |

All three OFF reproduces the verified recipe exactly. The gates are lazy - an
OFF path never *executes*. Speed boosters (Spectrum, TeaCache, block cache,
EasyCache) live on the **H3 Speed Boosters** node, not here. The panel keeps
its 2.5.x **eight output slots** in the original order (`two_pass`, `sol_attn`,
`chunk_ffn`, `spectrum`, `block_cache`, `dual_clock`, `hybrid_cond`,
`remote_encoder`) so saved graphs keep working; the five removed slots always
emit `False`. There is also an activation-reserve control here for overriding
ComfyUI's inference-memory estimate.

### Shipped workflow defaults

```
H3_Extend_Take (main): 1280x736 landscape | take_seconds 30, window auto | 14 steps | euler + beta57
H3_Seamless_Chain_v2:  736x1280 portrait  | 192 frames/shot x 4 shots | 14 steps | euler + beta57 (RES4LYF) / beta (CORE)
continuity = context_pin      |  ref2va checkpoint   |  bank ON   |  chain_gain_control = flatten
all VRAM switches OFF         |  all speed boosters OFF  |  preview_first_shot ON
24fps mux -> output/video/H3CHAIN/  (+ paired audio file)
```

---

## Prompting and boundary rules

These are render-verified. The FULL workflow's prompt writer applies them automatically via `join_style`; **hand-written scripts must follow them manually.** Ignoring them is the most common cause of a chain that looks seamless but sounds wrong.

- **AIRLOCK.** Every shot after the first **opens holding the previous shot's exact closing arrangement**, with about two quiet seconds before anyone speaks. Quiet means real micro-motion - a breath, a weight shift - not a freeze.
- **The first ~1 second of every chained shot is discarded replay.** Dialogue placed at frame 0 loses its opening syllables. This is not a bug to work around; it is the overlap window that makes the join seamless.
- **LAND SETTLED.** End each shot back in a stable arrangement, dialogue finished, with about two seconds spare.
- **A spoken line never straddles two shots.** Budget: *dialogue + 4 seconds of quiet must fit the shot length.* At 243 frames one long line fits. At 124 frames it does not.
- **Repeat verbatim.** Each character's appearance description **and** the room/lighting description are repeated word-for-word in every shot. See [Identity without reference images](#identity-without-reference-images).
- **Keep fps at 24.** Other frame rates audibly shift voice accents.

Script format: one prompt per shot, `---` between shots. JSON (`{"prompts": [...]}`) is also accepted.

---

## Files in the release zip

```
ComfyUI-H3-Multishot/                          this pack
  LICENSE  README.md  __init__.py              defensive loader
  apply_gguf_arch_patch.py                     on-disk GGUF arch fallback
  h3_advanced.py                               advanced sampling helpers
  h3_avbank_probe.py                           AV bank diagnostics
  h3_cartridge.py                              portable character cartridges
  h3_episode_tools.py                          StudioControls, StudioSwitches, AnySwitch
  h3_gguf_arch.py                              teaches ComfyUI-GGUF the minimax_h3 arch
  h3_interior_patch.py                         interior anchors (stands down for Motion Context)
  h3_keyframes.py                              keyframe anchor nodes
  h3_lora_stack.py                             H3LoraStack
  h3_multishot_utils.py                        samplers, loaders, gates
  h3_ref_folder.py                             reference-folder picker
  rift_prompt_source.py                        RiftPromptSource (prompt sets)
  rift_script_picker.py                        RiftScriptPicker (JSON scripts)
  rift_writer_unload.py                        adds unload_model_after to the writer
ComfyUI_JoyAI_Echo_GGUF_Nodes/                 the LLM prompt writer, modified (see its NOTICE)
INSTALL.md  PROMPTING.md  SETTINGS.md
example_script.txt                             worked four-shot two-hander
workflows/H3_Seamless_Chain_v2.json            everything, optional lanes gated off
workflows/H3_Seamless_Chain_CORE.json          same job, zero third-party packs
workflows/H3_Keyframes.json                    single-clip keyframe anchoring
```

---

## Troubleshooting

### A word is clipped at a join

Two causes, in order of likelihood.

1. **The script put dialogue at the head of a shot.** The first ~1s of every chained shot is discarded replay, so an opening syllable there is trimmed away with it. Fix it in the script: apply the AIRLOCK rule and give the shot about two quiet seconds before anyone speaks.
2. **The seam cut landed on speech.** v2.0's smart weld searches the incoming shot's first 0.75s for the quietest gap and cuts there instead of at sample 0. If you are on an older release, or the shot head has no quiet gap at all to find, the weld has nothing to work with - the fix is still the script.

Also check that no single spoken line straddles two shots, and that `dialogue + 4s` fits your `frames_per_shot`.

### Each shot is sharper than the last

This is the seam sharpening ratchet: texture compounds roughly **1.3x per join**, so it is invisible at 3 shots and obvious at 8. Set:

```
chain_gain_control = flatten
```

Recommended on any chain past about 5 shots.

### The render stalls at 0 steps, or runs ~5x slower than it should

Almost always a **VRAM spill into system RAM**: the DiT loads only partially and streams the remainder from RAM on every step. v2.0 detects this and names it in the ComfyUI console - check there first, because it used to present as an unexplained slowdown with no message.

Remedies, in order:

1. Lower resolution or `frames_per_shot`, or use a smaller quant (`Q5_1` for 24-32GB, `Q4_0` for 16GB).
2. Set the activation-reserve override on the VRAM/SPEED panel. ComfyUI's inference-memory estimate is very conservative at large frame counts, and reserving a measured value instead reclaims the difference for resident weights.
3. Enable the VRAM/SPEED switches (their packs are part of the full workflow's requirements). These change the numerics slightly, so they are OFF by default - turn them on only after you have a baseline you trust.

### Red or missing nodes when the workflow loads

You loaded `H3_Seamless_Chain_v2.json` (the FULL graph) without one of its required packs. Either:

- install the missing pack from the [Optional](#optional) table - the node's title tells you which one - or
- load `H3_Seamless_Chain_CORE.json` instead, which has no third-party nodes at all.

If **every** node from this pack is red, the pack itself did not load: confirm it is in `ComfyUI/custom_nodes/`, confirm ComfyUI is **v0.30.0+**, and read the console for an import error on startup.

### `Unexpected architecture type in GGUF file: 'minimax_h3'`

ComfyUI-GGUF validates a GGUF's architecture against a fixed list and rejects the file before reading any tensors; upstream's list has no `minimax_h3` entry. The quant is fine. This pack teaches it that architecture. If you see the error anyway, the patch is not active - apply it on disk and restart:

```bash
cd ComfyUI/custom_nodes/ComfyUI-H3-Multishot
python apply_gguf_arch_patch.py
```

If a **GGUF text encoder** fails with a state_dict or vision mismatch instead, you are loading it with the stock `CLIPLoaderGGUF`. Use this pack's **H3 CLIP Loader (safetensors + GGUF)**. If it reports no vision sidecar resolved, set its `mmproj_name` widget to the sidecar directly — that bypasses filename pairing entirely.

### Audio gets duller the longer the chain runs

Real and expected: each hop costs a little audio brightness, and it accumulates. There is no dial for it. The working practice is to **restart the chain on scene cuts** - render a long scene as several chains and cut between them, rather than as one chain of many hops. Very long chains are explicitly in the not-yet-verified list below.

---

## Verified / not yet verified

Stated honestly, because the difference matters when you are budgeting GPU hours.

**Verified**

- A 3-shot `context_pin` chain and multi-shot `first_frame` chains were reviewed **blind** by two independent video-understanding models. One described the result as one continuous unedited take, colour consistent, with nothing broken.
- Verified on **both** static talking-head content and dynamic moving content.
- Identity held across a **40-second two-character scene with zero reference images** supplied.
- The blind-reviewed recipe: **fl2va checkpoint, euler sampler, beta57 scheduler, 14 steps, 362 frames per shot (~15.1s at 24fps)**, all VRAM/SPEED switches OFF, no voice anchor.
- The **shipped** defaults differ deliberately: `ref2va` with `self_anchor_voice` and the bank on, for explicit voice and character identity. That combination has **not** been through blind review — it is the same chaining mechanism with reference rows added.
- `pass1_fraction = 0.4` for the two-pass upscale.
- `seed_per_shot` ON holds the face; one seed shared across shots drifted face and voice.

**Not yet verified**

- **Very long chains.** Audio dulls slightly per hop. Restart chains on scene cuts.
- **The ref2va + bank + `context_pin` combination.** Untested together.
- **Hard-FFLF boundary-plate mode** (`flf_chain`). It now fails loudly without plates rather than rendering an unanchored chain, but the mode itself has not been validated.

If you get a result outside this envelope, good or bad, an issue with the settings block is genuinely useful.

---

## Credits

- **MiniMax** - the H3 model.
- **Comfy-Org / ComfyUI** - native H3 support and the text encoder + VAE distribution.
- **NikoDemon80** - [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), the raw-latent motion-context mechanism `context_pin` is built on.
- **city96** - [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).
- **pythongosssss** - [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) (`ShowText` preview).
- **JoyAI Echo** - ComfyUI_JoyAI_Echo_GGUF_Nodes, the `JoyEcho_LLMEnhance` prompt writer.
- **ComfyUI-sol-attn** - memory-efficient attention and chunked feed-forward.
- **comfyui-minimax-h3-blockcache-T8** - block cache.
- **@viralesveras** - keyframe position parsing and `images_batch`, contributed upstream in this repo.

Pack and workflows by **jlucasmcrell** (GitHub) / **joeygambino** (Hugging Face, Civitai).

## Support

Everything here is free and stays free. If it saved you a night of debugging:

- [Ko-fi](https://ko-fi.com/joeygambino)
- [GitHub Sponsors](https://github.com/sponsors/jlucasmcrell)
- [Liberapay](https://liberapay.com/joeygambino) (recurring)

## License

See `ComfyUI-H3-Multishot/LICENSE` in the release.
