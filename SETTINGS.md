# Settings reference

Everything below documents what a dial does, what the shipped default is,
and what breaks if you move it.

The shipped values are a working production configuration, not a
theoretical best. Three of them are deliberately non-default:
`audio_lock` off, `join_anchor_noise` 0.002 and `join_blend` on.

Shipped defaults favour explicit identity: `ref2va` with voice self-anchoring
and the bank on. The configuration that went through blind review was `fl2va`,
which carries no reference rows, and switching to it is two changes (see
*Checkpoint choice*). Both chain seamlessly, and both files are the same size -
the difference is reference rows, not weight.

---

## The shipped recipe

```
checkpoint   MiniMax-H3 ref2va (GGUF Q8_0 / Q5_1 / Q4_0)
sampler      euler
scheduler    beta57     (full workflow; needs RES4LYF)
             beta       (CORE; stock ComfyUI)
steps        14
frames/shot  362        (~15.1 s at 24 fps, the trained maximum)
resolution   1280 x 736 (landscape) or 768 x 1344 (vertical)
fps          24
```

**On the scheduler.** `beta57` comes from RES4LYF, not stock ComfyUI. Measured
on an identical seed, it scored 10/10 for lip-sync against 8/10 for stock
`beta` ("slight synthetic stiff quality to the mouth"), with image quality,
skin texture, artifacts and audio judged equal. The full workflow ships
`beta57` and lists RES4LYF as required; CORE ships `beta` so it keeps its
zero-third-party-pack promise. Switching is one widget either way.

Both orientations sit under the model's pixel ceiling. Resolution **cannot
change mid-chain** — every shot in one run must be the same size.

---

### Checkpoint choice

`ref2va` (shipped) carries **reference rows** - the mechanism behind
`voice_ref`, `self_anchor_voice` and the identity bank. Those three do
nothing on `fl2va`, which has no reference rows; on `fl2va` they only cost
tokens on every sampling step. `fl2va` still chains well - the voice is
carried by the frame relay rather than pinned - and it is the configuration
that went through blind review.

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

## Automatic safeguards (no dials)

Two protections run on their own; you only ever see their log lines.

* **Leftover-VRAM sweep.** If a previous run died without cleaning up, the
  next render used to plan its memory against stale numbers and crawl at
  high utilisation but low wattage. The planner now unloads leftovers and
  re-measures before accepting a tight fit. Log line:
  `cleared N GB of leftovers before reserve planning`.
* **Master-assembly watchdog.** The final video assembly streams frames
  between two ffmpeg processes; if no bytes move for 3 minutes the pair is
  killed and the run fails loudly with the per-shot temp files kept, instead
  of freezing the queue.

## Drift on long chains, and the dials that fight it

Chained shots accrete detail. Each one conditions on the previous shot's own
output, so invented texture compounds. Measured over **ten shots at 960x544**
with all three of the following on:

| control | shipped | what it does |
| --- | --- | --- |
| `memory_frames` | `0` | The big one. The bank's RECENT slots feed accreted output forward on top of the pin. At `0` the only reference is shot 1, which nothing has been added to yet. Luma drift 1.055 -> 1.022 per hop, chroma 1.086 -> 1.039, and the drift stops *accelerating* (4.2->6.7%/hop becomes 2.3->2.0%). Identity and framing held; framing actually improved. |
| `master_normalize` | `luma+contrast` | Levels brightness **and** contrast of the finished chain to one target taken from shot 1. Matching only the mean masks a contrast ratchet while it compounds underneath. |
| `pin_renorm` | `on` | Holds each pinned latent at shot 1's sigma. The pin's own sigma climbs every hop, and that inflated pin is what the next shot is handed. |

Residual is about **1.02 per hop**. Not zero - long chains still drift, slowly.

`chain_gain_control` is the other half: texture ratchets about 1.3x per join, so
set it to `flatten` on chains longer than about five shots.

`pin_noise` is small and scene-dependent (-1% to -2% per hop) and gets worse
above `0.10`. It is not the fix it was described as in 2.1.5.

### Dials that do nothing under `context_pin`

Verified in source rather than assumed. All three ship at `0` so nobody spends
time tuning a dead control:

* `join_anchor_noise` - noises KEYFRAME latents, and `context_pin` creates no
  join keyframes.
* `handoff_release` - belongs to `continuity=latent_handoff`.
* `bank_ref_noise` - reaches bank reference *images*, which measurement shows
  are not the carrier.

### Two-pass upscale

Needs the H3 latent-upscaler pack, and is **not usable** with
`continuity = context_pin` or `latent_handoff`, or with an audio spine: those
carry raw latents or one locked denoise trajectory across the join, and a
two-pass render cannot preserve either. The node stops with an error naming the
clash rather than quietly weakening the join. Available on `cut`, `seamless`,
`seamless_tail`, `first_frame` and `flf_chain`.

* `two_pass_upscale` `OFF` - each shot renders low-res through `pass1_fraction`
  of the steps, upscales in latent space, finishes at full res.
* `upscale_factor` `1.5` - pass-1 size is size/factor, snapped to /32.
* `pass1_fraction` `0.4` - verified clean. Past ~0.5 pass 2 cannot erase the
  upscale pattern and a ghost/moire lattice appears.
* `upscale_audio_denoise` `0.35` - how much pass 2 may rewrite audio. `0` locks
  pass-1 audio (safest for voice), `1` is a full remix.

---

## MASTER CONTROLS (`H3StudioControls`)

One panel drives `width`, `height`, `frames_per_shot` and `steps` on the
sampler, and — in the full workflow — the prompt writer's dialogue pacing,
so the LLM sizes lines to the real shot length.

The panel also emits `sampler_name` and `scheduler` as strings. Those cannot
link to the sampler's combo widgets on current ComfyUI frontends, so they go
to the sampler's `sampler_override` / `scheduler_override` inputs instead:
when connected they win, and when nothing is connected the sampler's own
widgets apply.

`shot_count` on the panel drives the sampler *and* the prompt writer's
`num_shots`, so the two can never disagree. `0` means one shot per `---` block in
the script, and lets the writer decide how many to write.

`use_file_prompts` selects where the scene comes from: **off** reads the
manual scene-idea box, **on** reads the prompt set (file or folder). The
switch is lazy, so the branch you are not using never executes.

---

## Sampler dials

### Chaining

| Dial | Default | What it does |
|---|---|---|
| `shot_count` | `0` | The TOTAL number of shots, **not** shots per prompt. `0` = one shot per `---` block in the script - leave it there for a written scene. `1..8` forces the total: extra blocks are dropped, a short script repeats its last block. |
| `continuity` | `context_pin` (full) / n/a (CORE) | `context_pin` pins the previous shot's last 22 frames as **raw latents** — needs the Motion Context pack. `first_frame` uses the model's own trained hand-off — no extra pack. `cut` for episodic work. `seamless` and `seamless_tail` are **legacy** modes kept for comparison: `seamless` is a latent-only soft pin and often still reads as a cut; `seamless_tail` needs interior keyframe anchors and **conflicts with the Motion-Context pack** - with it installed the run stops up front with the alternatives named. |
| `chain_gain_control` | `off` | Set to `flatten` for chains past about 5 shots. Each shot's tail anchors the next and the model returns ~1.3× the anchor's texture energy, so sharpness **ratchets** across a long chain with a visible step at every seam. `flatten` levels every shot to one house texture. |
| `color_level` | `off` | Levels each shot's colour statistics to shot 1's settled tail. Not needed when chaining by latents — colour already carries. Useful if you see a warm/cool drift across a long chain. |

### Identity and voice

| Dial | Default | What it does |
|---|---|---|
| `seed_per_shot` | `ON` | **Leave it on.** Measured: varying the seed per shot *holds* the face; using one seed for every shot drifted both face and voice. Identity lives in the conditioning, not the seed. |
| `start_image` | unwired | An identity anchor image. Seeds shot 1 and anchors appearance. Optional — the frame relay plus verbatim descriptions usually suffice. |
| `reference_images` | **off** | A batch of character portraits carried into **every** shot as `<Picture 1>`, `<Picture 2>`… Bind them in the prompt text. Needs a ref2va checkpoint. Fed by the REFERENCE IMAGES group two ways: **USE AUTO REFS** on (photos found from the script, nothing else to flip), or the two `LoadImage` slots with the **MANUAL REFS gate** on; chain another `ImageBatch` for a third and fourth manual slot. Unlike `start_image` these are not a first frame — they do not constrain shot 1's composition, they only carry who the person is, and they are what covers shot 1 while the memory bank is still empty. |
| `voice_ref` | unwired | A clean solo speech clip, carried into every shot as `<Audio 1>`, pinning the voice. |
| `self_anchor_voice` | **`on`** (needs `ref2va`) | Shot 1's *own rendered voice* becomes the reference for every later shot — no file needed. Write shot 1 with a clean solo line. Needs a ref2va checkpoint; a wired `voice_ref` takes priority. Note it enlarges the activation pool on every shot after the first. |

### Upscaling

Upscaling happens **after decode**, per shot. That is deliberate: the previous
`two_pass_upscale` interpolated the raw latent between passes, and H3's latent is
not a spatially smooth representation - interpolated values land off-manifold and
the second pass, running at low sigma, cannot pull them back. It produced colour
noise on every configuration tested, including the one this file used to call
render-verified. It is gone.

| Dial | Default | What it does |
|---|---|---|
| `output_scale` | `1.0` (off) | Lanczos resize of each shot's finished frames. Adds resolution, **not** detail. Works with every continuity mode including `context_pin`, because it is downstream of the VAE. Measured: rendering at 448x256 with `output_scale 1.5` reached 672x384 in 45.5s against 80.9s rendering 672x384 natively - **1.78x faster, and visibly softer**; concrete pore texture that survives a native render washes out. Use it when the clock matters, or when the whole-chain upscale would not fit in memory; render native when texture matters. |
| `upscale_model` | unwired | Optional `UPSCALE_MODEL` link (ComfyUI's Load Upscale Model - ESRGAN and friends) to synthesise detail rather than resize. Applied per shot at the model's own factor; combine with `output_scale` to land on an exact size. Render-verified with RealESRGAN x2plus (448x256 -> 896x512, and combined with `output_scale` landing exactly on 672x384). |
| `master_normalize` | `off` | Deflicker the FINISHED chain: every frame driven to ONE global luma target, after the master exists — outside the feedback loop, so it cannot create a seam. Unit-verified: a 3-shot chain drifting +22% luma with +0.052 steps at the joins came out flat (+0.0%) with steps of +0.00001. Brightness only; texture drift is not fixable after the fact. |
| `upscale_model_name` | `(none)` | Pick an upscale model by name instead of wiring a loader — same behaviour as the `upscale_model` input, which wins if both are set. Reads `models/upscale_models/`. |

Both are applied **per shot**, after the memory bank has taken its
base-resolution reference clip. That keeps the conditioning payload and its VRAM
identical to an un-upscaled run, and means a long chain never holds a full
upscaled master in memory at once - which is the failure that cost one user a
three-hour render (issue #13).

### Workflow

| Dial | Default | What it does |
|---|---|---|
| `preview_first_shot` | `off` | Writes shot 1 to `output/video/H3_FIRSTSHOT/` the moment it decodes — minutes before the chain finishes — so a bad take can be cancelled early. The full path is printed to the console. |
| `save_every_shot` | `off` | Writes **every** shot to `output/video/H3_SHOTS/` as it decodes, alongside the master. Insurance for long chains: anything that fails after the last shot — a mux OOM, a full disk, a closed tab — otherwise destroys the entire render at once. Files are written *before* the seam trim, so consecutive shots overlap by about a second; the master is still the clean join. |
| `sigmas` | unwired | Optional custom sigma schedule (a `SIGMAS` link, no widget). Replaces sampler/scheduler + steps entirely — some turbo LoRAs ship a schedule they need in order to work at all. When connected, `steps` becomes `len(sigmas)-1`, the two-pass split is taken as a fraction of *your* curve, and the console says the steps/scheduler widgets are being ignored. |
| `reference_image_size` | `match` | `max` uses 2048 px references for best identity fidelity, but reference tokens ride through every sampling step, so it can be several times slower. |
| `seed` | randomize | Fix it to make a good take reproducible. |

**Audio dulls over long chains — measured, and the bank is the counter.** Five-arm
A/B (2026-08-11, 8-shot chains): with `bank_pinned=0` the conditioning is pure
recency — each shot hears only the one before it — and the voice band collapses
(84–92% of 4–10 kHz energy gone by shot 8). With the default pinned slot the
drift is 8–50%, depending on seed. Continuity mode is irrelevant to this; the
bank decides. There is no true "bank off": `memory_frames=0, bank_pinned=0`
still leaves one recency slot, which is the *worst* configuration, and the node
warns about it on chains past 4 shots. Keep `bank_pinned` at 1. The per-shot
audio leveller that once sat here has been removed — it corrected the decoded
audio while the raw-latent pin carried the drift forward untouched, which is
the same flaw that made the picture dials fail.


**The activation reserve never outbids the weights (2026-08-12).** Every GB the
reserve claims comes out of the weight budget, and a DiT that misses a *full*
load streams the remainder over PCIe on every step. Measured at 960x544: shot 1
reserved 7.8 GB and loaded completely at 18.8 s/it; shot 2's larger conditioning
payload asked for 9.4 GB, left the DiT 399 MB short, and ran at **283 s/it** - a
15x collapse bought by headroom the measurement did not need. The reserve is now
capped at `free - weights - 384 MB`, so the weights always load completely and
the pool takes what is left. Same three-shot chain: 36m54s -> 14m03s, every shot
`full load: True`, no OOM, and the clamped pool then measured a 7.3 GB peak
against the 8.0 GB it was given. If a clamped run ever does OOM, lower the
resolution or the reference count - do not raise the reserve.

**Drift, and what actually works on it (corrected 2026-08-12).** Every chained
lane drifts — texture up, luminance and audio spectrum down — because the model
regenerates from its own output. But **per-shot correction does not fix it**: a
render with every per-shot dial ON measured **+142% texture and +18% luma** over
three shots with a step at each join. Under `context_pin` the drift rides the
**raw latent pin**, which is stored before decode, and every per-shot dial
operates on decoded frames — they fix the picture you see, not the thing that
feeds forward. What works is correction *outside* the loop, on the finished
master, driven to ONE global target: `color_level=scene` for colour and
`master_normalize=luma` for brightness. Neither can create a seam, because
every frame lands on the same number. Texture is the one lane with no honest
after-the-fact fix — blur is the only lever and it destroys real detail.

**Best-audio recipe (cut-grammar content):** `continuity=cut` +
`bank_pinned=1, memory_frames=0` - a bank of exactly one slot, shot 1,
forever. Every shot is then a fresh generation whose only audio reference is
the original performance, and the measured drift is flat (−5% over 8 shots at
both seeds tested, where the shipped seamless config lost 8–50%). The joins
are cuts, not a continuous take - that is the trade. Identity and scene still
hold; the pinned slot carries them.


---

## Prompt writer (`JoyEcho_LLMEnhance`)

| Dial | Default | What it does |
|---|---|---|
| `unload_model_after` | `off` | Frees this writer's model from Ollama the moment the script is written, so the video model gets the card. Uses this node's own `base_url` and `model_name`. **Turn it on whenever the writer is local.** |

A local writer and H3 want the same GPU, and ComfyUI's eviction cannot help —
it frees models inside the ComfyUI process, while Ollama is a separate process
with its own allocator. Ollama's OpenAI-compatible endpoint cannot be asked
either: its request type has no `keep_alive` field and the shim never sets one,
so the parameter is silently dropped and the model sits for the server default
of five minutes — the whole of shot 1. This switch calls Ollama's native
endpoint, which honours it.

The switch is added to the writer **at runtime by this pack**, so that pack is
not modified and the switch is simply absent when it is not installed. It is
off by default. `JoyEcho_LLMEnhance` is RealRebelAI's node, from
ComfyUI_JoyAI_Echo_GGUF_Nodes.

---

## VRAM / SPEED panel (full workflow only)

**Start with every switch off and the reserve at 0, and try a render before
touching anything here.** That is both the verified recipe and, in practice,
the fastest route to a working chain: the activation reserve measures each
shape and conditioning payload as it renders and sizes the pool itself. It has
held on 32 GB and 24 GB cards alike. These switches are for digging out of a
spill the console has already reported, not for pre-emptive tuning.

**What "held on 24 GB" costs, measured 2026-08-17 on a 3090 at the shipped
736x1280 x 192 frames:** shot 1 (no references yet) keeps ~8 GB of the 14.5 GB
model resident and runs ~67 s/step; every later shot carries the memory bank as
references, its activation pool measures ~15 GB, only ~4 GB of weights stay
resident and it runs ~98 s/step - about 23 minutes per shot, streaming the rest
from RAM each step. It works, it is just slow. On a 24 GB card the levers, in
order: 640x1152 (keeps most weights resident, roughly 3x faster), 141-frame
windows, then `sol_attn` / `chunk_ffn`.

The gates are lazy — an off patch never executes, so leaving them alone costs
nothing.

- **`sol_attn`** — memory-efficient attention. The biggest VRAM saving at
  high resolution or long shots. Small quality risk; A/B one render before
  trusting it on a keeper.
- **`chunk_ffn`** — chunks the feed-forward pass. Moderate VRAM saving,
  small slowdown, no known quality cost.
- **`block_cache`** — skips near-duplicate transformer blocks. This buys
  **speed, not VRAM**, and can exaggerate high-frequency texture. Keep it
  off for final masters.
- Other toggles on the panel are not wired in these workflows.

**VRAM RESERVE** sets activation headroom on the model loader. **Leave it at
`0`.** A hand-set number *overrides* the measurement, so a value that suited
one shape becomes wrong for the next — and too little headroom makes a
high-resolution render **stall silently at 0 steps** rather than erroring,
because the driver pages to system RAM instead. Set a value only to recover
from a spill the console has named.

The reserve heuristic measures each shape *and conditioning payload*
separately (a bare shot 1 and a reference-laden shot 2 need different
pools), and prints a named diagnosis if a run does spill:

```
[H3AutoReserve] SLOWDOWN: 299s/step vs 59s/step earlier this session (5.1x).
  This is the VRAM-spill signature: the driver is paging to system RAM
  instead of erroring. Fix: raise the activation reserve, drop
  resolution/frames, or remove reference payload.
```

---

## Seam audio

Automatic, no dials. The boundary cut lands in the **quietest gap** within
the incoming shot's first 0.75 s rather than blindly at sample zero, then a
40 ms equal-power weld joins the two shots. A word placed at a shot head
survives.

If you still hear a clipped word at a join, the script put dialogue too
close to a boundary — see `PROMPTING.md`.

---

## Keyframe anchors and Motion Context

The pack can place keyframe anchors at **arbitrary** frame positions, not
just first and last. Two implementations exist for that and they patch the
same place, so exactly one owns it at a time:

- **ComfyUI-H3-Motion-Context installed** → that pack owns it. Its version
  is a superset (per-row coordinates plus audio timeline placement), so this
  pack detects it at import and stands down with a line in the log.
- **Not installed** → this pack's own `h3_interior_patch` fills the gap.

Either way, stock first/last anchors always work and you do not have to
choose. If you see a log line about standing down, that is the healthy path.

---

## New in 2.2.4

### `reference_subjects` - more than one person in your references

Until 2.2.4 every reference picture was declared to the model as a photograph of
`<Subject 1>`. With one character that is correct. With several it told the model
that pictures of different people were all the same individual, and it rendered
the average - which is why multi-character reference sets came back resembling
nobody.

`reference_subjects` groups the pictures. Comma counts, in picture order:

| value | meaning |
| --- | --- |
| *(empty)* | all pictures are one person - the pre-2.2.4 behaviour, unchanged |
| `3,3` | pictures 1-3 are person A, 4-6 are person B |
| `2,2,2` | three people, two pictures each |

Only `<Subject 1>` is described as speaking, because H3's voice conditioning is
single-speaker. Counts that do not add up are corrected rather than rejected.

### `walk_folder` on the prompt source - queue a folder of scenes

`start_index` normally selects a block *inside* one file. Turn `walk_folder` on
and it selects **which file**, walking the chosen folder in sorted order and
emitting the whole file.

Wire `start_index` to a `PrimitiveInt` set to `increment` and queue N times to
render a folder of finished scripts unattended. This exists because an H3 script
uses `---` for **shot** boundaries, so scenes cannot be concatenated into one
file the way LPFF prompt batches can - a folder is the only way to batch them.

### `reference_video` - hand H3 an existing clip

New optional inputs on the sampler, fed by the `V2V REFERENCE` lane in the
workflow (`Load Video` -> `Get Video Components` -> `H3 Reference Video`). Ships
muted.

**It is scene and appearance conditioning, not motion control.** H3 is told a
video reference is "a clip from an earlier moment of this same continuous scene"
and asked to keep its framing, camera distance, room contents and colour
temperature. There is no pose, depth or optical-flow path in H3, so the subject
will not copy the movement in your clip.

Keep the window short. References are subsampled to 2 fps and then ride through
**every** sampling step, so a 25-second clip is roughly 50 reference frames of
permanent per-step cost. `H3ReferenceVideo` trims for you and prints what the
window will cost. Requires `ref2va`; fl2va has no reference rows and ignores it.

---

## Automatic reference casting (`H3AutoRefs`)

Optional. Feeds the sampler's `reference_images` from a folder of character
photos, choosing which ones by reading your script - so a multi-shot episode
casts itself instead of you re-picking `LoadImage` slots per scene.

**How it decides.** It scans the prompt for the *names of subfolders* under
`refs_root`, case-insensitively and on word boundaries. One subfolder per
character:

```
<refs_root>/
    DANA/     dana_wide_01.png  dana_wide_02.png  ...
    ROOK/     rook_wide_01.png ...
    MARCUS/   ...
```

A script that says "Dana pushes her glasses up" matches `DANA/` and loads that
folder's images. Matching is on the folder name only - the image filenames
inside can be anything.

**Dialogue is stripped before the scan.** `<d>...</d>` blocks, quoted speech and
`says, '...'` are removed first, so a character who is *talked about* but not
present never gets cast. This is deliberate and field-proven.

**What it loads.** First three matched characters, in first-mention order,
`max_per_character` images each (default 3) taken in sorted filename order,
capped at the model's 9 reference slots. It then prepends identity bindings to
the prompt - `<Picture 1>, <Picture 2> are the same person (Dana).` - and prints
what it picked:

```
[H3AutoRefs] 3 ref(s): DANA/dana_wide_01.png -> <Picture 1>; ...
```

Glance at that line to confirm the right cast loaded.

**Switching it on (2.6.0).** In the REFERENCE IMAGES group, flip **USE AUTO
REFS** on. That alone is enough - the node's `refs_batch` output (every picked
photo in one batch) goes straight to the sampler. Before 2.6.0 a second
**REFERENCE gate** sat downstream and silently discarded auto refs unless you
also switched it on; the log said `3 ref(s)` and the render had none. That gate
is now the **MANUAL REFS gate** and only guards the two `LoadImage` slots. The
line that proves refs reached the sampler is
`[H3Memory] N reference image(s) ride in every shot` - if you see the
`[H3AutoRefs]` line without it, nothing went in.

The nine per-slot outputs `ref_1` ... `ref_9` are still there for graphs that
route photos individually; `refs_batch` carries every picked photo (up to the
model's 9-slot cap), so two or three matched characters no longer overflow.

**AutoRefs runs before the writer (2.6.0).** It scans the premise (wire the scene idea into `prompt_text`), is gated by `enabled` (wired from USE AUTO REFS), and its `found` output drives the writer's `refs_attached` - so the writer only points at photographs that actually exist, and a miss is known before any writing happens.

**The writer must not describe the person (2.6.0).** Measured on the same
seed: with photographs attached, a written identity sentence ("a woman in her
thirties with dark hair tied back") rendered *that* person; the same prompt
with the sentence replaced by "looks exactly as in the reference photographs -
same face, hair, age and clothing" rendered the person in the photographs. The
writer node has a `refs_attached` BOOLEAN input; the shipped canvases wire it
from USE AUTO REFS, so switching auto refs on also switches the writer into
pointer mode (log line: `LLMEnhance refs_attached: identity sentences will
point at the reference photographs`). Rules live in
`prompts/h3_refs_attached_rules.md`. Hand-written scripts: do the same thing
yourself - name the ID and point at the photographs, describe wardrobe only if
it must differ from them.

| widget | default | what it does |
| --- | --- | --- |
| `refs_root` | *(empty)* | Folder holding one subfolder per character. Empty resolves to `input/h3_refs/`. Relative paths resolve under `input/`; absolute paths work too. |
| `max_per_character` | `3` | Images per matched character. Front / three-quarter / profile sets hold identity best. |
| `characters` | *(empty)* | Comma-separated folder list that **overrides the scan entirely**. Use it when the prose does not name someone. |
| `overrides` | *(empty)* | Folder remaps, e.g. `dana=dana_outdoor` to swap a character's set for particular scenes. Clear it afterwards. |
| `on_no_match` | `no_reference` | `no_reference` (default) warns in the console and renders without photos when nothing matches; the writer is told (`found` = false) and describes the person normally. `error` stops the run instead - and because AutoRefs now runs BEFORE the writer, it stops before the writer spends its minutes. |

**The gotcha worth knowing.** The scan reads the *prose*, so the character must
be named there. A scene called `09_rook_reaches_through_the_front_door.txt`
whose description only says "a seven-foot figure of liquid chrome" matches
nothing - the filename is not scanned. Around 18% of our own archive scenes are
written that way. Two fixes: name the character in the description, or type the
folder name into `characters`.

**Scenes with no cast** - a dashcam, an empty harbour - should either set
`on_no_match` to `no_reference` or not use the node at all.

**What references do and do not carry.** Measured: they restore appearance
outside the model's usual range - a character's solid-black eyes came back
correctly with references and did not without. They do **not** carry *scale*: a
seven-foot character still rendered near human height, because nothing in a
photograph of someone standing alone states their size. Scale needs an in-frame
comparison or explicit prose.

---

## Known limits

- Audio dulls slightly per hop on **long** chains. Restart the chain on a
  scene cut, where a fresh start costs nothing.
- Resolution is fixed for the duration of a chain.
- `flf_chain` (hard first/last-frame boundary plates) is implemented but
  needs a colour-matched plate set; without plates wired it now raises a
  clear error rather than silently rendering unanchored.


## Speed boosters (measured 2026-08-16, same seed, 640x1152x192x14)

Optional third-party accelerators, switchable on the H3 Speed Boosters node.
All change the output slightly (trajectory divergence, like a seed change),
none require specific hardware, VRAM cost is negligible.

- baseline 621s
- Spectrum 441s (-29%) - github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3
- TeaCache 0.15 531s (-14%), 0.30 491s (-21%) - github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache
- block cache 551s (-11%) - github.com/T8mars/comfyui-minimax-h3-blockcache-T8 - CORRECTED 2026-08-17: 0 cache hits on every run at 14 steps on two cards; inert at the shipped step count, the -11% did not reproduce. Ships OFF; only meaningful at 30+ steps.
- Spectrum + TeaCache stacked: 450s - no faster than Spectrum alone. Pick one.

Eye-test verdicts (operator, same-seed masters, 2026-08-16): blockcache
indistinguishable from baseline (because it did nothing - see above; ships OFF). Spectrum and TeaCache both showed
visible distortion on people (environments unaffected). The stack was severely
damaged. Ambient audio acceptable on every arm.

A booster whose pack is missing prints an install link and passes the model
through unchanged.


## Memory systems (new in 2.5)

### Driver headroom (automatic)

When weights plus working memory would fill the card past roughly 95%, the
Windows driver starts demoting GPU memory and render times become a lottery
(the same job measured anywhere from 27 minutes to 3 hours). The pack now
detects that zone during its auto-reserve pass and deliberately streams a few
GB of weights instead - streamed weights are nearly free, the last few
percent of VRAM are not. The lottery case measured 15 minutes flat after the
fix. Fully automatic; the console prints a "driver headroom" line whenever it
engages.

### `low_ram_master` (Video Combine, default off)

Off, every finished shot is held in system RAM until the final join - fine
for a few shots, but a five-shot high-resolution chain can want tens of GB at
the very end. On, each shot streams to lossless temporary files as it
finishes and the master is assembled from disk through the same levelling
math (verified at 42.8 dB against the in-RAM path - codec noise, nothing
more). Peak RAM stays near two shots regardless of chain length. Turn it on
if long chains have ever crashed your machine at the join step, or if you run
under 32 GB of system RAM. The finished file's path is available on the new
`master_path` output either way.

### Remote text encoder (`H3 Remote Text Encoder`, optional)

The text encoder works for a few seconds per shot and holds 15+ GB the whole
render. If you have any second PC with ComfyUI, install this pack on it, point
the node at `http://THAT-PC:8188`, and turn `remote_encoder` ON in the
VRAM / SPEED SWITCHES panel - prompts are encoded over there and your render
card keeps the memory. Results are
identical (verified across machines), and repeated text is answered from a
local cache with no network call at all. The flag ships OFF; off, the
node is inert and single-PC setups are unaffected.

### `H3 TAE Decode` (draft previews)

A 9 MB tiny decoder that turns latents into full-resolution draft frames in
about 2 seconds, versus roughly a minute per shot through the real VAE.
Drafts smear fine texture but composition, framing and motion read clearly.
Use it to audition seeds or triage a batch, then decode keepers through the
real VAE. Never use it for finals.

## Extend take (new in 2.6)

One prompt, one continuous speech, as long as you want.

### `take_seconds` / `window` / `model` (MASTER CONTROLS)

Set `take_seconds` to the length you want and leave `window` on `auto`. The
panel picks the largest window whose estimated activation pool fits with
most of the weights resident on your card - wire the loader's MODEL into
the panel's `model` socket for a real weight size (15 GB assumed otherwise)
- and derives the number of windows that fills the time. It prints its plan
in the console. `frames_per_shot` and `shot_count` on the panel are
overridden while `take_seconds` is set; 0 turns it off. Pick a number for
`window` to override auto (bigger = fewer joins, smaller = less VRAM; 141
is the comfortable 24 GB window, 243 the 32 GB one).

### Writer join style: `extend take`

The writer receives the whole take's length and word budget, writes ONE
continuous speech, and cuts it across the windows ONLY at sentence or clause
boundaries - no airlock, no settle, no silence at the joins. It steers to the
upper half of each window's word budget because dead air is simply window
seconds minus speech seconds. Rules live in `prompts/h3_extend_rules.md`.

### `audio_pin_frames` (memory sampler)

Frames of the previous window's AUDIO to pin as reference, independent of
the picture pin. 0 = same as `pin_frames`. Longer audio context costs
conditioning rows but no delivered frames. 96 (4 s) reproduces the JoyEcho
audio memory window; measured neutral at n=1 on a same-seed A/B. Ships 0.

**Known limit (2.6.0):** the chain's texture ratchet is not fully solved for long takes - measured about +13% fine texture per join at 736x1280 with the anti-drift set on. Under ~4 windows (~30-40 s) it is slight; at 7 windows it is visible sharpening. Keep extend takes to ~4 windows for now; a pin-side fix is in progress for 2.6.1.
