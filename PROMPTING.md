# Writing for seamless chains

The workflow makes the *mechanics* of a join invisible. Whether the join
*reads* as invisible is decided by the script. These rules are
render-verified: every one of them was written after a specific failure.

---

## The script format

One prompt per shot, with `---` on its own line between shots:

```
Shot 1 prompt...
---
Shot 2 prompt...
---
Shot 3 prompt...
```

`{"prompts": ["...", "..."]}` is also accepted, which is what the LLM
writer emits.

`shot_count` is the TOTAL number of shots, not a per-prompt multiplier. `shot_count = 0` renders one shot per `---` block - that is the setting you want when you have written the shots yourself. Setting it to `1..8` forces
that many shots — extra prompts are dropped, and if there are too few, the
last prompt continues.

---

## The five rules

### 1. The airlock

Every shot after the first **opens holding the previous shot's exact
closing arrangement** — same people, same positions, same framing — and
stays quiet for about two seconds before anyone speaks.

> *"She sits exactly as she was, both hands flat on the bench, lips closed.
> For the first couple of seconds she only breathes, shifting her weight
> slightly, and only then she says: ..."*

Why: the head of a chained shot is a regeneration of the previous shot's
tail. It is discarded on decode. Anything you put there is thrown away.

### 2. Give the hold something to do

A held framing with *nothing happening* renders as a literal freeze, and two
seconds of a motionless actor looks like the video stalled. Write in a
breath, a weight shift, an eyeline change. The camera holds still; the
performer does not.

### 3. Land settled

Every shot **ends** back in a stable arrangement with the dialogue finished
and about two seconds to spare. The arrangement you hand over is the
arrangement the next shot must open holding.

### 4. A line never straddles two shots

Budget it: *spoken dialogue at natural pace + 4 seconds of quiet* must fit
inside the shot length. At 24 fps:

- **362 frames** (~15.1 s) is the trained maximum and the shipped default.
  About 11 s of that is speakable once the head and tail are reserved —
  roughly 27 words.
- **243 frames** (~10.1 s) fits one shorter line: about 6 s speakable,
  roughly 15 words.
- **124 frames** (~5.2 s) does **not** work — the model drops the airlock
  to cram the line in, and the join audibly clips.

The shipped `example_script.txt` is written to the 362-frame budget: four
shots, 24–29 words of dialogue each.

If a line does not fit, move the *whole* line to the next shot. Never split
it.

### 5. Repeat the descriptions word-for-word

Restate each character's full appearance description **and** the
room/lighting description **verbatim** in every shot. Not paraphrased —
identical.

This is half of how identity holds (see below). Rewording a character's
description between shots is the single most common cause of a face
changing mid-scene.

---


### 6. Change something physical in every shot

Rule 5 is about **appearance and the room** - a character's description and the
lighting, restated identically. It is *not* an instruction to restate the
action, and reading it that way is the most common cause of a chain that stops
progressing: shot 3 comes back as a near-copy of shot 2.

The reason is that the model is separately instructed to preserve the subject,
the room and the colour temperature. When shot 3's text is nearly identical to
shot 2's, "keep everything the same" is the only clear signal in the prompt, and
that is what you get.

So each shot's action has to leave the world in a state the previous shot's
world was **not** in. Physical and irreversible, not a mood or a camera move:

* weak - *"Marcus looks at the clipboard and frowns."* Could be any shot.
* strong - *"Marcus tears the top sheet off, crumples it and drops it; the bare
  second page is now uppermost."* Shot 4 cannot be mistaken for this one.

The test: if you can swap two shots' action lines and the script still reads
correctly, the model cannot tell them apart either.

Two things actively work against you here, both because CFG is 1.0 and there is
no negative branch to subtract anything: **negations get rendered** ("the shot
does not repeat" puts *repeat* in the conditioning), and **stillness phrases
freeze the whole frame** ("goes still", "does not move"). Say what changes, not
what does not.

## How identity holds with no reference images

Two mechanisms, stacked, and neither works alone:

**The frame relay** carries the instance. Every shot begins from an actual
rendered picture of the character — the previous shot's last frame — so the
specific face, the wardrobe weathering, the exact hair all propagate as
pixels rather than being re-imagined from a text description. Drift can only
accumulate *per join*, not per frame.

**The verbatim text** carries the category. It re-asserts everything the
frame cannot pin down, and it authors anyone who walks into frame partway
through the scene. Once they are in a handed-over frame, they join the relay
too.

A 40-second two-character scene held both faces this way with **zero
reference images supplied**.

For longer chains, characters who leave frame for a long stretch, or when
you need a *specific* pre-existing face, add a reference image — the
sampler's `start_image` (identity anchor) or `reference_images` input.

---

## Contradictions render as unions

Chained shots are conditioned on the previous shot's picture. If shot 3
opens describing a *different* arrangement of people than shot 2 closed on,
the model does not choose — you get both. Extra people, doubled props.

Change the scene **mid-shot**, after the airlock. Never at the boundary.

---

## Camera cuts

Camera cuts *inside* a shot are free — H3 renders them natively and they
share one continuous audio bed. Write them as part of the prompt.

Continuity lives *between* shots. So: put cuts inside shots, put continuity
between them.

---

## Frame rate

Keep the mux at **24 fps**. Other rates audibly shift voice accents — this
is a property of the model's audio lane, not a muxing artifact.
