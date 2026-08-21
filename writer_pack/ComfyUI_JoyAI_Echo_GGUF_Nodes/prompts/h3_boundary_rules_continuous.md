CONTINUOUS-SCENE BLOCK BOUNDARIES (chained rendering - these rules are
render-verified; violating them reproduces mid-word chops and pose jumps)

The blocks you write are rendered as a chain: the previous block's last
~1 second is replayed at the head of the next block and DISCARDED.
Consequences:

1. THE AIRLOCK. Every block after the first OPENS holding the previous
   block's exact closing arrangement - same characters, same positions,
   same framing - with NO dialogue for the first ~2 seconds. Give the
   hold real micro-motion (a breath, a weight shift, an eyeline change)
   so it does not read as a freeze. Only then does anyone speak.

2. LAND SETTLED - AND LAND ON THE FACE. Every block ENDS with ~2 seconds
   of quiet settle, back in a stable arrangement, all dialogue finished.
   The arrangement you hand over is the arrangement the next block must
   open holding - and those closing frames are ALSO the identity the next
   block inherits, so every recurring character's face is visible in
   frame at the close (three-quarter or profile is fine; turned away,
   exited, or covered is not). A block that ends on the back of a head
   hands the next block a stranger.

3. LINES NEVER CROSS A BLOCK BOUNDARY. <scenetrans> is for cuts INSIDE a
   block only. A supplied line must fit entirely inside one block with
   the 2-second head and tail intact. Budget: spoken dialogue at natural
   pace plus 4 seconds of hold/settle must fit the block length. If it
   does not fit, move the whole line to the next block - never split it.

4. NO CONTRADICTIONS AT THE BOUNDARY. The pinned frames are not a
   suggestion: a block that opens describing a different arrangement gets
   the union of both (extra people, doubled props). Change the scene
   MID-block, after the airlock, never at the boundary.

5. IDENTICAL SCENE BLOCK. The style/location/lighting description - not
   just the character blocks - is restated byte-identically in every
   block. An unnamed light source gets reinvented per block, and that is
   where colour drift is born.

6. PER-CALL AIRLOCK SUBTRACTION. The system prompt you receive contains
   a per-call BUDGET BLOCK built by the renderer. For every shot after
   the first (SHOTS_AFTER_FIRST > 0), the BUDGET BLOCK states:
       airlock_seconds          = 2.0
       line_must_start_by_sec   = 2.0
       line_must_finish_by_sec  = (num_frames - 14) / fps - 1.5
   Use those numbers literally. The spoken line begins at second
   line_must_start_by_sec and ends no later than line_must_finish_by_sec.
   For the first shot in a chain (SHOTS_AFTER_FIRST = 0), the line may
   begin at second 1.0 (a one-second settle, no airlock). A line that
   starts before line_must_start_by_sec on a chained shot renders as
   clipped speech and the airlock's silence becomes garbled filler.
   The system prompt's WORDS_MIN and WORDS_MAX already account for the
   airlock; you do not need to subtract anything from them by hand.

7. A LOCKED ARRANGEMENT MEANS THE STORY COMES TO THE FRAME. These rules
   hold framing and arrangement steady across boundaries - so events must
   happen INSIDE that framing: the creature flies INTO frame and lands on
   the desk, the dogs burst INTO the shot, the window that breaks is the
   one visible behind the character. Never park the character in frame
   describing events that are happening outside it. If the premise's
   events cannot physically fit inside one framing, the story belongs in
   the CUTS join mode, where every shot may frame its own event - write
   the shots faithfully anyway and keep each event inside the frame.
