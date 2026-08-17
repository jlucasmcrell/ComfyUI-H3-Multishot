## REFERENCE PHOTOGRAPHS ARE ATTACHED - identity comes from the pictures, not from your prose

Reference photographs of every recurring character are attached to the render and bound to that character. The video model copies face, skin, hair, age and build from those photographs. Prose that describes those same things COMPETES with the photographs, and the prose wins - render-verified on the same seed: a written identity sentence ("a woman in her thirties with dark hair tied back") produced a different person from the reference; the same prompt with that sentence replaced by a pointer to the photographs produced the referenced person exactly.

While this block is present, these rules override anything above about identity sentences and IDs:

0. IDs are the characters' OWN NAMES from the premise (write `Dana`, `Theo`), never `ID_A` / `ID_B`. The reference photographs are filed under those names and are matched by scanning your prose for them; an anonymised ID casts nobody. Use the name exactly as the premise spells it, in every shot, byte-identical.
1. The base identity sentence names the character and points at the photographs, and says nothing else: `Dana looks exactly as in the reference photographs - same face, hair, age and clothing.` No age, ethnicity, hair colour or style, face shape, eye colour, skin, build, eyewear, jewellery or make-up anywhere in the prompt, in any shot.
2. Clothing: the photographs' clothing is the default and is covered by the sentence above. Do not invent clothing. Only if the PREMISE itself names what they wear, replace "and clothing" with what they wear, e.g. `Dana looks exactly as in the reference photographs - same face, hair and age - and wears a grey raincoat.` Keep that byte-identical across shots.
3. Voice sentence, expression sentence, action, camera, background and sound are unchanged and still required exactly as the rules above say. Expression and gaze are still allowed and still per-shot.
4. If two characters both have photographs, each gets its own pointer sentence; never describe how they differ from each other in looks.
5. Everything else about the output format is unchanged: the same JSON shape, the same byte-identical repeated blocks, the same word budgets.
