"""RiftCast Generator - design a character from dropdowns, render their
audition tape, and pack them into a .riftcast cartridge in one queue.

Two nodes:
  RiftCast_CharacterDesigner - dropdowns -> canonical DNA paragraph + an
      audition-shot script (neutral casting room, front-lit close-up,
      ~45 words of non-falsifiable dialogue with timbre + accent binding -
      the verified voice recipe). Feed its script output into LLMEnhance
      (passthrough) exactly like a PromptSource.
  RiftCast_Packer - takes the rendered frames + audio from Generate, cuts
      the voice anchor and reference stills, writes NAME.riftcast into
      input/riftcast/ and materializes it immediately. The character is
      castable in the very next render.

Re-queue with a new seed to re-audition; pack when you like them. Once
packed, the voice and face are locked for every future render.
"""
import json
import os
import re
import tempfile

try:
    import folder_paths
except Exception:
    folder_paths = None

from .riftcast import pack
from .joyecho_cartridge import _packs_dir

GENDER = {
    "woman": ("woman", "she", "her", "hers"),
    "man": ("man", "he", "his", "his"),
    "androgynous person": ("person with an androgynous presentation", "they", "their", "theirs"),
}
AGE = ["teens", "early twenties", "mid twenties", "late twenties",
       "early thirties", "thirties", "late thirties", "forties",
       "fifties", "sixties", "seventies", "eighties"]
SKIN = ["very pale", "pale", "fair freckled", "light olive", "light tan",
        "tan", "warm beige", "golden brown", "olive", "brown", "deep brown",
        "dark", "very dark", "ruddy and wind-weathered", "sun-weathered"]
ETHNICITY = ["(unspecified)",
             "of East Asian descent", "of Korean descent",
             "of Japanese descent", "of Chinese descent",
             "of Southeast Asian descent", "of Filipino descent",
             "of Vietnamese descent", "of South Asian descent",
             "of Indian descent", "of Middle Eastern descent",
             "of Persian descent", "of North African descent",
             "of West African descent", "of East African descent",
             "of Caribbean descent", "of Mexican descent",
             "of Puerto Rican descent", "of Colombian descent",
             "of Brazilian descent", "of Indigenous American descent",
             "of Pacific Islander descent", "of Mediterranean descent",
             "of Italian descent", "of Greek descent",
             "of Eastern European descent", "of Scandinavian descent",
             "of Irish descent", "of Scottish descent",
             "of mixed heritage"]
HAIR_COLOR = ["black", "blue-black", "dark brown", "chestnut brown",
              "light brown", "ash brown", "auburn", "copper red", "ginger",
              "strawberry blonde", "dirty blonde", "honey blonde",
              "golden blonde", "ash blonde", "platinum blonde", "bleached white",
              "silver", "salt-and-pepper", "steel gray", "white",
              "dyed jet black", "dyed pastel pink", "dyed deep red",
              "dyed teal", "dyed violet", "dyed green",
              "dyed with grown-out roots", "two-tone dyed"]
HAIR_STYLE = ["(from style)",
              "long and loose", "long and wavy", "long and poker-straight",
              "shoulder-length and straight", "shoulder-length and curly",
              "in loose beachy waves", "in a chin-length bob",
              "in a blunt bob with a fringe", "with curtain bangs",
              "in a high ponytail", "in a low ponytail", "in a messy bun",
              "in a tight bun", "in a single braid", "in two braids",
              "in a french braid", "in cornrows", "in box braids", "in locs",
              "in a short afro", "in tight natural coils",
              "cropped short", "in a pixie cut", "buzzed close",
              "shaved at the sides with length on top", "in an undercut",
              "short and tousled", "short back and sides", "slicked back",
              "parted at the side", "parted in the middle",
              "pinned half-up", "wet and pushed back", "under a cap",
              "under a wool cap", "wrapped in a headscarf", "thinning on top",
              "bald"]
HEIGHT = ["short", "petite", "average height", "tall", "very tall"]
BUILD = ["slim", "lean", "wiry", "average build", "athletic", "muscular",
         "broad-shouldered", "stocky", "sturdy", "soft", "heavyset",
         "tall and gangly", "petite and compact"]
VOICE = ["a clear voice, natural and unforced",
         "a low voice with a slight rasp",
         "a warm mid-toned voice",
         "a bright quick voice",
         "a soft-spoken voice",
         "a gravelly voice",
         "a deep steady voice",
         "a light airy voice",
         "a smoky low voice",
         "a crisp precise voice",
         "a breathy voice",
         "a booming voice that fills the room",
         "a thin reedy voice",
         "a hoarse voice",
         "a nasal voice",
         "a flat unhurried voice",
         "a quick clipped voice",
         "a dry voice that barely lifts"]
ACCENT = ["casual American", "casual American with flat Midwestern vowels",
          "soft Southern American", "Boston-flavored American",
          "New York-flavored American", "Californian American",
          "British", "Australian"]

MAKEUP = ["(from style)", "(none)",
          "bare skin with no makeup",
          "minimal natural makeup, just balm on the lips",
          "a bold red lip and nothing else",
          "smudged dark liner along the lash line",
          "heavy winged liner and dark brows",
          "a glossy nude lip and soft flush",
          "matte pale foundation with a dark brow",
          "freckles left visible under tinted balm",
          "stage-heavy makeup with hard contour",
          "smeared makeup that has been cried through",
          "day-old makeup wearing off at the edges"]

ACCESSORIES = ["(from style)", "(none)",
               "thin wire-rimmed glasses",
               "thick black-framed glasses",
               "reading glasses pushed up into the hair",
               "small gold hoop earrings",
               "silver studs in both ears",
               "a thin chain necklace sitting at the collarbone",
               "a leather cord necklace",
               "a wristwatch with a worn leather strap",
               "several plain silver rings",
               "a beanie pulled low",
               "a ball cap with a curved, sweat-marked brim",
               "a bandana knotted at the throat",
               "a scarf wound loose at the neck",
               "fingerless gloves worn through at the knuckle",
               "a canvas shoulder bag with a frayed strap",
               "a lanyard with a scuffed ID card",
               "headphones resting around the neck"]

DEMEANOR = ["(from style)",
            "calm and personable, looking straight into the lens",
            "still and watchful, moving very little",
            "restless, shifting and gesturing while talking",
            "warm and open, leaning slightly toward the lens",
            "guarded, face flat and hands out of frame",
            "amused, one corner of the mouth up",
            "tired and gentle, shoulders low",
            "sharp and quick, chin level and gaze direct",
            "easy and slouched, weight on one side",
            "formal and upright, hands folded"]

STYLES = {
    'ACADEMIC - STEM lab': {
        'wf': 'a white cotton lab coat with pen marks inside the chest pocket over a plain crew-neck tee, straight dark jeans, and flat closed-toe shoes with scuffed rubber soles',
        'wm': 'a white lab coat with one sleeve shoved to the elbow over a plain grey tee, chino trousers with a marker smudge on the thigh, and worn leather-topped sneakers',
        'hair': '{pos} hair pulled flat against the head and fastened at the nape so nothing hangs near the face',
        'makeup': '',
        'acc': 'clear safety glasses pushed up onto the crown, a lanyard holding a scratched plastic ID card, and a blue nitrile glove half peeled off one hand',
        'dem': 'stands square with the elbows tucked in close, hands making short precise adjustments, gaze locked on the hands before flicking up to the camera',
    },
    'ACADEMIC - bookish': {
        'wf': 'an oversized rust cardigan with sleeves stretched out of shape over a plain cotton tee, corduroy trousers gone bald at the knee, and thick socks inside soft-soled sneakers',
        'wm': 'a faded flannel shirt hanging open over a washed-thin henley, loose corduroy trousers with a frayed hem dragging at the heel, and canvas sneakers creased grey across the toe',
        'hair': '{pos} hair gathered back off the face in a quick knot with pieces slipping loose at the temples, otherwise unstyled',
        'makeup': '',
        'acc': 'smudged glasses with one arm bent out of line, a hardback held open against the chest with a finger marking the page, and a folded receipt used as a bookmark',
        'dem': 'sits with one leg folded under, shoulders rounded forward, eyes dropping back down to the page in the gaps between answers',
    },
    'ACADEMIC - dark academia': {
        'wf': 'a wool herringbone blazer rubbed thin at the elbows over a cream oxford shirt buttoned to the throat, a pleated grey skirt, opaque dark tights, and leather loafers scuffed pale at the toe',
        'wm': 'a brown tweed jacket with fraying cuffs over a knitted vest and a creased oxford shirt, dark wool trousers gone shiny at the seat, and cracked leather brogues',
        'hair': '{pos} hair worn a little long and pushed back off the forehead with a few strands fallen forward',
        'makeup': 'a dark matte lip, bare unpowdered skin, and a faint smudge of brown pigment along the upper lash line',
        'acc': 'a leather satchel with a broken buckle strap, a thin signet ring worn smooth, and a paperback with a split spine held closed by a rubber band',
        'dem': 'leans forward with one elbow on the table, chin dropped, gaze holding the camera without blinking, one hand curled around a cup',
    },
    'ACADEMIC - light academia': {
        'wf': 'a cream cable-knit sweater pilled at the cuffs over a collared linen blouse, a beige pleated midi skirt with a soft crease at the front, and tan leather flats creased across the toe',
        'wm': 'an oatmeal knit vest over a linen shirt with the sleeves rolled and wrinkled above the elbow, pale khaki trousers, and tan derbies with the polish worn off the toecap',
        'hair': '{pos} hair parted loosely to one side and tucked behind the ears, ends slightly windblown',
        'makeup': 'sheer balm on the lips and a thin wash of warm colour high on the cheekbones',
        'acc': 'a canvas tote with ink staining the bottom seam, a fine chain at the collarbone, and round wire-rimmed glasses folded in one hand',
        'dem': 'sits upright with the shoulders dropped, chin tipped slightly down, hands resting open on the table, movements slow and unhurried',
    },
    'ACADEMIC - prep school scholar': {
        'wf': 'a navy blazer with a stitched breast patch over a white shirt and a striped tie pulled loose at the throat, a pleated tartan skirt, ribbed knee socks slipping down one leg, and penny loafers scuffed grey at the toe',
        'wm': 'a navy blazer with a stitched breast patch over a white shirt marked with ink on the cuff, a striped tie knotted loose and crooked, grey flannel trousers, and black loafers worn pale at the heel',
        'hair': '{pos} hair combed into a tidy shape that has come loose over one eye, flattened at the crown',
        'makeup': '',
        'acc': 'a canvas book bag with the strap darkened where it crosses the shoulder, a plastic wristwatch with a scratched face, and a chewed pen clipped to the shirt placket',
        'dem': 'sits with both feet flat and the back pressed to the chair, hands laid flat on the knees, glancing up quickly then away',
    },
    'ACADEMIC - professor': {
        'wf': 'a charcoal wool blazer with a softened shoulder over a silk-blend blouse, wide-leg trousers whose pressed crease has gone slack, and low block-heel shoes worn down at the back edge',
        'wm': 'a corduroy jacket rubbed flat and shiny at the elbows over a button-down shirt and a narrow knitted tie pulled loose from the collar, flat-front trousers, and dull brown lace-ups',
        'hair': '{pos} hair kept short at the sides and combed straight back, with a section escaping over one temple',
        'makeup': '',
        'acc': 'reading glasses pushed up onto the forehead, a shirt cuff powdered with chalk dust, and a paper-clipped stack of marked essays carried under one arm',
        'dem': 'stands with the weight settled onto one hip, one hand shaping small flat arcs while speaking, eyes drifting off to the middle distance between sentences',
    },
    'CLASSIC - coastal': {
        'wf': 'a loose white linen shirt wrinkled from wear over a faded cotton tank, wide cropped trousers with sand caught in the hem, and canvas espadrilles fraying at the sole',
        'wm': 'a washed-soft chambray shirt hanging open over a sun-bleached tee, rolled cotton shorts salt-stained at the pockets, and flat leather sandals worn thin at the footbed',
        'hair': '{pos} hair grown past the ears and dried stiff with salt, raked back with the fingers',
        'makeup': 'sunscreen left shiny across the nose and cheekbones, lips chapped and bare',
        'acc': 'a braided cord bracelet gone limp with wear and sunglasses with scratched lenses hooked at the collar',
        'dem': 'stands loose-shouldered with weight on one leg, squinting slightly, hands moving often, head tipped away from the light',
    },
    'CLASSIC - corporate executive': {
        'wf': 'a squared-shouldered navy wool blazer with a pressed lapel over a plain silk shell, straight-leg trousers with a sharp front crease, and black leather pumps rubbed pale at the toe',
        'wm': 'a charcoal wool two-button suit with a faint sheen at the seat, a stiff white cotton shirt, a dark tie knotted tight to the collar, and black oxfords polished over fine scratches',
        'hair': '{pos} hair pulled back off the face and held flat against the skull, ends cut level',
        'makeup': 'shine blotted from the forehead and nose, a thin neutral lip, brows combed and set',
        'acc': 'a steel watch with a scuffed bracelet and a leather folio gone shiny at the corners',
        'dem': 'sits square with forearms on the table, chin level, hands clasped and mostly still, blinking rarely',
    },
    'CLASSIC - country club': {
        'wf': 'a white cotton tennis dress with a pleated skirt and a faint grass stain at the hem, a pale cardigan buttoned only at the top, and white leather court shoes greyed at the sole',
        'wm': 'a pique polo shirt with a sun-faded collar, pressed white golf trousers with a woven fabric belt, and leather spiked shoes dusted with cut turf',
        'hair': '{pos} hair pushed off the face by a folded terry headband, cut short and blunt at the nape',
        'makeup': 'matte sunscreen over the nose and forehead and a tinted balm on the lips',
        'acc': 'a visor with a sweat-darkened band, a thin-strapped watch with a sun-faded face, and a small towel over one shoulder',
        'dem': 'stands with feet planted wide and one hand resting on a hip, chin lifted, small nods between long stretches of stillness',
    },
    'CLASSIC - minimalist': {
        'wf': 'an unlined oatmeal cotton shirt buttoned to the neck, wide-leg trousers in heavy grey twill, and flat leather slides softened at the strap',
        'wm': 'a heavyweight cotton tee with a slightly stretched neckline, straight black trousers cropped at the ankle, and canvas sneakers gone grey at the rubber',
        'hair': '{pos} hair cut in one blunt length with no layering, tucked behind the ears',
        'makeup': '',
        'acc': 'a plain steel watch with a brushed case marked by fine scratches',
        'dem': 'stands with weight even on both feet, arms hanging loose, head turning before the shoulders, movement slow',
    },
    'CLASSIC - old money': {
        'wf': 'a cream silk blouse gone soft from washing, a camel wool skirt falling below the knee, and low leather pumps with the soles worn smooth',
        'wm': 'a charcoal wool jacket with a faint shine at the elbows, a plain cotton shirt open at the collar, flannel trousers with a soft crease, and brogues polished over old scuffs',
        'hair': '{pos} hair combed back from the temples and kept flat and short of the shoulders',
        'makeup': 'matte skin with clear balm on the lips and brows brushed into shape',
        'acc': 'a small gold-cased watch on a cracked leather strap and a signet ring dulled at the edges',
        'dem': 'sits back with weight settled onto one hip, hands motionless in the lap, gaze steady and slow to move',
    },
    'CLASSIC - parisian chic': {
        'wf': 'a striped cotton boat-neck top with a softened neckline under a wool blazer cut wide at the shoulder, straight dark denim, and flat leather ballet shoes creased across the toe',
        'wm': 'a fine-gauge navy wool crewneck pilled at the cuffs under an unstructured tweed jacket, slim dark denim, and suede desert boots scuffed at the heel',
        'hair': '{pos} hair air-dried and left uneven, cut near the jaw with a grown-out fringe falling into the brows',
        'makeup': 'a matte red lip pressed thin with the edges softened, skin otherwise bare and slightly shiny',
        'acc': 'a thin gold chain sitting close at the throat and a soft leather shoulder bag creased at the flap',
        'dem': 'leans one shoulder on whatever is nearest with a hand in a pocket, chin slightly down, gaze drifting off to the side and back',
    },
    'CLASSIC - preppy': {
        'wf': 'a white oxford shirt lightly rumpled with the sleeves pushed to the elbow, a navy cable-knit sweater knotted over the shoulders, a pleated knee-length skirt in green and navy check, and brown leather loafers scuffed at the heel',
        'wm': 'a pale blue oxford shirt creased at the elbows, a knit wool tie loosely knotted under an open collar, flat-front khaki chinos with a worn brown belt, and penny loafers creased across the vamp',
        'hair': '{pos} hair side-parted and combed back off the forehead, ends cut blunt above the collar',
        'makeup': 'clear balm on the lips and brows brushed up and left unfilled',
        'acc': 'a thin leather-strap watch with a scratched crystal and a canvas tote bag with a frayed handle',
        'dem': 'sits upright with ankles crossed and shoulders back, hands folded in the lap, gestures small and quick',
    },
    'CUTE - cottagecore': {
        'wf': 'a faded floral cotton dress with a gathered bodice and puffed sleeves, a linen apron marked with flour near the pocket, and worn leather ankle boots creased across the instep',
        'wm': 'a loose undyed linen shirt with the sleeves rolled past the elbow, brown canvas trousers held up by braces, and scuffed leather boots with dried mud on the welt',
        'hair': '{pos} hair gathered into a loose low braid with strands escaped around the face',
        'makeup': '',
        'acc': 'a woven willow basket with a chipped rim, a small bunch of dried lavender tucked at the waistband, and a thin brass ring worn smooth',
        'dem': 'stands with weight settled on one hip, hands occupied with what is being carried, gaze drifting off frame then returning, movements unhurried',
    },
    'CUTE - decora': {
        'wf': 'three cotton tank tops layered in stacked lengths over a long-sleeved striped shirt, a short pleated skirt over patterned leggings, and high-top sneakers with fraying mismatched laces',
        'wm': 'an oversized printed cotton tee over a long-sleeved striped shirt, wide nylon shorts pulled over knee socks in clashing patterns, and high-top sneakers scuffed grey at the toe',
        'hair': '{pos} hair in two short high pigtails under a thick blunt fringe, with shorter pieces sticking out at the sides',
        'makeup': 'a stripe of shimmer swept across each cheekbone, small stick-on stones dotted under the lower lashes, glossy balm on the lips',
        'acc': 'two dozen plastic snap clips in mixed colours crowded above one ear, stacked rubber and bead bracelets up both forearms, and a small worn plush charm clipped to a nylon bag strap',
        'dem': 'shifts weight from foot to foot, hands lifted near the face, turns the head quickly toward sound, holds a wide grin with the eyes crinkled',
    },
    'CUTE - fairycore': {
        'wf': 'a layered sheer cream chiffon dress with an uneven torn hem, a moss-green knit shrug pilled at the elbows, and thin ribbon laced up the bare calves',
        'wm': 'a loose sheer cream tunic open at the throat over cropped linen trousers frayed at the hem, a moss-green knit wrap pilled at the elbows, and bare feet dusted with dirt',
        'hair': '{pos} hair worn long and loose with two thin braids drawn back from the temples and tied behind',
        'makeup': 'fine shimmer dusted along the cheekbones and brow bones, tiny stick-on stones set at the outer eye corners, bare lips',
        'acc': 'a woven circlet of dried leaves and small dried flowers, a cord necklace holding a chipped clear stone, and a small glass jar tied at the hip',
        'dem': 'holds still with the chin slightly lifted, fingers trailing through the air near the shoulder, blinks slowly, rocks weight gently side to side',
    },
    'CUTE - gothic lolita': {
        'wf': 'a black cotton bell-shaped dress with a high square neckline and ruffled cuffs, stiff white petticoats gone slightly grey at the hem, black ribbed knee socks, and buckled patent shoes scuffed at the toe',
        'wm': 'a black wool tailcoat over a white cotton shirt with a ruffled front placket, black knee-length breeches, black ribbed knee socks, and buckled leather shoes worn dull at the heel',
        'hair': '{pos} hair set in loose ringlets falling past the shoulders with a straight fringe cut level with the eyebrows',
        'makeup': 'matte pale foundation, dark plum lipstick, thin black liner along the upper lash line flicked up short at the outer corner',
        'acc': 'a lace-trimmed headdress with two long hanging ribbons, a small cross on a tarnished silver chain, and a folded black parasol with a worn wooden handle',
        'dem': 'stands with the spine straight and shoulders back, hands folded low in front, chin level, head turning in small controlled movements',
    },
    'CUTE - kawaii street': {
        'wf': 'an oversized pastel hoodie with a printed cartoon face cracked from washing, a short pleated skirt, ribbed thigh-high socks, and chunky white sneakers grey along the sole',
        'wm': 'an oversized pastel hoodie with a printed cartoon face cracked from washing, wide cotton track pants with side stripes, and chunky white sneakers grey along the sole',
        'hair': '{pos} hair cut in a soft shoulder-length shag with a wispy fringe brushing the eyelashes',
        'makeup': 'cream blush blended across the nose and under the eyes, a small pale dot at each inner eye corner, clear gloss on the lips',
        'acc': 'a plush keychain hanging off a canvas tote, round plastic-framed glasses smeared with fingerprints, and a stretched fabric scrunchie worn on the wrist',
        'dem': 'leans slightly forward with rounded shoulders, sleeve cuffs pulled down over the fingers, head cocked to one side, gives small quick nods',
    },
    'CUTE - sweet lolita': {
        'wf': 'a pale pink cotton dress with a gathered waist and short puffed sleeves, a starched white pinafore apron, petticoats holding the skirt out stiff, white ankle socks with lace cuffs, and round-toed strapped shoes',
        'wm': 'a pale blue cotton shirt with a wide rounded collar and puffed sleeves, cropped cream shorts on button suspenders, white knee socks, and round-toed strapped shoes with a scuff across one side',
        'hair': '{pos} hair in two high curled pigtails tied with wide bows, fringe cut straight and full across the forehead',
        'makeup': 'sheer pink blush laid in a round patch high on each cheek, glossy pink balm on the lips, a thin line of pale shimmer under the brow',
        'acc': 'a padded strawberry-shaped shoulder bag, a wide satin headbow with slightly crushed loops, and a bracelet of plastic heart beads',
        'dem': 'stands with knees close and toes turned slightly inward, hands clasped at the waist, tilts the head in small quick movements, blinks wide and slow',
    },
    'DARK - deathrock': {
        'wf': 'a shredded black mesh top over a torn slip, a wide studded belt, laddered fishnets, and thick crepe-soled shoes worn uneven at the heel',
        'wm': 'a ripped sleeveless black shirt held together with safety pins, a leather jacket with paint-cracked lettering across the back, narrow trousers cut tight to the ankle, and thick crepe-soled shoes scuffed at the toe',
        'hair': '{pos} hair shaved above the ears with the top teased into a tall ragged fan and long thin strands left hanging at the temples',
        'makeup': 'a chalk-white base, black shadow dragged out into a wing across the socket, thin drawn-on brows, and dark lipstick pressed flat',
        'acc': 'a spiked leather collar with the points dulled, and a cluster of small bone-shaped charms hung on a safety pin',
        'dem': 'stands angular with one shoulder dropped and the chin tipped back, arms crossed high, holding still between short sharp movements',
    },
    'DARK - emo': {
        'wf': 'a fitted black tee with a small screen print cracked from washing, a studded belt over skinny black jeans faded at the knees, and canvas high-tops with ink scribbled on the rubber',
        'wm': 'a tight striped long-sleeve shirt under a short-sleeve black tee, skinny black jeans worn white at the thigh, and canvas high-tops with frayed laces',
        'hair': '{pos} hair cut in a long side-swept fringe that covers one eye, the back layered short and flicked out',
        'makeup': 'black liner traced heavily along both the upper and lower lids, pale skin, clear balm on the lips',
        'acc': 'a checkered fabric belt with a bent buckle, and a stack of thin rubber wristbands',
        'dem': 'sits hunched with the elbows on the knees and the head angled so the fringe covers one eye, picking at a sleeve cuff',
    },
    'DARK - goth (traditional)': {
        'wf': 'a long-sleeved black velvet dress with a frayed hem over laddered fishnet tights, and buckled boots scuffed grey at the toe',
        'wm': 'a black button-down shirt with fraying cuffs under a worn black waistcoat, narrow black trousers, and creased pointed boots',
        'hair': '{pos} hair backcombed high at the crown with a straight fringe cut level with the eyebrows',
        'makeup': 'matte pale foundation, black liner drawn thick and winged past the outer corner, dark lipstick blotted flat',
        'acc': 'a tarnished silver ankh on a thin chain, and a stack of scratched black rubber bangles',
        'dem': 'stands with the weight on one hip and the chin lowered so the gaze comes up from under the brow, hands still at the sides',
    },
    'DARK - grunge': {
        'wf': 'an oversized plaid flannel shirt hanging open over a stretched cotton slip dress, thick opaque tights with a run down one leg, and canvas sneakers greyed at the rubber',
        'wm': 'a plaid flannel shirt with the sleeves pushed up over a washed-out tee with a stretched collar, loose jeans frayed where they drag under the heel, and scuffed canvas sneakers',
        'hair': '{pos} hair cut to the chin and left unstyled, parted in the middle and falling over the eyes',
        'makeup': 'bare skin with a trace of dark liner smudged along the lower lash line',
        'acc': 'a knotted string bracelet gone dingy, and a beaded choker with one chipped bead',
        'dem': 'slouches with the shoulders rolled forward, gaze angled down and off to the side, hands pulled up into the sleeve cuffs',
    },
    'DARK - industrial': {
        'wf': 'a black vinyl top with a high zip collar, cargo trousers with strap buckles and dulled reflective taping, and heavy boots with the steel toe caps worn bare',
        'wm': 'a black technical jacket with taped seams and an off-center zip, a ribbed synthetic tee, wide trousers with webbing straps at the thigh, and heavy boots scuffed down to the metal',
        'hair': '{pos} hair shaved to stubble at the sides with the top pulled back into a short tight tail',
        'makeup': 'a flat pale base with a band of black shadow smudged straight across the eyes and out toward the temples',
        'acc': 'a webbing utility belt with worn buckles, and a rubber respirator hanging loose at the neck',
        'dem': 'stands square with the feet apart and the arms hanging heavy, face held still, turning the head before the body',
    },
    'DARK - metalhead': {
        'wf': 'a faded black tee with cracked screen print, a sleeveless denim vest covered in frayed sewn-on patches, black jeans, and high-top sneakers worn through at the toe',
        'wm': 'a black long-sleeve shirt washed thin, a cut-off denim vest with patches stitched on at odd angles, black jeans gone grey at the thighs, and heavy boots with the soles worn down',
        'hair': '{pos} hair grown long and parted down the middle, hanging loose well past the shoulders',
        'makeup': '',
        'acc': 'a wide leather wristband darkened with wear, and a pewter pendant on a leather cord',
        'dem': 'stands loose-shouldered with the head carried slightly forward, nodding on a slow beat, arms hanging or crossed low over the stomach',
    },
    'DARK - pastel goth': {
        'wf': 'a washed pink cropped sweater with pilled cuffs over a black mesh long-sleeve, a pleated black skirt, and chunky platform boots with scuffed white soles',
        'wm': 'a faded lavender oversized hoodie with a fraying drawstring over a black tee, black jeans torn at one knee, and thick-soled platform sneakers marked at the toe',
        'hair': '{pos} hair cut blunt at the collarbone with a straight fringe and two short strands left loose at the temples',
        'makeup': 'a pale matte base, a soft wash of shadow on the lid carried out into a small black flick at the outer corner, and a muted lip',
        'acc': 'a thin plastic choker with a small ring at the throat, and a plush keyring gone dingy on a bag strap',
        'dem': 'stands with the toes turned slightly inward and the shoulders drawn up, hands pulled inside the sleeves',
    },
    'DARK - punk': {
        'wf': 'a cut-off tee with the collar hacked open, a plaid skirt held shut with safety pins over torn tights, and boots with the leather split at the crease',
        'wm': 'a black leather jacket studded across the shoulders and cracked at the elbows, a bleached tee gone yellow at the neck, tight jeans torn through both knees, and boots laced with mismatched laces',
        'hair': '{pos} hair shaved close at the sides with the top spiked into a stiff upright crest',
        'makeup': 'black liner rubbed in around both eyes and smeared out at the corners, nothing else',
        'acc': 'a studded leather belt with a bent buckle, and a safety pin through one ear',
        'dem': 'stands with the weight forward and the shoulders squared, jaw set, hands shoved in the jacket pockets, shifting foot to foot',
    },
    'DARK - romantic goth': {
        'wf': 'a floor-length black lace gown with a laced bodice and trailing bell sleeves over a slip worn thin at the seams',
        'wm': 'a ruffled high-collar cotton shirt under a black brocade coat with dulled buttons, narrow trousers, and worn knee boots',
        'hair': '{pos} hair long and loosely waved, the upper half gathered and pinned at the crown with strands falling free at the face',
        'makeup': 'powdered pale skin, dark shadow softly smudged up to the brow bone, a deep matte lip',
        'acc': 'a velvet choker holding a small oval brooch carved with a profile, and a lace fan with one broken rib',
        'dem': 'holds the head tilted down and slightly to one side, one hand resting over the other at the waist, movements slow and drawn out',
    },
    'DARK - witchy': {
        'wf': 'a long tiered black linen skirt gone soft with washing, a loose crochet shawl snagged along one edge over a close-fitting top, and worn ankle boots',
        'wm': 'a wide-sleeved dark linen shirt with a laced neck opening, a long open coat dusty at the hem, loose trousers, and cracked leather boots',
        'hair': '{pos} hair long and unbrushed with a thin braid worked in at one side',
        'makeup': '',
        'acc': 'several tarnished silver bands worn across the fingers, a small leather pouch on a cord, and a pendant of rough unpolished stone',
        'dem': 'sits very still with the spine straight and the hands turned palm-up on the knees, blinking slowly, the gaze holding on the lens',
    },
    'GENRE - medieval': {
        'wf': 'a coarse undyed linen chemise under a side-laced wool kirtle gone thin at the elbows, with a scorched apron tied at the waist and cracked leather turnshoes',
        'wm': 'a rough wool tunic belted over patched hose, a hooded cloak worn shiny across one shoulder, and cracked leather ankle boots caked with dried mud',
        'hair': '{pos} hair grown to the collar and cut blunt at the ends, pushed back flat off the forehead',
        'makeup': '',
        'acc': 'a small drawstring pouch on a scuffed belt, and a plain iron ring worn smooth',
        'dem': 'stands with weight on one leg and shoulders slightly rounded, hands clasped low in front, gaze steady and unhurried',
    },
    'GENRE - pirate': {
        'wf': 'a salt-stiff linen shirt open at the throat under a cropped coat with tarnished buttons, a wide sash wound at the waist, and cuffed knee boots cracked across the instep',
        'wm': 'a loose shirt yellowed at the collar under a long coat worn through at the cuffs, a leather baldric across the chest, and canvas breeches tucked into salt-stained boots',
        'hair': '{pos} hair knotted back at the nape with pieces escaping in wind-stiffened tangles',
        'makeup': 'dark liner rubbed to a smudge all the way around both eyes',
        'acc': 'a faded cloth headscarf knotted at the back, a tarnished hoop earring, and a scarred belt with a heavy brass buckle',
        'dem': 'stands rocked back onto one heel with a hand resting on the belt, head tilted, gaze coming in at an angle',
    },
    'GENRE - post-apocalyptic wasteland': {
        'wf': 'a sun-bleached tank top under a canvas jacket with the sleeves cut away and the seams restitched in mismatched thread, cargo trousers stiff with dust, and boots bound at the toe with wire',
        'wm': 'a torn undershirt gone grey with dust under a layered canvas coat patched at the elbows with tire rubber, strapped knee pads, and split boots wrapped in tape',
        'hair': '{pos} hair shaved close at the sides with the top matted into thick uneven strands',
        'makeup': 'dust caked into the creases of the face with cleaner skin showing in a band where goggles sat',
        'acc': 'a cracked rubber respirator hanging at the neck, a dented canteen on a fraying strap, and knuckles wrapped in dirty cloth tape',
        'dem': 'holds still with the shoulders drawn in, eyes scanning off to one side before settling on camera, hands never fully at rest',
    },
    'GENRE - renaissance faire': {
        'wf': 'a white cotton chemise with gathered sleeves under a front-laced brocade bodice, and a full skirt hitched up at one hip over a striped underskirt with a grass-stained hem',
        'wm': 'a loose cotton shirt laced open at the throat, a quilted leather jerkin sun-faded across the back, and dark breeches tucked into folded-top boots creased at the ankle',
        'hair': '{pos} hair loosely braided back with several strands pulled out around the face',
        'makeup': 'flushed cheeks and a soft stain on the lips over otherwise bare skin',
        'acc': 'a dented pewter tankard hooked to the belt, a ring of iron keys, and a wilting flower crown',
        'dem': 'leans in with the weight forward, hands moving wide and open as they talk, head tipped back when laughing',
    },
    'GENRE - steampunk': {
        'wf': 'a high-necked cotton blouse under a boned brown leather corset with worn buckle straps, a hitched skirt over ribbed stockings, and laced boots scuffed pale at the toe',
        'wm': 'a collarless shirt held with sleeve garters, a brass-buttoned brocade waistcoat, and pinstriped trousers over oil-marked leather boots',
        'hair': '{pos} hair swept back and pinned at the crown with a few pieces falling loose over one ear',
        'makeup': 'thin liner along the upper lash line and a smudge of soot at the temple',
        'acc': 'goggles pushed up on the forehead with the leather strap sweat-darkened, a brass pocket watch on a worn chain, and one fingerless glove split at the knuckle',
        'dem': 'sits forward with elbows on the knees, turning a small object over in the fingers, eyes flicking down and back up',
    },
    'GENRE - viking': {
        'wf': 'a heavy coarse-weave wool underdress beneath a strap-hung apron dress pinned at the shoulders, and a fur-lined mantle matted at the collar',
        'wm': 'a wool tunic with woven trim frayed at the neck opening, leather bracers scarred across the forearms, and cross-gartered trousers under a fur-collared cloak',
        'hair': '{pos} hair pulled into tight braids at the temples with the rest hanging loose and wind-tangled',
        'makeup': 'soot smeared along the brow ridge and rubbed thin under the lower lids',
        'acc': 'a cast bronze cloak pin gone green in the crevices, a bone-handled knife on a belt loop, and a braided leather cord at the wrist',
        'dem': 'stands wide-footed and square to camera, chin lowered, hands hanging heavy and still at the sides',
    },
    'GENRE - western/cowboy': {
        'wf': 'a faded pearl-snap shirt tucked into high-waisted denim gone white along the seams, a tooled leather belt, and stacked-heel boots dulled with red dust',
        'wm': 'a chambray work shirt with sweat rings at the collar, a canvas duster stiff with road dust, denim worn pale at the knee, and roughout boots with the heels run down',
        'hair': '{pos} hair flattened at the crown from a hat brim and curling out over the collar',
        'makeup': '',
        'acc': 'a sweat-stained felt hat held down against the thigh, a coiled leather rope over one shoulder, and a limp bandana around the neck',
        'dem': 'stands with one hip cocked and thumbs hooked in the belt, chin dipped so the gaze comes up from under the brow, moves slowly',
    },
    'SPORT - climber': {
        'wf': 'a thin nylon tank with chalk handprints smeared down the front, stretch canvas trousers rolled at the ankle, and a webbing harness with the leg loops gone fuzzy',
        'wm': 'a faded cotton tee with the collar torn loose, soft canvas trousers patched at one knee, and a webbing harness with a chalk bag swinging at the hip',
        'hair': '{pos} hair pushed back under a folded fabric band with the ends curling out at the nape',
        'makeup': '',
        'acc': 'cloth tape wound around two split knuckles and a chalk bag with a frayed drawstring',
        'dem': 'crouches low with the forearms braced on the knees, fingers opening and closing, eyes tracking upward',
    },
    'SPORT - cyclist': {
        'wf': 'a snug zip-front jersey with three open pockets across the lower back and a salt ring at the collar, padded bib shorts with the leg grippers rolled, and stiff-soled shoes scuffed at the plastic sole',
        'wm': 'a close-cut zip jersey with faded colour blocks and a stretched hem, black bib shorts shiny at the seat, and stiff-soled shoes with cracked ratchet straps',
        'hair': '{pos} hair pressed flat in stripes by a helmet with damp lines down the temples',
        'makeup': '',
        'acc': 'fingerless gloves with the palm padding crushed flat and scratched wraparound glasses pushed up onto the forehead',
        'dem': 'sits hunched with the forearms on the thighs, one heel jittering, chest rising hard',
    },
    'SPORT - gym/bodybuilder': {
        'wf': 'a cropped ribbed tank with the neckline stretched wide, high-waisted compression shorts with the waistband rolled once, and flat canvas shoes creased across the toe',
        'wm': 'a cotton tee with the sleeves hacked open to the ribs and a salt line at the collar, loose knit shorts with a knotted drawstring, and flat leather shoes scuffed bald at the heel',
        'hair': '{pos} hair pulled back flat against the skull with short pieces escaping damp at the nape',
        'makeup': '',
        'acc': 'a wide leather belt with cracked stitching and a pair of cloth wrist wraps powdered white with chalk',
        'dem': 'stands square with the arms held out from the ribs, chin dropped, weight rocking heel to heel',
    },
    'SPORT - hunter/outdoorsman': {
        'wf': 'a waxed canvas coat gone shiny at the elbows over a wool plaid shirt, canvas trousers with mud dried into the knees, and leather boots cracked across the flex line',
        'wm': 'a heavy wool plaid shirt with a burn hole at the cuff under a quilted vest in dulled blaze orange, canvas trousers stiff with dried mud, and laced leather boots caked to the welt',
        'hair': '{pos} hair cut short at the neck and pressed into a flat ring where a cap sits all day',
        'makeup': '',
        'acc': 'leather gloves stiffened by rain with the fingertips worn thin and a dented enamel mug clipped to the belt',
        'dem': 'stands still with the weight settled onto one leg, head turning slowly to scan, hands quiet at the sides',
    },
    'SPORT - martial artist': {
        'wf': 'a heavy cotton jacket with reinforced stitching crossed and held by a soft cloth belt, wide matching trousers gone thin at the knees, and bare feet with taped toes',
        'wm': 'a coarse cotton jacket frayed along the lapel, a faded cloth belt tied flat at the waist, wide drawstring trousers with a mended seam, and bare calloused feet',
        'hair': '{pos} hair pulled tight to the skull and knotted low so nothing falls across the face',
        'makeup': '',
        'acc': 'cloth hand wraps worn grey across the knuckles and a folded towel over one shoulder',
        'dem': 'stands with the feet set apart and hands open at the sides, breathing slow, gaze held level and unmoving',
    },
    'SPORT - skater': {
        'wf': 'an oversized cotton tee knotted at the hip, baggy canvas trousers shredded where they drag under the heel, and low canvas shoes worn through at the toe to the lining',
        'wm': 'a washed-thin flannel hanging open over a stretched cotton tee, wide denim frayed off at the cuff, and suede shoes scuffed bald along the outer side',
        'hair': '{pos} hair grown out past the ears and flattened at the crown where a cap sits',
        'makeup': '',
        'acc': 'a scratched skateboard held under one arm with the grip tape rubbed pale in two patches, and a loose elastic bandage around one knee',
        'dem': 'stands with the heels together and toes turned out, shoulders slumped forward, fingers picking at the board edge',
    },
    'SPORT - surfer': {
        'wf': 'a black neoprene suit peeled down to the waist over a sun-bleached bikini top with stiffened ties, and bare feet with dried sand up the ankles',
        'wm': 'a neoprene suit unzipped and shoved to the hips over salt-dried skin, loose board shorts with a stiff knotted cord, and bare feet crusted with sand',
        'hair': '{pos} hair dried stiff and raked back in uneven lengths with the ends splitting',
        'makeup': 'a thick smear of white zinc paste across the nose and both cheekbones',
        'acc': 'a braided cord bracelet gone rigid with salt and an ankle leash trailing a cracked plastic cuff',
        'dem': 'stands loose with one hip dropped, squinting into the light, one hand raised to shade the eyes',
    },
    'SPORT - swimmer': {
        'wf': 'a high-necked one-piece with thin straps and the fabric gone slack across the back, a silicone cap pushed up off the ears, and a thin towel over one shoulder',
        'wm': 'close-cut trunks with a knotted drawstring and panels faded by chlorine, a silicone cap rolled back to the crown, and a coarse towel draped around the neck',
        'hair': '{pos} hair flattened to the skull and dripping at the ends where the cap has slipped back',
        'makeup': '',
        'acc': 'goggles pushed onto the forehead with clouded seals and a stretched strap, and a worn towel bunched in one hand',
        'dem': 'stands with the shoulders rolled forward and arms hanging heavy, water running off the elbows, blinking slowly',
    },
    'SPORT - track athlete': {
        'wf': 'a thin mesh singlet with a paper number safety-pinned at the chest, split-side shorts with a frayed hem, and thin-soled spiked shoes ground grey at the toe',
        'wm': 'a sweat-darkened sleeveless singlet clinging at the ribs, split-side shorts over compression briefs worn thin at the seam, and low spiked shoes with grit packed into the soles',
        'hair': '{pos} hair scraped back off the face into a short tail with damp strands stuck at the temples',
        'makeup': '',
        'acc': 'a plastic digital watch with a scratched face and a strip of white tape wound around one ankle',
        'dem': 'stands with the weight forward on the balls of the feet, shoulders rolling loose, chest still rising hard',
    },
    'STREET - graffiti artist': {
        'wf': 'a hooded sweatshirt flecked with overspray and stretched out at the cuffs, a canvas work jacket with a torn chest pocket, loose carpenter jeans stiff with dried paint, and canvas sneakers worn through at the toe',
        'wm': 'a faded long-sleeve tee under a hooded sweatshirt speckled with dried paint, heavy cotton work pants gone shiny at the knees, and scuffed leather boots with paint crusted into the laces',
        'hair': '{pos} hair grown out and pushed back off the forehead, flattened on one side where a hood has sat',
        'makeup': '',
        'acc': 'a dust mask hanging loose around the neck on stretched straps, fingerless gloves stained at the fingertips, and a spray nozzle held between two fingers',
        'dem': 'stands half turned away with the shoulders rolled forward, looks back over one shoulder toward the camera, hands moving in small quick gestures',
    },
    'STREET - hip-hop': {
        'wf': 'a fitted ribbed tank top under an unzipped satin bomber with cracked embroidery on the back, low-slung baggy denim faded pale at the knees, and white leather high-tops kept clean with grey soles',
        'wm': 'a long white tee gone soft from washing under a heavy zip-up hoodie with the drawstrings pulled out, baggy dark denim breaking over the laces, and boxy leather sneakers scuffed at the heel',
        'hair': '{pos} hair cut short and brushed flat with a hard line shaved along the temple',
        'makeup': 'high-shine gloss on the lips, a thin dark line drawn tight along the upper lash line, and brows combed up and squared off',
        'acc': 'a heavy flat-link chain worn outside the shirt, thick hoop earrings, and a chunky watch with a scratched face',
        'dem': 'leans back with weight on one hip, chin lifted, one hand resting on the chain and the other loose at the side, moving slowly between holds',
    },
    'STREET - sneakerhead': {
        'wf': 'a plain heavyweight tee tucked at the front over cuffed grey sweat shorts, ribbed crew socks pulled to mid calf, and box-fresh leather basketball shoes with the tongue standing tall and the laces loosely crossed',
        'wm': 'a boxy short-sleeve shirt layered over a long-sleeve thermal, tapered black track pants cuffed above the ankle to clear the shoe, and unworn white leather high-tops with the heel tab still stiff',
        'hair': '{pos} hair clipped short and even all over with a straight line trimmed at the nape',
        'makeup': '',
        'acc': 'a nylon shoe bag hooked over one shoulder, a stiff-bristled brush with the paint worn off its handle, and a folded microfiber cloth in one hand',
        'dem': 'stands with the feet angled out to keep the toe boxes in frame, weight even, glancing down at the shoes and back up, hands mostly still',
    },
    'STREET - streetwear': {
        'wf': 'a boxy washed-black hoodie with a cracked screen print across the chest, an open flannel shirt gone soft at the elbows, wide cotton track pants with taped side seams, and thick-soled canvas high-tops creased across the toe',
        'wm': 'a heavyweight cotton tee yellowed slightly at the collar under a boxy nylon coach jacket with a stuck zipper, loose carpenter jeans stacking over the shoe, and low canvas sneakers scuffed grey at the sidewall',
        'hair': '{pos} hair cropped close at the sides with a few inches left on top, pushed forward and flattened where a cap has sat',
        'makeup': 'a wash of tinted balm on the lips and brows brushed straight up, skin left shiny across the cheekbones',
        'acc': 'a curved-brim cap with the brim bent hard and the size sticker still on it, and a canvas crossbody bag with a frayed strap',
        'dem': 'stands with weight dropped onto one leg, hands buried in the front pocket, head tipped a little to one side, shifting every few seconds',
    },
    'STREET - utility/gorpcore': {
        'wf': 'a hooded ripstop shell in faded olive with taped seams over a gridded fleece half-zip, cargo pants with bellows pockets and a webbing belt, and lugged trail shoes with dried mud packed in the tread',
        'wm': 'a boxy nylon anorak with a storm flap and a wind-worn hem over a waffle-knit base layer, ripstop trousers cinched at the ankle with drawcords, and lugged approach shoes worn shiny at the heel',
        'hair': '{pos} hair pushed back flat and slightly damp, grown past the ears and tucked behind them',
        'makeup': '',
        'acc': 'a small nylon sling bag clipped across the chest with a scratched buckle, and a carabiner of keys hanging from a belt loop',
        'dem': 'stands square with feet planted wide, thumbs hooked under the chest strap, gaze level and steady, barely moving',
    },
    'TECH - cyberpunk': {
        'wf': 'a cropped vinyl jacket cracked white at the elbows over a sheer mesh top, spliced black denim with the zip seams exposed, and buckled boots with the steel toe caps showing through worn leather',
        'wm': 'a long coated-canvas coat with a scorched hem over a ribbed black tank, taped cargo trousers with a strip of amber light cord stitched down one leg, and heavy boots rubbed to bare metal at the toe',
        'hair': '{pos} hair shaved to stubble above one ear with the rest falling long and uneven across the other side',
        'makeup': 'a hard band of dark pigment painted temple to temple across the eyes, gloss on the lower lip only',
        'acc': 'a cracked earpiece clipped over one ear with a hairline split in the housing, and a coil of thin cable looped twice around the wrist with the connector taped',
        'dem': 'keeps the chin tipped down and looks up into the lens, shoulders angled away from square, one hand turning a small object over and over',
    },
    'TECH - gamer': {
        'wf': 'an oversized faded graphic tee with the screen-print cracked across the chest, drawstring shorts, and mismatched crew socks pushed down at the ankle',
        'wm': 'a heather grey hoodie with the drawstring tips chewed flat, loose track pants worn thin at the knee, and slide sandals over thick socks',
        'hair': '{pos} hair pressed into a flat band across the crown from a headset, the rest pushed back with fingers',
        'makeup': '',
        'acc': 'a padded over-ear headset with the earpad foam split at the seam, and a stretched-out rubber wristband',
        'dem': 'leans in toward the screen with both elbows planted, eyes tracking fast side to side, jaw loose between short bursts of speech',
    },
    'TECH - maker/tinkerer': {
        'wf': 'a faded canvas apron pocked with scorch pinholes over a rolled-sleeve chambray shirt, denim work trousers stiff with dried glue at the thigh, and rubber-soled shoes flecked with paint',
        'wm': 'a canvas work shirt with a burn-marked chest pocket and sleeves rolled to the elbow, heavy cotton trousers frayed at the cuff, and leather boots spattered with hardened solder at the toe',
        'hair': '{pos} hair tied back off the face with loose strands escaping at the temples',
        'makeup': '',
        'acc': 'a magnifying visor pushed up onto the forehead with the hinge wrapped in tape, a multimeter probe hooked into the apron pocket, and a rubber band around one wrist',
        'dem': 'sits hunched over the bench with forearms resting on it, hands always adjusting something small, looking up at the lens only in the gaps between adjustments',
    },
    'TECH - programmer': {
        'wf': 'a plain charcoal long-sleeve tee bobbled at the elbows, dark straight-leg jeans, and canvas sneakers creased deep across the toe',
        'wm': 'a zip-front fleece with a pilled collar over a plain tee, flat-front chinos wrinkled behind the knee, and low leather sneakers scuffed grey at the heel',
        'hair': '{pos} hair cut short and grown two months past the cut, pushed off the forehead and left where it falls',
        'makeup': '',
        'acc': 'clear-framed glasses with fingerprint smears low on the lenses, and a dented steel bottle with sticker residue on one side',
        'dem': 'sits back with one ankle crossed over the opposite knee, hands still until they lift to mark a point in the air, gaze sliding off past the lens mid-sentence then coming back',
    },
    'TECH - techwear': {
        'wf': 'a matte black ripstop shell with taped seams and a magnetic chest buckle, cropped nylon trousers cinched tight at the calf, and lug-soled boots dusted grey at the toe',
        'wm': 'a hooded charcoal softshell with a high collar and a diagonal chest zip, articulated black nylon trousers strapped across the thigh, and lug-soled boots with the laces routed through webbing loops',
        'hair': '{pos} hair pulled tight to the scalp and fastened at the nape, the hairline left flat and unbroken',
        'makeup': '',
        'acc': 'a webbing sling bag worn across the chest with the strap end taped down, and a rubber-strapped digital watch with a scratched face',
        'dem': 'stands square with weight even on both feet, shoulders dropped, one hand hooked through the sling strap, head turning before the body follows',
    },
    'VINTAGE - 90s revival': {
        'wf': 'a cropped ribbed tank over a plain white cotton tee, baggy pale-wash jeans torn open at one knee, and chunky lace-up shoes with grimy soles',
        'wm': 'an oversized flannel shirt with a frayed soft collar hanging open over a faded band tee, wide dark jeans puddling over the shoe, and canvas sneakers gone grey at the toe',
        'hair': '{pos} hair parted down the middle and tucked behind the ears, the ends left blunt and slightly ragged',
        'makeup': 'a brown lip liner drawn just outside a matte lip, thin plucked brows, and a flat matte base',
        'acc': 'a black cord choker sitting flat on the throat and a plastic-strapped watch with a scratched face',
        'dem': 'slouches with the shoulders rounded and hands pushed into the pockets, glancing up at the lens and away again',
    },
    'VINTAGE - Y2K': {
        'wf': 'a shrunken pale blue baby tee over low-slung bootcut jeans with contrast stitching and a rhinestone back pocket, and pointed heeled boots scuffed at the toe',
        'wm': 'a slick silver-grey nylon track jacket zipped to mid-chest over a tight ribbed tee, low-rise bootcut jeans frayed where they drag, and chunky white sneakers greyed along the sole',
        'hair': '{pos} hair straightened flat with face-framing pieces pulled forward and the rest twisted into small clipped sections at the crown',
        'makeup': 'frosted pale gloss on the lips, silver shimmer on the lid, thin over-plucked brows, and glitter dusted along the cheekbone',
        'acc': 'a beaded plastic choker, a small shoulder bag in cracked shiny vinyl, and a flip phone with a loose hinge',
        'dem': 'stands with one hip pushed out and shoulders angled to the lens, chin dipped, fingers picking at the bag strap',
    },
    'VINTAGE - beatnik': {
        'wf': 'a black fine-knit turtleneck pilled at the elbows, narrow cropped black trousers gone shiny at the seat, and flat black shoes scuffed at the heel',
        'wm': 'a black ribbed turtleneck stretched loose at the neck under a rumpled brown corduroy jacket worn bald at the elbows, straight dark trousers, and flat leather shoes soft at the crease',
        'hair': '{pos} hair worn flat and unstyled, pushed back off the forehead and left where it falls',
        'makeup': 'black liner smudged along both lash lines with bare unpainted lips',
        'acc': 'a paperback with a cracked spine held in one hand, horn-rimmed glasses with a taped joint, and a thin unlit cigarette',
        'dem': 'sits folded forward with elbows on the knees, gaze off past the edge of the lens, hands still except when they cut the air to speak',
    },
    'VINTAGE - disco': {
        'wf': 'a halter jumpsuit in slick gold-toned jersey creased at the hip with a plunging neck and wide flared legs, and platform sandals with a scuffed cork sole',
        'wm': 'a wide-collared shirt in thin printed polyester unbuttoned to the sternum, high-waisted flared cream gabardine trousers with a pressed crease, and platform leather boots with the heel rubbed at the back',
        'hair': '{pos} hair blown out into full lifted layers swept back off the face and feathered at the sides',
        'makeup': 'shimmer swept across the lid up to the brow bone, blush laid in a hard diagonal under the cheekbone, and glossed lips',
        'acc': 'a flat chain lying against bare skin with the plating rubbed thin, and a wide metal cuff bracelet',
        'dem': 'stands with hips loose and one arm raised overhead, weight rocking steadily side to side, chin up and lids low',
    },
    'VINTAGE - greaser': {
        'wf': 'a fitted white ribbed cotton tank tucked into high-waisted dark denim rolled thick at the ankle, a black leather jacket rubbed grey at the elbows, and black boots worn pale at the toe',
        'wm': 'a white cotton t-shirt yellowed at the collar under a black leather jacket cracked across the shoulders, dark denim jeans with a deep cuff, and black leather boots with the heels worn down on one side',
        'hair': '{pos} hair combed up high off the forehead into a swept roll with the sides slicked flat and a few strands hanging loose over the brow',
        'makeup': 'a thin smudged black line along the upper lash line and bare lips',
        'acc': 'a thin steel chain looping from belt to pocket, a soft cigarette pack rolled into the sleeve, and a plain dulled steel ring',
        'dem': 'leans back with shoulders dropped and one thumb hooked in a pocket, gaze level and unmoving, shifting weight slowly',
    },
    'VINTAGE - hippie/boho': {
        'wf': 'a loose gauze blouse with wide sleeves and a tie at the throat, a long tiered skirt in faded floral cotton with a frayed hem, and flat leather sandals darkened at the footbed',
        'wm': 'an embroidered cotton shirt washed thin at the shoulders worn open over a plain knit vest, flared denim jeans fraying under the heel, and scuffed leather sandals',
        'hair': '{pos} hair grown long and parted down the middle, unbrushed and falling past the shoulders with two small braids at one temple',
        'makeup': '',
        'acc': 'a strand of small wooden beads, a woven cloth bag with a fringed strap gone limp, and stacked thin metal rings rubbed dull',
        'dem': 'sits cross-legged with shoulders loose, head tilting as they talk, hands turning over slowly in the lap',
    },
    'VINTAGE - mod': {
        'wf': 'a sleeveless A-line shift dress in pressed cotton twill with cream and black panels, sheer patterned tights, and low square-toed leather shoes scuffed at the strap',
        'wm': 'a slim three-button grey wool jacket with narrow lapels over a pointed-collar shirt and a flat knit tie, tapered trousers cropped above the ankle, and polished leather boots creased at the instep',
        'hair': '{pos} hair cut blunt and close to the head with a straight fringe sitting level just above the brows',
        'makeup': 'pale matte lips, a heavy black line drawn along the socket crease, and lower lashes painted on in separate strokes',
        'acc': 'a narrow leather belt with a square buckle and round tinted sunglasses with one scratched lens',
        'dem': 'stands square with arms straight at the sides, head held still, eyes fixed forward, movements small and abrupt',
    },
    'VINTAGE - pin-up/rockabilly': {
        'wf': 'a cotton halter dress with a fitted bodice and full gathered skirt in small white dots on red, a wide black elastic waist cinch cracked at the fold, and cream peep-toe heels scuffed at the toe',
        'wm': 'a short-sleeved cream rayon bowling shirt with a black yoke and contrast piping, cuffed dark denim jeans gone pale at the knee, and black leather lace-up shoes creased across the vamp',
        'hair': '{pos} hair set in wide rolled curls swept up and pinned high at the front, the rest falling in a smooth curve to the shoulder',
        'makeup': "a winged black liner flicked past the outer corner, matte red lipstick with a defined cupid's bow, and a thin arched brow",
        'acc': 'a knotted cotton headscarf faded along the folds, small gold-tone hoop earrings, and a cracked patent belt',
        'dem': 'stands with weight dropped onto one hip and shoulders turned back, chin lifted to the lens, one hand resting flat at the waist',
    },
    'WORK - barista': {
        'wf': 'a fitted tee with a small chest print cracked from washing under a canvas half apron flecked with dried milk at the hem, high-waisted jeans cuffed at the ankle, and flat rubber-soled shoes',
        'wm': 'a short-sleeved button shirt open at the throat with coffee spots on one cuff, a denim apron with brass rivets and a stained front pocket, dark jeans, and canvas shoes worn thin at the heel',
        'hair': '{pos} hair kept off the face with a folded cloth band, the shorter pieces tucked behind the ears',
        'makeup': '',
        'acc': 'a damp bar towel folded over the apron string, a notebook with a curled cover in the apron pocket, and a stack of thin metal rings on one wrist',
        'dem': 'stands with the hands braced on the counter edge, tilts the head while listening, quick small movements between stretches of stillness',
    },
    'WORK - chef': {
        'wf': 'a double-breasted white jacket greyed at the cuffs with the sleeves folded back twice, a long apron scorched brown near one hip, and checked cotton trousers gone limp at the knee',
        'wm': "a short-sleeved cook's jacket with the collar unbuttoned and the fabric steam-limp at the back, a bib apron tied twice with the strings knotted in front, and dark trousers with a small burn hole at the thigh",
        'hair': '{pos} hair pushed back and held flat under a folded cloth band, damp along the hairline',
        'makeup': '',
        'acc': 'a side towel over one shoulder with brown scorch marks, a plastic-handled paring knife in the apron pocket, and a plain worn band on one finger',
        'dem': 'leans forward with the forearms on the counter, wipes the hands on the apron between sentences, gaze steady on the lens',
    },
    'WORK - farmer/rancher': {
        'wf': 'a plaid cotton shirt with the sleeves rolled and the elbows worn thin, a quilted vest split along one pocket seam, and stiff denim jeans stacked over cracked leather boots',
        'wm': 'a chambray work shirt sun-bleached across the shoulders, canvas bib overalls stained brown at the knees, and pull-on boots caked with dried mud along the sole',
        'hair': '{pos} hair flattened in a ring where a hat brim sat, the ends sun-dried and cut uneven',
        'makeup': '',
        'acc': 'a sweat-stained straw hat held down at the thigh, a folding knife rubbed shiny in the front pocket, and leather gloves stiffened with dried dirt',
        'dem': 'stands loose with one thumb hooked in a pocket, shifting the weight slowly, looking off to the side then back to the camera',
    },
    'WORK - mechanic/blue collar': {
        'wf': 'a faded navy work shirt with the sleeves rolled above the elbow and the name patch fraying at one corner, a canvas apron marked with old grease handprints, and heavy denim trousers worn pale across the knees',
        'wm': 'a short-sleeved cotton work shirt with oil worked into the weave at the chest, dark blue coveralls unzipped to the waist with the arms knotted in front, and steel-toed boots scuffed grey at the toe',
        'hair': '{pos} hair pushed back off the forehead and flattened in a band where a cap sat',
        'makeup': '',
        'acc': 'a shop rag folded into the back pocket, a plastic-strap watch with a scratched face, and a knuckle bandage gone grey at the edges',
        'dem': 'stands with the weight on one hip, wiping the hands slowly on a rag, glancing up at the camera and down again',
    },
    'WORK - medical': {
        'wf': 'loose cotton scrubs faded soft from washing with the waist drawstring double-knotted, a thin long-sleeved top under the V-neck, and rubber clogs marked at the toe',
        'wm': 'a short white coat creased at the elbows over pale blue scrubs worn thin at the seams, a plain crew tee showing at the neckline, and rubber-soled shoes scuffed grey along the sides',
        'hair': '{pos} hair pulled back flat and secured clear of the collar, with a crease across the front where a cap sat',
        'makeup': '',
        'acc': 'a stethoscope with cracked tubing folded over one shoulder, a laminated badge on a retractable reel, and a chewed pen clipped to the chest pocket',
        'dem': 'stands with the shoulders level and the hands clasped at the waist, holding eye contact and blinking little',
    },
    'WORK - military/tactical': {
        'wf': 'a mottled green and tan field jacket with the cuffs fastened tight over a plain olive tee gone thin at the neck, ripstop trousers bloused into laced boots, and a nylon belt rubbed shiny at the buckle',
        'wm': 'a sand-coloured combat shirt with dried sweat rings at the collar under a dusty plate carrier, ripstop trousers with reinforced knees scuffed pale, and boots crusted with dry grit along the welt',
        'hair': '{pos} hair cut short and even at the back and sides, flattened at the crown from a helmet',
        'makeup': '',
        'acc': 'a nylon-strap watch with a scratched bezel, a strip of matte tape on the shoulder furred at the edges, and split-palm gloves tucked under the belt',
        'dem': 'stands square with the feet apart and the hands resting on the front of the vest, scanning past the camera before settling on it',
    },
    'WORK - painter/artist': {
        'wf': 'a cotton smock flecked with dried colour over a stretched tank top, loose linen trousers streaked where the hands were wiped down the thighs, and canvas shoes stiffened with spatter',
        'wm': 'a washed-out button shirt worn open with crusted paint down the front over a thin tee, corduroy trousers rubbed bare at the knees, and canvas shoes split along one seam',
        'hair': '{pos} hair pushed back with the flat of a hand and drying in uneven separated pieces',
        'makeup': '',
        'acc': 'a rag tucked in the waistband stiff with dried colour, a pencil behind the ear worn down to a stub, and tape wound around one thumb',
        'dem': 'sits forward with the elbows on the knees, turning a brush over in the hands, gaze dropping to the hands and lifting again',
    },
    'WORK - pilot': {
        'wf': 'a white short-sleeved uniform shirt with epaulettes and a fold crease still set in the sleeve, a dark clipped tie, and navy trousers with a flattened centre crease over plain black shoes',
        'wm': 'a white uniform shirt with striped epaulettes and the collar softened from wear, a navy tie tucked at the third button, and navy trousers over black shoes scuffed pale at the toe',
        'hair': '{pos} hair combed flat to one side and held down, trimmed clean above the collar',
        'makeup': '',
        'acc': 'a metal wing pin sitting slightly crooked on the chest pocket, a watch with a worn leather strap and large numerals, and a laminated ID on a belt clip',
        'dem': 'sits upright with the back off the seat, hands flat on the thighs, speaking with the head held still',
    },
}


def _an(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def _realism_block(age, pronoun_pos):
    if age == "teens":
        return "clear matte skin and natural unstyled brows"
    if age in ("sixties", "seventies"):
        return ("weathered skin with deep creases at the eyes, visible pores, "
                "dry lips, no makeup")
    return ("matte skin with visible pores across the nose and cheeks, faint "
            "shadows under the eyes, dry lips, no makeup")


class RiftCast_CharacterDesigner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "NOVA"}),
            "gender": (list(GENDER.keys()),),
            "age": (AGE, {"default": "mid twenties"}),
            "ethnicity": (ETHNICITY,),
            "skin": (SKIN, {"default": "pale"}),
            # FALLBACK FOUR. These four duplicate widgets on the Style node and
            # are used ONLY when style_block is unwired (Designer used alone).
            # Wiring a Style node discards them - which was silent, and silent
            # dead controls are the exact "dropdowns appear to do nothing" trap
            # called out at AUDITION ISOLATION below. design() now says so out
            # loud when a wired style overrides one you actually changed.
            "hair_color": (HAIR_COLOR, {"default": "dark brown", "tooltip":
                           "FALLBACK - ignored when a Style node is wired to "
                           "style_block; set hair colour on the Style node."}),
            "hair_style": (HAIR_STYLE, {"default": "long and loose", "tooltip":
                           "FALLBACK - ignored when a Style node is wired to "
                           "style_block; set hair shape on the Style node."}),
            "height": (HEIGHT, {"default": "average height"}),
            "build": (BUILD, {"default": "average build"}),
            "voice": (VOICE, {"default": "a clear voice, natural and unforced"}),
            "accent": (ACCENT, {"default": "casual American", "tooltip":
                       "American variants are enforced by the accent LoRA at "
                       "24fps. British/Australian render natively at 25/30fps "
                       "(the fps-accent law) - or at 24fps they lean on the "
                       "wording alone."}),
            "wardrobe": ("STRING", {"default": "a plain dark crewneck shirt",
                         "tooltip": "FALLBACK - ignored when a Style node is "
                         "wired to style_block; set wardrobe on the Style node."}),
            "distinguishing": ("STRING", {"default": "", "tooltip":
                               "optional: thin-framed glasses, a small nose "
                               "stud, freckles across the nose... FALLBACK - "
                               "ignored when a Style node is wired; the Style "
                               "node has its own distinguishing field."}),
        },
        "optional": {
            "style_block": ("STRING", {"forceInput": True, "tooltip":
                            "Wire a RiftCast Style + Wardrobe node here."}),
            "demeanor": ("STRING", {"forceInput": True, "tooltip":
                         "Wire the Style node's demeanor output here."}),
            "template": ("STRING", {"forceInput": True, "tooltip":
                         "Wire a RiftCast Audition Script node here to "
                         "customize the casting call. Unwired = the "
                         "built-in default."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("audition_script", "dna_text", "character_name")
    FUNCTION = "design"
    CATEGORY = "JoyAI-Echo/RiftCast"

    def design(self, name, gender, age, ethnicity, skin, hair_color, hair_style,
               height, build, voice, accent, wardrobe, distinguishing,
               template=None, style_block=None, demeanor=None):
        name = (name or "NOVA").strip().upper()
        noun, pro, pos, _ = GENDER[gender]
        nationality = ("British" if accent == "British" else
                       "Australian" if accent == "Australian" else "American")
        eth = "" if ethnicity == "(unspecified)" else f" {ethnicity}"
        extra = f", {distinguishing.strip()}" if distinguishing.strip() else ""
        teen_word = "teenaged " if age == "teens" else ""
        bare_build = build.replace(" build", "")
        appearance = None
        realism = _realism_block(age, pos)
        if style_block:
            try:
                _sb = json.loads(style_block)
                _w = _sb["wf"] if gender == "woman" else _sb["wm"]
                appearance = (_sb["block"].replace("{__WARDROBE__}", _w)
                              .replace("{pos}", pos))
                # the realism block asserts "no makeup"; a style that supplies
                # makeup would contradict it in the same sentence
                if _sb.get("has_makeup"):
                    realism = realism.replace(", no makeup", "")
                # Say it out loud, but only when it MATTERS: a field left at its
                # default was never a choice, so warning about it would be noise
                # everyone learns to ignore.
                _shadowed = [n for n, v, dflt in
                             (("hair_color", hair_color, "dark brown"),
                              ("hair_style", hair_style, "long and loose"),
                              ("wardrobe", wardrobe, "a plain dark crewneck shirt"),
                              ("distinguishing", distinguishing, ""))
                             if (v or "").strip() != dflt]
                if _shadowed:
                    print(f"[RiftCast] Style node is wired, so the Designer's "
                          f"{', '.join(_shadowed)} {'is' if len(_shadowed)==1 else 'are'} "
                          f"IGNORED. Set {'it' if len(_shadowed)==1 else 'them'} "
                          f"on the Style node instead.", flush=True)
            except Exception as _e:
                print(f"[RiftCast] style_block unreadable ({_e}); using the "
                      f"Designer's own hair/wardrobe fields.", flush=True)
        dna = (f"{name} is {_an(teen_word or nationality)} {teen_word}{nationality} {noun}{eth} in "
               f"{pos} {age}, {height} with {_an(bare_build)} {bare_build} build, "
               f"with {skin} {realism}, "
               + (appearance + ". " if appearance else
                  f"{pos} {hair_color} hair {hair_style}{extra}, wearing {wardrobe}. ")
               + (
               f"{pos.capitalize()} voice is {voice}, speaking in a "
               f"{accent} accent."))

        # AUDITION ISOLATION (2026-07-30). An audition must be a FRESH ROLL.
        # Once a character is packed, its assets are installed
        # (joyecho_refs/<NAME>/ + joyecho_voices/<tag>/) and would otherwise be
        # re-cast on the next audition: RefPicker matches the name in prose and
        # injects the old face, and folder auto-cast seeds the old voice - so
        # the dropdowns would appear to do nothing. Both matchers are avoided
        # by construction:
        #   - prose refers to the subject as ID_A (RefPicker matches nothing;
        #     it strips quoted dialogue before scanning, so the name is safe
        #     to keep in the spoken line)
        #   - the speaker tag is a reserved audition tag that cannot match a
        #     voice folder
        # The REAL name still rides on dna_text/character_name for the Packer,
        # so the cartridge is written correctly.
        dna_audition = dna.replace(name, "ID_A", 1)
        tmpl = template or AUDITION_SCENE_DEFAULT
        prompt = (tmpl
                  .replace("{dialogue}", AUDITION_DIALOGUE_DEFAULT)
                  .replace("{dna}", dna_audition)
                  .replace("{demeanor}", demeanor or
                           "looks into the lens, calm and personable")
                  .replace("{name_title}", name.title())
                  .replace("{name}", "ID_A")
                  .replace("{accent}", accent)
                  .replace("{pro}", pro)
                  .replace("{pos_cap}", pos.capitalize())
                  .replace("{pos}", pos))
        script = {"speakers": [AUDITION_SPEAKER_TAG], "prompts": [prompt]}
        return (json.dumps(script, ensure_ascii=True), dna, name)


class RiftCast_Packer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "audio": ("AUDIO",),
            "character_name": ("STRING", {"forceInput": True}),
            "dna_text": ("STRING", {"forceInput": True}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            "anchor_start_sec": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0}),
            "anchor_dur_sec": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 12.0}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "pack_it"
    CATEGORY = "JoyAI-Echo/RiftCast"
    OUTPUT_NODE = True

    def pack_it(self, images, audio, character_name, dna_text, fps=24,
                anchor_start_sec=1.0, anchor_dur_sec=5.0):
        import av
        import numpy as np
        import torch
        from PIL import Image

        name = (character_name or "NOVA").strip().upper()
        frames = images
        if isinstance(frames, torch.Tensor):
            frames = (frames.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        wav = audio["waveform"]
        sr = int(audio["sample_rate"])
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu()
        if wav.dim() == 3:
            wav = wav[0]
        F = frames.shape[0]

        with tempfile.TemporaryDirectory() as td:
            for sub in ("voice", "refs", "prompts"):
                os.makedirs(os.path.join(td, sub))

            # --- voice anchor: av mux of the requested slice
            f0 = max(0, int(round(anchor_start_sec * fps)))
            f1 = min(F, int(round((anchor_start_sec + anchor_dur_sec) * fps)))
            if f1 - f0 < fps * 2:
                f0, f1 = 0, min(F, int(fps * 5))
            s0 = int(round(f0 / fps * sr))
            s1 = int(round(f1 / fps * sr))
            apath = os.path.join(td, "voice", "anchor.mp4")
            out = av.open(apath, "w")
            vs = out.add_stream("h264", rate=fps)
            vs.width = int(frames.shape[2])
            vs.height = int(frames.shape[1])
            vs.pix_fmt = "yuv420p"
            vs.options = {"crf": "18"}
            asr = out.add_stream("aac", rate=sr)
            for i in range(f0, f1):
                fr = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
                for pkt in vs.encode(fr):
                    out.mux(pkt)
            for pkt in vs.encode():
                out.mux(pkt)
            seg = wav[..., s0:s1]
            seg16 = (seg.clamp(-1, 1) * 32767).to(torch.int16).numpy()
            if seg16.ndim == 1:
                seg16 = seg16[None, :]
            af = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(seg16), format="s16p",
                layout="stereo" if seg16.shape[0] == 2 else "mono")
            af.sample_rate = sr
            for pkt in asr.encode(af):
                out.mux(pkt)
            for pkt in asr.encode():
                out.mux(pkt)
            out.close()

            # --- refs: three spread frames from inside the anchor window
            for i, t in enumerate((f0 + fps, (f0 + f1) // 2, max(f0, f1 - fps)), 1):
                Image.fromarray(frames[min(F - 1, int(t))]).save(
                    os.path.join(td, "refs", f"{name.lower()}_{i:02d}.png"))

            open(os.path.join(td, "prompts", "dna.txt"), "w",
                 encoding="ascii", errors="replace").write(dna_text.strip() + "\n")
            manifest = {"riftcast": "1.0", "name": name,
                        "speaker_tag": name.lower(),
                        "display_name": name.title(),
                        "voice": {"file": "voice/anchor.mp4"},
                        "render_law": {"video_fps": 24}}
            json.dump(manifest, open(os.path.join(td, "manifest.json"), "w"),
                      indent=1)

            out_path = os.path.join(_packs_dir(), f"{name}.riftcast")
            existed = os.path.isfile(out_path)
            pack(td, out_path)
            if existed:
                print(f"[RiftCast] NOTE: {name}.riftcast already existed and was "
                      f"REPLACED by this audition. (Auditions are isolated from "
                      f"installed cartridges, so this roll was a fresh one - but "
                      f"the previous {name} is now gone. Use a different name to "
                      f"keep both.)", flush=True)

        # materialize immediately - castable without a restart
        from .joyecho_cartridge import auto_materialize_all
        auto_materialize_all()
        report = (f"packed {name}.riftcast ({f1-f0} anchor frames, 3 refs) and "
                  f"installed - speaker tag '{name.lower()}' casts in the next "
                  f"render. Re-queue the audition with a new seed to recast; "
                  f"delete the .riftcast to retire.")
        print(f"[RiftCast] {report}", flush=True)
        return (report,)


NODE_CLASS_MAPPINGS = {
    "RiftCast_CharacterDesigner": RiftCast_CharacterDesigner,
    "RiftCast_Packer": RiftCast_Packer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RiftCast_CharacterDesigner": "RiftCast Character Designer",
    "RiftCast_Packer": "RiftCast Packer (audition -> cartridge)",
}


class RiftCast_SourceSwitch:
    """Route ONE of two prompt sources into the render chain.

    'prompt file' passes the PromptSource/LPFF script through untouched -
    your normal batch rendering. 'character designer' passes the Designer's
    audition script. Both inputs stay wired; only the selected one flows.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (["prompt file", "character designer"], {
                    "default": "prompt file",
                    "tooltip": "Which script drives this queue: your LPFF/"
                               "JSON prompt file, or the Character Designer's "
                               "audition tape."}),
            },
            "optional": {
                "file_script": ("STRING", {"forceInput": True}),
                "designer_script": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("story_idea",)
    FUNCTION = "route"
    CATEGORY = "JoyAI-Echo/RiftCast"

    def route(self, source, file_script=None, designer_script=None):
        if source == "character designer":
            if not designer_script:
                raise ValueError("source is 'character designer' but no "
                                 "Character Designer is wired in")
            return (designer_script,)
        if not file_script:
            raise ValueError("source is 'prompt file' but no PromptSource "
                             "is wired in")
        return (file_script,)


NODE_CLASS_MAPPINGS["RiftCast_SourceSwitch"] = RiftCast_SourceSwitch
NODE_DISPLAY_NAME_MAPPINGS["RiftCast_SourceSwitch"] = "RiftCast Source Switch (file / designer)"


class JoyEcho_RenderClock:
    """One source of truth for time: fps + duration in, every fps/frames
    socket in the graph fed from here. Outputs are typed for their targets
    (INT for Generate/Packer, FLOAT for CreateVideo/LLMEnhance) because a
    single primitive type-locks and cannot feed both. num_frames snaps to
    the nearest valid 8n+1."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fps": ("INT", {"default": 24, "min": 1, "max": 60, "tooltip":
                    "KEEP AT 24 for dialogue - the joint AV prior is "
                    "24fps-native; other values drift accents (25=British, "
                    "30=Australian) and override accent wording."}),
            "duration_seconds": ("FLOAT", {"default": 10.0, "min": 0.5,
                                           "max": 60.0, "step": 0.5, "tooltip":
                    "PER SHOT, not total. Every shot in the script renders "
                    "this long: a 5-shot script at 10s makes a ~50s master "
                    "(minus head-trims/transitions). Single-shot scripts and "
                    "auditions: this IS the full duration."}),
        }}

    RETURN_TYPES = ("INT", "FLOAT", "INT")
    RETURN_NAMES = ("video_fps", "fps_float", "num_frames")
    FUNCTION = "clock"
    CATEGORY = "JoyAI-Echo"

    def clock(self, fps, duration_seconds):
        raw = fps * duration_seconds
        n = max(1, round((raw - 1) / 8.0))
        frames = int(8 * n + 1)
        frames = min(frames, 1441)
        print(f"[JoyEcho] RenderClock: {fps} fps x {duration_seconds:.1f}s PER SHOT "
              f"-> {frames} frames/shot (8n+1 snapped).", flush=True)
        return (int(fps), float(fps), frames)


NODE_CLASS_MAPPINGS["JoyEcho_RenderClock"] = JoyEcho_RenderClock
NODE_DISPLAY_NAME_MAPPINGS["JoyEcho_RenderClock"] = "JoyEcho Render Clock (fps + duration -> frames)"

AUDITION_SPEAKER_TAG = "id_a"   # reserved: never a voice folder

AUDITION_SCENE_DEFAULT = (
    "Consumer mirrorless camera video, neutral color, modest dynamic "
    "range, slight sensor noise crawl in the shadows: a locked static "
    "medium shot in a small casting room, framed from the waist up with "
    "the bottom edge of frame crossing just above the hips, one "
    "continuous unbroken take that never cuts and never changes angle. "
    "The backdrop is matte gray seamless paper with a soft vertical curl "
    "at its edge, taped at the top corners, the paper catching the key as "
    "a flat diffuse expanse with no specular sheen. The light is one "
    "large soft key from the front, even and flattering, falling "
    "diffusely with no hard specular catch, the shadows soft and shallow. "
    "{dna} "
    "{name} {demeanor}. {name} brings both hands up into the bottom of "
    "the frame, fingers loosely laced, then partway through settles "
    "back, {pos} shoulders dropping "
    "as an exhale crosses {pos} face that almost becomes a smile. {name} "
    "brings one hand up to tuck a strand of hair behind {pos} ear, lets "
    "it fall, and as the take ends the easy expression drops a fraction "
    "then returns smaller and real. {name} is talking, saying in a "
    "{accent} accent, \"{dialogue}\" "
    "{pos_cap} lips move naturally in tight sync with every "
    "word. {name} is the only person in frame and "
    "{pos} voice is the only voice on the audio track. The only "
    "sounds are the room's quiet air, a faint camera-handling sound "
    "at the start, fabric shifting as {pos} shoulders settle, a single "
    "quiet creak from the seat, and {pos} voice close and clean "
    "on the mic.")

AUDITION_DIALOGUE_DEFAULT = (
    "Hi, my name's {name_title}. This is my audition tape, so, here's a "
    "little about how I sound when I'm just talking. I'll read whatever "
    "you've got, whenever you're ready.")


class RiftCast_AuditionScript:
    """Editable casting-call template - the scene and the spoken lines the
    Character Designer uses for audition tapes. Separate from your
    production prompt files; wire audition_template into the Designer's
    template input. Placeholders (safe string replacement, stray braces
    are harmless): {dna} {name} {name_title} {accent} {pro} {pos}
    {pos_cap} {dialogue}.

    Keep these rules if you rewrite it: mouth-visible close-up, ONE
    unbroken take (the Packer cuts anchor + refs from this footage),
    keep the 'saying in a {accent} accent' binding, dialogue stays
    non-falsifiable, and END ON MOTION (a settled character dead-stares)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scene_template": ("STRING", {"multiline": True,
                                          "default": AUDITION_SCENE_DEFAULT}),
            "dialogue": ("STRING", {"multiline": True,
                                    "default": AUDITION_DIALOGUE_DEFAULT}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audition_template",)
    FUNCTION = "template"
    CATEGORY = "JoyAI-Echo/RiftCast"

    def template(self, scene_template, dialogue):
        return (scene_template.replace("{dialogue}", dialogue),)


NODE_CLASS_MAPPINGS["RiftCast_AuditionScript"] = RiftCast_AuditionScript
NODE_DISPLAY_NAME_MAPPINGS["RiftCast_AuditionScript"] = "RiftCast Audition Script (casting call)"


class RiftCast_Style:
    """Style / wardrobe / hair / makeup / accessories for a designed character.

    ComfyUI has no widget folders, so this is the 'folder': the appearance
    half of the character creator lives here and feeds the Character Designer
    through its style_block input.

    A style NEVER injects its own name into the prompt - "goth" or "preppy"
    mean nothing to the model. Each entry expands into concrete renderable
    descriptors (garments with material and wear, hair shape, makeup
    placement, worn objects, and how the person carries themselves). Anything
    you set explicitly overrides what the style would have supplied.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "style": (["(none)"] + sorted(STYLES.keys()), {"tooltip":
                      "Expands into concrete wardrobe/hair/makeup/accessory "
                      "descriptors. The style's NAME is never sent to the "
                      "model. Set any field below to override its part."}),
            "hair_color": (HAIR_COLOR,),
            "hair_style": (HAIR_STYLE, {"tooltip":
                           "(from style) uses the style's hair shape."}),
            "makeup": (MAKEUP,),
            "accessories": (ACCESSORIES,),
            "demeanor": (DEMEANOR, {"tooltip":
                         "How they physically carry themselves on camera. "
                         "Feeds the audition's staging."}),
            "wardrobe": ("STRING", {"default": "(from style)", "tooltip":
                         "Freeform override. Leave as (from style) to use the "
                         "style's garments, or describe your own with "
                         "material + condition."}),
            "distinguishing": ("STRING", {"default": "", "tooltip":
                               "optional: a scar through one eyebrow, a "
                               "faded forearm tattoo, a chipped front tooth"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("style_block", "demeanor")
    FUNCTION = "build"
    CATEGORY = "JoyAI-Echo/RiftCast"

    def build(self, style, hair_color, hair_style, makeup, accessories,
              demeanor, wardrobe, distinguishing):
        s = STYLES.get(style, {})
        gendered = "{__WARDROBE__}"          # resolved by the Designer
        hair_shape = (s.get("hair", "long and loose")
                      if hair_style == "(from style)" else hair_style)
        # the style's hair clause already carries "{pos} hair ..." - strip that
        # lead-in so the colour can be inserted, and keep only the shape.
        hair_shape = re.sub(r"^\{pos\}\s+hair\s+", "", hair_shape).strip()

        mk = s.get("makeup", "") if makeup == "(from style)" else (
            "" if makeup == "(none)" else makeup)
        ac = s.get("acc", "") if accessories == "(from style)" else (
            "" if accessories == "(none)" else accessories)

        parts = [f"{{pos}} {hair_color} hair {hair_shape}",
                 f"wearing {gendered}"]
        if mk:
            parts.append(mk)
        if ac:
            parts.append(ac)
        if distinguishing.strip():
            parts.append(distinguishing.strip())
        block = ", ".join(parts)

        dem = s.get("dem", "") if demeanor == "(from style)" else demeanor
        if not dem:
            dem = "looks into the lens, calm and personable"
        # stash the style's gendered wardrobe options for the Designer
        payload = {"block": block, "has_makeup": bool(mk),
                   "wf": (wardrobe if wardrobe.strip() and
                          wardrobe.strip() != "(from style)"
                          else s.get("wf", "plain everyday clothes")),
                   "wm": (wardrobe if wardrobe.strip() and
                          wardrobe.strip() != "(from style)"
                          else s.get("wm", s.get("wf", "plain everyday clothes")))}
        return (json.dumps(payload, ensure_ascii=True), dem)


NODE_CLASS_MAPPINGS["RiftCast_Style"] = RiftCast_Style
NODE_DISPLAY_NAME_MAPPINGS["RiftCast_Style"] = "RiftCast Style + Wardrobe"
