// H3 widget persistence + a repair for the one historical layout shift.
//
// ComfyUI serializes widgets_values as a POSITIONAL array. Insert a widget in
// the middle of INPUT_TYPES and every saved value after it lands in the wrong
// slot - silently, with no warning, until something happens to be out of range
// and the queue dies on a message that names the wrong dial. That is exactly
// what "Value 4 bigger than max of 3: memory_frames" is: a workflow saved
// before v1.2 reading its old anchor_frames into memory_frames.
//
// Two parts:
//
// 1. PERSISTENCE (forward-looking). On serialize, also store {name: value} in
//    node.properties - a dict, so it is immune to ordering. On configure,
//    re-apply by name after the stock positional restore. From the first save
//    under this extension onward, no future layout change can shift anything.
//    NOTE for anyone editing workflow JSON by hand: values now live in TWO
//    places. Patch h3_widget_values as well as widgets_values, or the shadow
//    copy restores over your edit on load.
//
// 2. REPAIR (backward-looking). Persistence cannot help a workflow saved
//    before it existed, so the one shift that already shipped is repaired on
//    load. v1.2 (2026-08-05) inserted seed_per_shot at widget index 8, ahead
//    of memory_frames and anchor_frames. Discriminator: index 8 is a BOOLEAN
//    in every version from v1.2 on, and an INT (memory_frames, 0-3) in v1.0 /
//    v1.1. If it is not a boolean, splice the default back in and everything
//    downstream falls back into place.

import { app } from "../../scripts/app.js";

const PROP = "h3_widget_values";
const NODES = new Set([
    "H3MultishotMemorySampler",
    "H3MultishotSampler",
    "H3StudioSwitches",
]);

// H3StudioSwitches, 2.6.0: four flags that drove nothing in any shipped
// workflow were removed (two_pass_upscale - the feature itself left in 2.1.3;
// spectrum - the Speed Boosters node owns it; dual_clock_sampler and
// hybrid_cond - never wired), and block_cache moved to the Speed Boosters
// node. Old positional layouts:
//   2.5.1-2.5.5 (8): [two_pass, sol_attn, chunk_ffn, spectrum, block_cache,
//                     dual_clock, hybrid_cond, remote_encoder]
//   <=2.5.0     (7): same without remote_encoder
// New (3):           [sol_attn, chunk_ffn, remote_encoder]
// A positional load of an old array would put the old sol_attn value into
// chunk_ffn - so map by position ONCE here, before any widget reads it.
function repairSwitchesLayout(node) {
    const wv = node?.widgets_values;
    if (!Array.isArray(wv) || wv.length < 7) return false;
    const sol = wv[1], chunk = wv[2], remote = wv.length >= 8 ? wv[7] : false;
    node.widgets_values = [!!sol, !!chunk, !!remote];
    return true;
}

// v1.1 and earlier: [script, shot_count, width, height, frames_per_shot,
//                    seed, (seed control), steps, memory_frames, anchor_frames]
// v1.2 and later:   [..., steps, seed_per_shot, memory_frames, anchor_frames]
const SEED_PER_SHOT_INDEX = 8;

function repairLegacyLayout(node) {
    const wv = node?.widgets_values;
    if (!Array.isArray(wv) || wv.length <= SEED_PER_SHOT_INDEX) return false;
    if (typeof wv[SEED_PER_SHOT_INDEX] === "boolean") return false;  // current layout
    // A pre-v1.2 save: an INT (the old memory_frames) sits where the boolean
    // belongs. seed_per_shot defaults ON, and that is the measured recipe.
    wv.splice(SEED_PER_SHOT_INDEX, 0, true);
    return true;
}

app.registerExtension({
    name: "h3.widgetPersistence",

    beforeConfigureGraph(graphData) {
        // Runs before the nodes are built, so the splice lands before any
        // widget reads its value - and before ComfyUI range-checks it.
        let n = 0, s = 0;
        for (const node of graphData?.nodes ?? []) {
            if (node?.type === "H3StudioSwitches") { if (repairSwitchesLayout(node)) s++; continue; }
            if (NODES.has(node?.type) && repairLegacyLayout(node)) n++;
        }
        if (s) {
            console.warn(`[H3-Multishot] mapped ${s} VRAM/SPEED switches panel(s) from the pre-2.6 layout by name (four unused flags removed).`);
        }
        if (n) {
            console.warn(
                `[H3-Multishot] repaired ${n} sampler node(s) saved before ` +
                `v1.2 - seed_per_shot was inserted ahead of memory_frames in ` +
                `that release, shifting every dial after it. Save the ` +
                `workflow to make the repair permanent.`);
        }
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODES.has(nodeData?.name)) return;

        const origSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            origSerialize?.apply(this, arguments);
            if (!this.widgets?.length) return;
            const map = {};
            for (const w of this.widgets) {
                if (w.name !== undefined && w.value !== undefined) {
                    map[w.name] = w.value;
                }
            }
            o.properties = o.properties || {};
            o.properties[PROP] = map;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            origConfigure?.apply(this, arguments);
            const saved = o?.properties?.[PROP];
            if (!saved || !this.widgets?.length) return;
            for (const w of this.widgets) {
                if (!(w.name in saved)) continue;
                const v = saved[w.name];
                const opts = w.options?.values;
                if (Array.isArray(opts) && !opts.includes(v)) continue;  // renamed combo option
                w.value = v;
                w.callback?.(v);
            }
        };
    },
});
