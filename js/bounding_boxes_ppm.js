const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_CLASS = "CreateBoundingBoxesPPM";
const WIDGET_NAME = "ppm_bboxes";

const MIN_CANVAS_PX = 24;
const DEFAULT_NODE_W = 420;
const MAX_COLORS = 5;
const PICKER_W = 195;
const PICKER_SV_H = 120;
// Between the node body and the canvas: the frontend insets DOM widgets by 10px
// on every side, and the canvas wrap adds a 1px border of its own.
const NODE_PAD = 10 * 2 + 2;

const editors = new Map();

const clamp01 = (v) => Math.max(0, Math.min(1, v));

function hexToRgb(hex) {
    const h = String(hex ?? "").replace("#", "");
    if (h.length !== 6) return [140, 140, 140];
    const n = parseInt(h, 16);
    if (isNaN(n)) return [140, 140, 140];
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function readableOn(hex) {
    const [r, g, b] = hexToRgb(hex);
    return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? "#000000" : "#ffffff";
}

function hexToHsv(hex) {
    const [r, g, b] = hexToRgb(hex).map((v) => v / 255);
    const max = Math.max(r, g, b);
    const delta = max - Math.min(r, g, b);
    let h = 0;
    if (delta) {
        if (max === r) h = ((g - b) / delta) % 6;
        else if (max === g) h = (b - r) / delta + 2;
        else h = (r - g) / delta + 4;
        h = Math.round(h * 60);
        if (h < 0) h += 360;
    }
    return { h, s: max ? (delta / max) * 100 : 0, v: max * 100 };
}

function rgbToHex(r, g, b) {
    const byte = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
    return `#${byte(r)}${byte(g)}${byte(b)}`.toUpperCase();
}

function hsvToHex({ h, s, v }) {
    const sf = s / 100;
    const vf = v / 100;
    const c = vf * sf;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = vf - c;
    const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x]
        : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
    const byte = (u) => Math.round((u + m) * 255).toString(16).padStart(2, "0");
    return `#${byte(r)}${byte(g)}${byte(b)}`.toUpperCase();
}

function el(tag, css, attrs) {
    const node = document.createElement(tag);
    if (css) node.style.cssText = css;
    for (const [k, v] of Object.entries(attrs ?? {})) node.setAttribute(k, v);
    return node;
}

function options(select, pairs) {
    for (const [value, label] of pairs) {
        const option = el("option", "", { value });
        option.textContent = label;
        select.append(option);
    }
}

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// A linked widget is display-only, so writing it must not mark the graph dirty.
function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget || widget.value === value) return;
    widget.value = value;
    if (!isLinkedInput(node, name)) widget.callback?.(value);
}

function dims(node) {
    const pick = (name, fallback) => {
        const v = getWidget(node, name)?.value;
        return typeof v === "number" && v > 0 ? v : fallback;
    };
    return [pick("width", 1024), pick("height", 1024)];
}

function isLinkedInput(node, name) {
    const slot = node.findInputSlot?.(name) ?? -1;
    return slot >= 0 && node.isInputConnected?.(slot) === true;
}

// width/height can be driven by links: once linked, the widget value is stale
// (or gone) and only the size the node ran with describes the canvas.
function dimsOf(st, node) {
    const local = dims(node);
    if (!st.runDims) return local;
    const untouched = local[0] === st.widgetDims[0] && local[1] === st.widgetDims[1];
    return untouched ? st.runDims : local;
}

// The canvas box carries the aspect ratio itself (see buildUi), so it has to
// follow the dimensions the boxes are drawn against.
function setDims(st, dims) {
    st.dims = dims;
    if (st.canvasWrap) st.canvasWrap.style.aspectRatio = `${dims[0]} / ${dims[1]}`;
}

function isBox(b) {
    return !!b && typeof b === "object" &&
        typeof b.x === "number" && typeof b.y === "number" &&
        typeof b.width === "number" && typeof b.height === "number";
}

function toRegions(boxes, w, h) {
    return (Array.isArray(boxes) ? boxes : []).filter(isBox).map((b) => {
        const meta = (b.metadata && typeof b.metadata === "object") ? b.metadata : {};
        return {
            x: b.x / w,
            y: b.y / h,
            w: Math.max(0, b.width / w),
            h: Math.max(0, b.height / h),
            type: meta.type === "text" ? "text" : "obj",
            text: typeof meta.text === "string" ? meta.text : "",
            desc: typeof meta.desc === "string" ? meta.desc : "",
            palette: Array.isArray(meta.palette)
                ? meta.palette.filter((c) => typeof c === "string" && c)
                : [],
        };
    });
}

function toBoxes(regions, w, h) {
    return regions.map((r) => ({
        x: Math.round(r.x * w),
        y: Math.round(r.y * h),
        width: Math.round(r.w * w),
        height: Math.round(r.h * h),
        // The frame the pixels are drawn in: the node can run with a linked
        // width/height the editor only learns about once it has executed, and the
        // boxes have to be read in the frame they were written for.
        metadata: { type: r.type, text: r.text, desc: r.desc, palette: [...r.palette], frame: [w, h] },
    }));
}

function normalizeRegion(r) {
    let { x, y, w, h } = r;
    if (w < 0) { x += w; w = -w; }
    if (h < 0) { y += h; h = -h; }
    x = clamp01(x);
    y = clamp01(y);
    return { ...r, x, y, w: Math.max(0, Math.min(w, 1 - x)), h: Math.max(0, Math.min(h, 1 - y)) };
}

function applyDrag(mode, start, dx, dy) {
    let { x, y, w, h } = start;
    switch (mode) {
        case "move":
            return { ...start, x: clamp01(Math.min(x + dx, 1 - w)), y: clamp01(Math.min(y + dy, 1 - h)) };
        case "draw":
        case "resize-br":
            w += dx; h += dy; break;
        case "resize-tl":
            x += dx; y += dy; w -= dx; h -= dy; break;
        case "resize-tr":
            y += dy; w += dx; h -= dy; break;
        case "resize-bl":
            x += dx; w -= dx; h += dy; break;
        case "resize-t":
            y += dy; h -= dy; break;
        case "resize-b":
            h += dy; break;
        case "resize-l":
            x += dx; w -= dx; break;
        case "resize-r":
            w += dx; break;
    }
    return normalizeRegion({ ...start, x, y, w, h });
}

function hitRegions(regions, px, py, logW, logH, active) {
    const handlePx = 8;
    const rx = handlePx / Math.max(1, logW);
    const ry = handlePx / Math.max(1, logH);
    const hits = [];
    for (let i = 0; i < regions.length; i++) {
        const b = regions[i];
        const x1 = b.x, y1 = b.y, x2 = b.x + b.w, y2 = b.y + b.h;
        const near = (cx, cy) => Math.abs(px - cx) < rx && Math.abs(py - cy) < ry;
        let mode = null;
        if (near(x1, y1)) mode = "resize-tl";
        else if (near(x2, y1)) mode = "resize-tr";
        else if (near(x1, y2)) mode = "resize-bl";
        else if (near(x2, y2)) mode = "resize-br";
        else if (px >= x1 && px <= x2 && Math.abs(py - y1) < ry) mode = "resize-t";
        else if (px >= x1 && px <= x2 && Math.abs(py - y2) < ry) mode = "resize-b";
        else if (py >= y1 && py <= y2 && Math.abs(px - x1) < rx) mode = "resize-l";
        else if (py >= y1 && py <= y2 && Math.abs(px - x2) < rx) mode = "resize-r";
        else if (px >= x1 && px <= x2 && py >= y1 && py <= y2) mode = "move";
        if (mode) hits.push({ index: i, mode });
    }
    const ai = hits.findIndex((h) => h.index === active);
    if (ai > 0) hits.unshift(hits.splice(ai, 1)[0]);
    return hits;
}

function gridSteps(st) {
    const [w, h] = st.dims;
    // Cells never shrink below 32px and no axis gets more than 64 of them.
    const axis = (size) => Math.max(32, Math.ceil(size / 64)) / size;
    return { fx: axis(w), fy: axis(h) };
}

function snapRegion(st, region, mode) {
    if (!st.grid) return region;
    const { fx, fy } = gridSteps(st);
    const snap = (v, step) => clamp01(Math.round(v / step) * step);
    if (mode === "move") {
        return {
            ...region,
            x: Math.min(snap(region.x, fx), 1 - region.w),
            y: Math.min(snap(region.y, fy), 1 - region.h),
        };
    }
    const left = mode === "draw" || mode === "resize-l" || mode === "resize-tl" || mode === "resize-bl";
    const right = mode === "draw" || mode === "resize-r" || mode === "resize-tr" || mode === "resize-br";
    const top = mode === "draw" || mode === "resize-t" || mode === "resize-tl" || mode === "resize-tr";
    const bottom = mode === "draw" || mode === "resize-b" || mode === "resize-bl" || mode === "resize-br";
    const x1 = left ? snap(region.x, fx) : region.x;
    const y1 = top ? snap(region.y, fy) : region.y;
    const x2 = right ? snap(region.x + region.w, fx) : region.x + region.w;
    const y2 = bottom ? snap(region.y + region.h, fy) : region.y + region.h;
    return { ...region, x: x1, y: y1, w: Math.max(0, x2 - x1), h: Math.max(0, y2 - y1) };
}

function wrapLines(ctx, text, maxW) {
    const out = [];
    for (const para of String(text).split("\n")) {
        let line = "";
        for (const word of para.split(/\s+/)) {
            if (!word) continue;
            const test = line ? `${line} ${word}` : word;
            if (line && ctx.measureText(test).width > maxW) {
                out.push(line);
                line = word;
            } else {
                line = test;
            }
        }
        if (line) out.push(line);
    }
    return out;
}

function pointerNorm(st, e) {
    const rect = st.canvas.getBoundingClientRect();
    return {
        x: clamp01((e.clientX - rect.left) / Math.max(1, rect.width)),
        y: clamp01((e.clientY - rect.top) / Math.max(1, rect.height)),
    };
}

function requestDraw(st) {
    if (st.raf) return;
    st.raf = requestAnimationFrame(() => {
        st.raf = 0;
        draw(st);
    });
}

// The box being edited is drawn last so nothing covers its fill, outline or tag.
function drawOrder(st) {
    const order = [...st.regions.keys()];
    if (st.active >= 0) order.push(order.splice(st.active, 1)[0]);
    return order;
}

function tagAt(st, x, y) {
    const order = drawOrder(st);
    for (let n = order.length - 1; n >= 0; n--) {
        const i = order[n];
        const r = st.tagRects[i];
        if (r && x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return i;
    }
    return -1;
}

// A tag belongs to its own box whatever is drawn over it, so it wins over
// hit-testing and drags the box it labels.
function hitAt(st, p, cycle) {
    const tag = tagAt(st, p.x, p.y);
    return tag >= 0 ? { index: tag, mode: "move" } : pick(st, p, cycle);
}

function draw(st) {
    const canvas = st.canvas;
    const wrap = st.canvasWrap;
    const W = Math.max(1, wrap.clientWidth);
    const H = Math.max(1, wrap.clientHeight);
    const tagRects = [];
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.max(1, Math.round(W * dpr));
    const bh = Math.max(1, Math.round(H * dpr));
    if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    if (st.bg) {
        ctx.drawImage(st.bg, 0, 0, W, H);
        ctx.fillStyle = "rgba(0,0,0,0.75)";
        ctx.fillRect(0, 0, W, H);
    }

    if (st.grid) {
        const { fx, fy } = gridSteps(st);
        ctx.save();
        ctx.globalAlpha = 0.18;
        ctx.fillStyle = "#ffffff";
        for (let i = 0; i <= Math.round(1 / fx); i++) {
            for (let j = 0; j <= Math.round(1 / fy); j++) {
                ctx.fillRect(Math.min(1, i * fx) * W - 0.5, Math.min(1, j * fy) * H - 0.5, 1, 1);
            }
        }
        ctx.restore();
    }

    ctx.font = "bold 11px monospace";
    for (const i of drawOrder(st)) {
        const b = st.regions[i];
        const active = i === st.active;
        const color = b.palette.find(Boolean) || "#8c8c8c";
        const x1 = b.x * W, y1 = b.y * H;
        const w = b.w * W, h = b.h * H;
        const hovered = active || i === st.hover;

        if (active) {
            ctx.fillStyle = "rgba(26,26,26,0.88)";
            ctx.fillRect(x1, y1, w, h);
        }
        const [r, g, bl] = hexToRgb(color);
        ctx.fillStyle = `rgba(${r},${g},${bl},${hovered ? 0.23 : 0.13})`;
        ctx.fillRect(x1, y1, w, h);

        const lw = active ? 2 : hovered ? 1.5 : 1;
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.strokeRect(x1 + lw / 2, y1 + lw / 2, Math.max(0, w - lw), Math.max(0, h - lw));

        if (b.palette.length && w > 2) {
            const seg = w / b.palette.length;
            for (let p = 0; p < b.palette.length; p++) {
                ctx.fillStyle = b.palette[p];
                ctx.fillRect(x1 + Math.round(p * seg), y1, Math.round(seg), 7);
            }
        }

        ctx.save();
        ctx.beginPath();
        ctx.rect(x1, y1, w, h);
        ctx.clip();

        let body = b.desc || "";
        if (b.type === "text" && b.text) body = `"${b.text}"` + (body ? ` — ${body}` : "");
        if (body && w > 8) {
            ctx.font = "12px monospace";
            ctx.fillStyle = readableOn(color);
            let ty = y1 + 27;
            for (const line of wrapLines(ctx, body, w - 8)) {
                if (ty > y1 + h) break;
                ctx.fillText(line, x1 + 4, ty);
                ty += 14;
            }
        }

        const tag = String(i + 1).padStart(2, "0");
        ctx.font = "bold 11px monospace";
        const tw = ctx.measureText(tag).width + 8;
        tagRects[i] = { x: x1 / W, y: y1 / H, w: tw / W, h: 14 / H };
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1, tw, 14);
        if (i === st.hoverTag) {
            ctx.fillStyle = "rgba(255,255,255,0.25)";
            ctx.fillRect(x1, y1, tw, 14);
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1;
            ctx.strokeRect(x1 + 0.5, y1 + 0.5, tw - 1, 13);
        }
        ctx.fillStyle = readableOn(color);
        ctx.fillText(tag, x1 + 4, y1 + 11);
        ctx.restore();
    }
    st.tagRects = tagRects;
}

function sync(st, node) {
    const target = getWidget(node, "editor_state");
    if (!target) return;
    const [w, h] = st.dims;
    st.loadedFrom = toBoxes(st.regions, w, h);
    target.value = st.loadedFrom;
    // Reading the value back is not guaranteed to hand us the same object (Vue
    // wraps node state in reactive proxies), so remember when the write was ours.
    st.syncedAt = performance.now();
    if (node.graph) node.graph.incrementVersion();
    node.setDirtyCanvas(true, false);
    requestDraw(st);
}

function commit(st, node) {
    renderInspector(st, node);
    sync(st, node);
}

function readState(st, node) {
    const target = getWidget(node, "editor_state");
    st.loadedFrom = target?.value;
    setDims(st, dimsOf(st, node));
    st.regions = toRegions(Array.isArray(st.loadedFrom) ? st.loadedFrom : [], st.dims[0], st.dims[1]);
    if (st.active >= st.regions.length) st.active = st.regions.length ? 0 : -1;
    renderInspector(st, node);
    requestDraw(st);
}

function removeRegion(st, node, i) {
    st.regions.splice(i, 1);
    if (!st.regions.length) st.active = -1;
    else if (i <= st.active) st.active = Math.max(0, st.active - 1);
    commit(st, node);
}

function activeRegion(st) {
    return st.active >= 0 ? st.regions[st.active] : undefined;
}

// Toolbar, inspector and hint are content-sized, so measuring them is exact and
// independent of the node height. Deriving the chrome from the node's own height
// makes the reported widget height change from pass to pass, and the frontend
// grows the node towards it once per frame: the node creeps until both agree.
function chromeHeight(st) {
    const ROW_GAP = 4;
    // Before the editor is mounted nothing has a height yet, so fall back to the
    // height each row settles on.
    let chrome = (st.toolbar?.offsetHeight || 24) + ROW_GAP + NODE_PAD;
    if (st.inspector?.style.display !== "none") chrome += (st.inspector?.offsetHeight || 110) + ROW_GAP;
    if (st.hint?.style.display !== "none") chrome += (st.hint?.offsetHeight || 14) + ROW_GAP;
    return chrome;
}

// Reporting the same value as min and max height pins the canvas to its own
// aspect-ratio box (like the stock editor), so stretching the node vertically
// leaves dead space instead of distorting the boxes.
function widgetHeight(st, node) {
    // Measured so the node is sized for what the layout really renders; the maths
    // only covers the pre-mount pass, where nothing has a size yet.
    if (st.canvasWrap?.clientHeight) return st.canvasWrap.clientHeight + chromeHeight(st);
    const [dw, dh] = st.dims;
    const box = Math.max(1, (st.widget?.width ?? node.width ?? DEFAULT_NODE_W) - NODE_PAD);
    return Math.max(MIN_CANVAS_PX, Math.round((box * dh) / Math.max(1, dw))) + chromeHeight(st);
}

function fitNode(st, node) {
    if (!node.computeSize || !node.setSize) return;
    const height = node.computeSize()[1];
    if (Math.abs(height - node.size[1]) > 1) node.setSize([node.size[0], height]);
}

function renderInspector(st, node) {
    const b = activeRegion(st);
    st.inspector.style.display = b ? "flex" : "none";
    st.hint.style.display = b || !st.regions.length ? "none" : "block";
    st.textField.style.display = b?.type === "text" ? "flex" : "none";
    if (b) {
        if (st.typeSelect.value !== b.type) st.typeSelect.value = b.type;
        if (document.activeElement !== st.textInput) st.textInput.value = b.text;
        if (document.activeElement !== st.descInput) st.descInput.value = b.desc;

        const swatches = b.palette.map((color, i) => makeSwatch(st, node, i, color));
        st.palette.replaceChildren(...swatches, st.paletteAdd);
        st.paletteAdd.style.display = b.palette.length < MAX_COLORS ? "" : "none";
        closePicker(st);
    } else {
        closePicker(st, true);
    }
    // Size only once every row is in its final state, otherwise the reported
    // height is one toggle behind and has to be corrected again next frame.
    fitNode(st, node);
}

function makeSwatch(st, node, index, hex) {
    const swatch = el("button", "width:20px;height:20px;padding:0;border:1px solid #555;border-radius:3px;" +
        "cursor:pointer;touch-action:none;", {
        type: "button",
        title: "Click edit · drag reorder · right-click remove",
        "data-index": String(index),
    });
    swatch.style.background = hex;
    swatch.addEventListener("click", () => {
        if (performance.now() < st.swatchMovedAt) return;
        if (st.picker?.anchor === swatch) {
            closePicker(st, true);
            return;
        }
        openPicker(st, node, index, swatch);
    });
    swatch.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const b = activeRegion(st);
        if (!b) return;
        b.palette.splice(index, 1);
        commit(st, node);
    });
    swatch.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        st.swatchDrag = { index, x: e.clientX, y: e.clientY, moved: false };
    });
    return swatch;
}

function swatchUnder(st, cx, cy) {
    for (const other of st.palette.querySelectorAll("[data-index]")) {
        const r = other.getBoundingClientRect();
        if (cx < r.left || cx > r.right || cy < r.top - 6 || cy > r.bottom + 6) continue;
        return Number(other.dataset.index);
    }
    return null;
}

// Drag a swatch onto another one to reorder the palette, like the stock editor.
function bindSwatchDrag(st, node) {
    document.addEventListener("pointermove", (e) => {
        const d = st.swatchDrag;
        if (!d) return;
        if ((e.buttons & 1) === 0) {
            st.swatchDrag = null;
            return;
        }
        if (!d.moved && Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) < 4) return;
        d.moved = true;
        const to = swatchUnder(st, e.clientX, e.clientY);
        st.swatchDrag = null;
        if (to === null || to === d.index) return;
        const b = activeRegion(st);
        if (!b) return;
        const next = b.palette.slice();
        const [moved] = next.splice(d.index, 1);
        next.splice(to, 0, moved);
        b.palette = next;
        st.swatchMovedAt = performance.now() + 250;
        commit(st, node);
    });
    document.addEventListener("pointerup", () => { st.swatchDrag = null; });
    document.addEventListener("pointercancel", () => { st.swatchDrag = null; });
}

function drawPicker(st) {
    const p = st.picker;
    if (!p) return;
    const dpr = window.devicePixelRatio || 1;

    p.sv.width = PICKER_W * dpr;
    p.sv.height = PICKER_SV_H * dpr;
    const sv = p.sv.getContext("2d");
    sv.setTransform(dpr, 0, 0, dpr, 0, 0);
    sv.fillStyle = hsvToHex({ h: p.hsv.h, s: 100, v: 100 });
    sv.fillRect(0, 0, PICKER_W, PICKER_SV_H);
    let grad = sv.createLinearGradient(0, 0, PICKER_W, 0);
    grad.addColorStop(0, "#ffffff");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    sv.fillStyle = grad;
    sv.fillRect(0, 0, PICKER_W, PICKER_SV_H);
    grad = sv.createLinearGradient(0, 0, 0, PICKER_SV_H);
    grad.addColorStop(0, "rgba(0,0,0,0)");
    grad.addColorStop(1, "#000000");
    sv.fillStyle = grad;
    sv.fillRect(0, 0, PICKER_W, PICKER_SV_H);
    sv.beginPath();
    sv.arc((p.hsv.s / 100) * PICKER_W, (1 - p.hsv.v / 100) * PICKER_SV_H, 5, 0, Math.PI * 2);
    sv.strokeStyle = "#ffffff";
    sv.lineWidth = 1.5;
    sv.stroke();

    p.hue.width = PICKER_W * dpr;
    p.hue.height = 12 * dpr;
    const hu = p.hue.getContext("2d");
    hu.setTransform(dpr, 0, 0, dpr, 0, 0);
    const hg = hu.createLinearGradient(0, 0, PICKER_W, 0);
    for (let i = 0; i <= 6; i++) hg.addColorStop(i / 6, hsvToHex({ h: (i * 60) % 360, s: 100, v: 100 }));
    hu.fillStyle = hg;
    hu.fillRect(0, 0, PICKER_W, 12);
    hu.fillStyle = "#ffffff";
    hu.fillRect((p.hsv.h / 360) * PICKER_W - 1.5, 0, 3, 12);
}

function positionPicker(st) {
    const p = st.picker;
    if (!p) return;
    // Leaving classic rendering (Nodes 2.0, subgraph change) detaches the widget,
    // which would otherwise strand the panel at 0,0.
    if (!p.anchor.isConnected || !st.root?.isConnected) {
        closePicker(st, true);
        return;
    }
    const r = p.anchor.getBoundingClientRect();
    const h = p.pop.offsetHeight || 190;
    p.pop.style.left = `${Math.max(4, Math.min(window.innerWidth - PICKER_W - 22, r.left))}px`;
    p.pop.style.top = `${r.bottom + 6 + h > window.innerHeight - 4 ? Math.max(4, r.top - h - 6) : r.bottom + 6}px`;
}

// Keep the picker open across palette rebuilds caused by its own edits; only a
// different selection or a removed colour closes it.
function closePicker(st, force = false) {
    const p = st.picker;
    if (!p) return;
    if (!force && p.active === st.active) {
        const swatch = st.palette?.querySelector(`[data-index="${p.index}"]`);
        if (swatch) {
            p.anchor = swatch;
            positionPicker(st);
            return;
        }
    }
    document.removeEventListener("pointerdown", p.onDocPointerDown, true);
    document.removeEventListener("keydown", p.onKeyDown, true);
    window.removeEventListener("blur", p.onBlur);
    clearInterval(st.pickerTimer);
    st.pickerTimer = 0;
    p.pop.remove();
    st.picker = null;
}

function refreshPickerFields(st, skip) {
    const p = st.picker;
    if (!p) return;
    const hex = hsvToHex(p.hsv);
    if (p.hexInput !== skip && document.activeElement !== p.hexInput) p.hexInput.value = hex;
    const [r, g, b] = hexToRgb(hex);
    p.rgbInputs.forEach((input, i) => {
        if (input === skip || document.activeElement === input) return;
        input.value = String([r, g, b][i]);
    });
}

function applyPicker(st, node, typing) {
    const p = st.picker;
    if (!p) return;
    const b = activeRegion(st);
    if (!b || !b.palette[p.index]) {
        closePicker(st, true);
        return;
    }
    b.palette[p.index] = hsvToHex(p.hsv);
    if (!typing) refreshPickerFields(st);
    const swatch = st.palette.querySelector(`[data-index="${p.index}"]`);
    if (swatch) swatch.style.background = b.palette[p.index];
    drawPicker(st);
    sync(st, node);
}

function bindPickerArea(elm, onPoint) {
    elm.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        const at = (ev) => {
            const r = elm.getBoundingClientRect();
            onPoint(
                clamp01((ev.clientX - r.left) / Math.max(1, r.width)),
                clamp01((ev.clientY - r.top) / Math.max(1, r.height)),
            );
        };
        at(e);
        // Document-level tracking: dragging keeps working when the pointer
        // leaves the strip, which element capture does not guarantee here.
        const move = (ev) => (((ev.buttons & 1) === 0) ? stop() : at(ev));
        const stop = () => {
            document.removeEventListener("pointermove", move);
            document.removeEventListener("pointerup", stop);
            document.removeEventListener("pointercancel", stop);
        };
        document.addEventListener("pointermove", move);
        document.addEventListener("pointerup", stop);
        document.addEventListener("pointercancel", stop);
    });
}

// Spin-button styling needs a stylesheet; inline styles cannot reach the pseudo-elements.
function ensurePickerStyles() {
    if (document.getElementById("ppm-bbox-picker-style")) return;
    const style = el("style", "", { id: "ppm-bbox-picker-style" });
    style.textContent = ".ppm-bbox-num{-moz-appearance:textfield;appearance:textfield;}" +
        ".ppm-bbox-num::-webkit-outer-spin-button,.ppm-bbox-num::-webkit-inner-spin-button{" +
        "-webkit-appearance:none;margin:0;}";
    document.head.append(style);
}

function applyPickerMode(st) {
    const p = st.picker;
    if (!p) return;
    const rgba = p.modeSelect.value === "rgba";
    p.rgbRow.style.display = rgba ? "flex" : "none";
    p.hexInput.style.display = rgba ? "none" : "block";
}

function openPicker(st, node, index, anchor) {
    const b = activeRegion(st);
    if (!b || !b.palette[index]) return;
    closePicker(st, true);
    ensurePickerStyles();

    const pop = el("div", "position:fixed;z-index:9999;box-sizing:border-box;display:flex;flex-direction:column;" +
        "gap:6px;width:211px;padding:8px;border:1px solid #4a4a4a;border-radius:8px;background:#1b1b1b;" +
        "box-shadow:0 8px 28px rgba(0,0,0,0.55);color:#ddd;font:11px monospace;");
    const sv = el("canvas", `width:${PICKER_W}px;height:${PICKER_SV_H}px;display:block;border-radius:4px;` +
        "cursor:crosshair;touch-action:none;");
    const hue = el("canvas", `width:${PICKER_W}px;height:12px;display:block;border-radius:4px;` +
        "cursor:pointer;touch-action:none;");
    const hexInput = el("input", FIELD + "flex:1;min-width:0;text-align:center;", {
        spellcheck: "false",
        "aria-label": "hex color",
        title: "hex color",
    });
    const rgbInputs = ["R", "G", "B"].map((label) => el("input", FIELD + "flex:1;min-width:0;padding:3px 2px;text-align:center;", {
        type: "number",
        min: "0",
        max: "255",
        class: "ppm-bbox-num",
        "aria-label": `${label} channel`,
        title: `${label} channel`,
    }));
    const rgbRow = el("div", "display:none;gap:4px;flex:1;min-width:0;");
    rgbRow.append(...rgbInputs);
    const modeSelect = el("select", FIELD + "flex:0 0 auto;width:62px;padding:3px 2px;", { title: "Color format" });
    options(modeSelect, [["hex", "Hex"], ["rgba", "RGBA"]]);
    const fieldRow = el("div", "display:flex;gap:6px;align-items:center;");
    fieldRow.append(modeSelect, hexInput, rgbRow);
    pop.append(sv, hue, fieldRow);
    document.body.append(pop);

    st.picker = {
        pop, sv, hue, hexInput, rgbInputs, rgbRow, modeSelect, anchor,
        index, active: st.active, hsv: hexToHsv(b.palette[index]),
    };
    const p = st.picker;
    p.onDocPointerDown = (e) => {
        if (!p.pop.contains(e.target) && e.target !== p.anchor) closePicker(st, true);
    };
    p.onKeyDown = (e) => { if (e.key === "Escape") closePicker(st, true); };
    p.onBlur = () => closePicker(st, true);
    document.addEventListener("pointerdown", p.onDocPointerDown, true);
    document.addEventListener("keydown", p.onKeyDown, true);
    window.addEventListener("blur", p.onBlur);

    bindPickerArea(sv, (x, y) => {
        p.hsv.s = x * 100;
        p.hsv.v = (1 - y) * 100;
        applyPicker(st, node);
    });
    bindPickerArea(hue, (x) => {
        p.hsv.h = Math.min(359, Math.round(x * 360));
        applyPicker(st, node);
    });
    hexInput.addEventListener("input", () => {
        const hex = hexInput.value.trim().replace(/^#?/, "#");
        if (!/^#[0-9a-fA-F]{6}$/.test(hex)) return;
        p.hsv = hexToHsv(hex);
        applyPicker(st, node, true);
    });
    rgbInputs.forEach((input) => input.addEventListener("input", () => {
        if (rgbInputs.some((x) => x.value.trim() === "")) return;
        const values = rgbInputs.map((x) => {
            const n = Math.round(Number(x.value));
            return Number.isFinite(n) ? Math.max(0, Math.min(255, n)) : 0;
        });
        p.hsv = hexToHsv(rgbToHex(...values));
        applyPicker(st, node, true);
    }));
    modeSelect.addEventListener("change", () => {
        st.colorMode = modeSelect.value;
        applyPickerMode(st);
        refreshPickerFields(st);
    });
    const resetFields = () => refreshPickerFields(st);
    hexInput.addEventListener("blur", resetFields);
    rgbInputs.forEach((input) => input.addEventListener("blur", resetFields));

    refreshPickerFields(st);
    modeSelect.value = st.colorMode;
    applyPickerMode(st);
    drawPicker(st);
    positionPicker(st);
    // The panel lives outside the widget, so poll while it is open: unmounting
    // the widget does not run any of our other callbacks.
    st.pickerTimer = setInterval(() => positionPicker(st), 250);
}

function loadBackground(st, node, imageInfo) {
    const params = new URLSearchParams({
        filename: imageInfo.filename,
        subfolder: imageInfo.subfolder ?? "",
        type: imageInfo.type ?? "temp",
    });
    const img = new Image();
    img.onload = () => {
        st.bg = img;
        const snap = (v) => Math.max(16, Math.round(v / 16) * 16);
        setWidgetValue(node, "width", snap(img.naturalWidth));
        setWidgetValue(node, "height", snap(img.naturalHeight));
        setDims(st, dimsOf(st, node));
        sync(st, node);
    };
    img.src = api.apiURL(`/view?${params}`);
}

// The saved boxes are pixel coordinates, so re-read them in the new frame.
function applyDims(st, node, dimsMessage) {
    const [w, h] = Array.isArray(dimsMessage) ? dimsMessage : [];
    if (typeof w !== "number" || typeof h !== "number" || w <= 0 || h <= 0) return;
    if (st.dragging) return;
    st.widgetDims = dims(node);
    st.runDims = [w, h];
    setWidgetValue(node, "width", w);
    setWidgetValue(node, "height", h);
    setDims(st, dimsOf(st, node));
    sync(st, node);
    fitNode(st, node);
}

function applyExecuted(st, node, message) {
    if (!message) return;

    applyDims(st, node, message.dims);

    const incoming = message.input_bboxes;
    if (Array.isArray(incoming) && incoming.length) {
        const last = getWidget(node, "last_incoming");
        const applied = Array.isArray(last?.value) ? last.value : [];
        if (JSON.stringify(incoming) !== JSON.stringify(applied)) {
            setDims(st, dimsOf(st, node));
            st.regions = toRegions(incoming, st.dims[0], st.dims[1]);
            st.active = st.regions.length ? 0 : -1;
            if (last) {
                last.value = incoming;
                if (node.graph) node.graph.incrementVersion();
            }
            commit(st, node);
        }
    }

    const background = message.background_images?.[0];
    if (background?.filename) loadBackground(st, node, background);
}

// A connected widget keeps its canvas row whatever the tier says: the node size
// counts it while the widget layout skips it, so hiding one prints the dimmed
// label over whatever is drawn there and leaves the row as dead space. Leave
// those to the frontend, which dims the row instead. Every write moves a whole
// row in or out of the layout, so the node has to be sized again.
function setCanvasSurface(st, node, name, tier) {
    const surfaces = getWidget(node, name)?.visibility?.surfaces;
    if (!surfaces) return;
    const canvas = tier === "never" && isLinkedInput(node, name) ? "shown" : tier;
    if (surfaces.canvas === canvas) return;
    surfaces.canvas = canvas;
    st.needsFit = true;
}

// The Vue editor renders for every BOUNDING_BOXES widget, and the frontend only
// hides the internal ones on the core CreateBoundingBoxes node.
function hideInternalWidgets(st, node) {
    for (const widget of node.widgets ?? []) {
        if (widget.name === "last_incoming") widget.hidden = true;
    }
    setCanvasSurface(st, node, "editor_state", "never");
}

function updateSurfaces(st, node) {
    const fromImage = isLinkedInput(node, "background");
    // Nothing replaces the image once the background input is gone.
    if (st.bg && !fromImage) {
        st.bg = null;
        requestDraw(st);
    }
    setCanvasSurface(st, node, "editor_state", "never");
    for (const name of ["width", "height"]) {
        setCanvasSurface(st, node, name, fromImage ? "never" : "shown");
    }
}

function maintain(st, node) {
    updateSurfaces(st, node);
    positionPicker(st);
    const target = getWidget(node, "editor_state");
    const ours = st.dragging || performance.now() - st.syncedAt < 300;
    if (!ours && target && target.value !== st.loadedFrom) readState(st, node);
    const [w, h] = dimsOf(st, node);
    if (w !== st.dims[0] || h !== st.dims[1]) {
        setDims(st, [w, h]);
        fitNode(st, node);
        // editor_state holds pixels for the current frame, so re-emit them (like
        // the stock editor's watch on width/height); re-reading them against the
        // new frame would squish every box.
        sync(st, node);
    }
    // The size set while attaching uses static chrome estimates because the editor
    // is not in the document yet, and a link change adds or removes widget rows:
    // correct the node on the next draw, once the layout has settled.
    if (st.needsFit && st.root?.isConnected) {
        st.needsFit = false;
        fitNode(st, node);
        // Nothing else resizes the node here, and setSize does not ask for a redraw.
        node.setDirtyCanvas(true, true);
    }
}

const TOOLBAR_BTN = "flex:0 0 auto;padding:2px 8px;border:1px solid #4a4a4a;border-radius:4px;" +
    "background:#2a2a2a;color:#ddd;font:11px monospace;cursor:pointer;";
const FIELD = "box-sizing:border-box;width:100%;padding:3px 4px;border:1px solid #4a4a4a;border-radius:4px;" +
    "background:#111;color:#eee;font:11px monospace;resize:none;";

function buildUi(st, node) {
    st.root = el("div", "box-sizing:border-box;display:flex;flex-direction:column;gap:4px;width:100%;height:100%;" +
        "overflow:hidden;color:#ddd;font:11px monospace;user-select:none;");
    st.root.className = "ppm-bbox-editor";

    st.toolbar = el("div", "flex:0 0 auto;display:flex;gap:4px;align-items:center;");
    st.gridButton = el("button", TOOLBAR_BTN, { type: "button", title: "Toggle grid snapping" });
    st.gridButton.textContent = "Grid";
    st.gridButton.addEventListener("click", () => {
        st.grid = !st.grid;
        node.setProperty?.("bboxGrid", st.grid);
        paintToolbar(st);
        requestDraw(st);
    });
    const clearButton = el("button", TOOLBAR_BTN + "margin-left:auto;", { type: "button", title: "Remove all boxes" });
    clearButton.textContent = "Clear all";
    clearButton.addEventListener("click", () => {
        st.regions = [];
        st.active = -1;
        const last = getWidget(node, "last_incoming");
        if (last) last.value = [];
        commit(st, node);
    });
    st.toolbar.append(st.gridButton, clearButton);

    // The canvas box carries its own aspect ratio (like the stock editor), so a
    // node that is briefly too short clips its bottom rows instead of squashing
    // the canvas: the element height only catches up with the node on the next
    // draw, and flex would otherwise absorb that missing space here.
    st.canvasWrap = el("div", "position:relative;flex:0 0 auto;box-sizing:content-box;overflow:hidden;" +
        `aspect-ratio:1/1;min-height:${MIN_CANVAS_PX}px;` +
        "border:1px solid #4a4a4a;border-radius:4px;background:#181818;");
    st.canvas = el("canvas", "position:absolute;inset:0;width:100%;height:100%;display:block;cursor:crosshair;outline:none;", {
        tabindex: "0",
    });
    st.canvasWrap.append(st.canvas);

    st.inspector = el("div", "flex:0 0 auto;flex-direction:column;gap:4px;padding:4px;border-radius:4px;background:#202020;display:none;");
    st.typeSelect = el("select", FIELD, { title: "Box type" });
    options(st.typeSelect, [["obj", "Object"], ["text", "Text"]]);
    st.typeSelect.addEventListener("change", () => {
        const b = activeRegion(st);
        if (!b) return;
        b.type = st.typeSelect.value;
        commit(st, node);
    });

    st.textField = el("div", "flex-direction:column;gap:2px;");
    st.textInput = el("textarea", FIELD + "height:34px;", { placeholder: "Text to render" });
    st.textInput.addEventListener("input", () => {
        const b = activeRegion(st);
        if (!b) return;
        b.text = st.textInput.value;
        sync(st, node);
    });
    st.textField.append(st.textInput);

    st.descInput = el("textarea", FIELD + "height:44px;", { placeholder: "Description" });
    st.descInput.addEventListener("input", () => {
        const b = activeRegion(st);
        if (!b) return;
        b.desc = st.descInput.value;
        sync(st, node);
    });

    st.palette = el("div", "display:flex;gap:4px;align-items:center;");
    st.paletteAdd = el("button", TOOLBAR_BTN, { type: "button", title: "Add a color" });
    st.paletteAdd.textContent = "+";
    st.paletteAdd.addEventListener("click", () => {
        const b = activeRegion(st);
        if (!b || b.palette.length >= MAX_COLORS) return;
        b.palette.push("#ffffff");
        commit(st, node);
    });
    st.palette.append(st.paletteAdd);

    st.inspector.append(st.typeSelect, st.textField, st.descInput, st.palette);
    st.hint = el("div", "flex:0 0 auto;padding:0 2px;color:#9a9a9a;display:none;", { title: "Select a box to edit it" });
    st.hint.textContent = "Click a region to edit it.";
    st.root.append(st.toolbar, st.canvasWrap, st.inspector, st.hint);

    paintToolbar(st);
    bindCanvas(st, node);
    bindSwatchDrag(st, node);
    new ResizeObserver(() => requestDraw(st)).observe(st.canvasWrap);
}

function paintToolbar(st) {
    st.gridButton.style.background = st.grid ? "#3d6a86" : "#2a2a2a";
}

function pick(st, p, cycle) {
    const hits = hitRegions(st.regions, p.x, p.y, st.canvasWrap.clientWidth, st.canvasWrap.clientHeight, st.active);
    if (!hits.length) return null;
    if (cycle && hits.length > 1) {
        const pos = hits.findIndex((h) => h.index === st.active);
        return hits[(pos + 1) % hits.length];
    }
    return hits.find((h) => h.index === st.active && h.mode !== "move") || hits[0];
}

function bindCanvas(st, node) {
    const canvas = st.canvas;

    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.stopPropagation();
        canvas.focus();
        const p = pointerNorm(st, e);
        const hit = hitAt(st, p, e.altKey);
        if (hit) {
            st.active = hit.index;
            st.drag = { mode: hit.mode, start: { ...st.regions[hit.index] }, origin: p };
        } else {
            const region = { x: p.x, y: p.y, w: 0, h: 0, type: "obj", text: "", desc: "", palette: [] };
            st.regions.push(region);
            st.active = st.regions.length - 1;
            st.drag = { mode: "draw", start: { ...region }, origin: p };
        }
        st.dragging = true;
        canvas.setPointerCapture(e.pointerId);
        renderInspector(st, node);
        requestDraw(st);
    });

    canvas.addEventListener("pointermove", (e) => {
        const p = pointerNorm(st, e);
        if (st.drag) {
            const region = applyDrag(st.drag.mode, st.drag.start, p.x - st.drag.origin.x, p.y - st.drag.origin.y);
            st.regions[st.active] = snapRegion(st, region, st.drag.mode);
            requestDraw(st);
            return;
        }
        const hoverTag = tagAt(st, p.x, p.y);
        const hover = hoverTag >= 0 ? hoverTag : pick(st, p, false)?.index ?? -1;
        st.canvas.style.cursor = hoverTag >= 0 ? "pointer" : "crosshair";
        if (hover !== st.hover || hoverTag !== st.hoverTag) {
            st.hover = hover;
            st.hoverTag = hoverTag;
            requestDraw(st);
        }
    });

    const finish = (e) => {
        if (!st.drag) return;
        st.drag = null;
        st.dragging = false;
        if (e?.pointerId !== undefined && canvas.hasPointerCapture?.(e.pointerId)) {
            canvas.releasePointerCapture(e.pointerId);
        }
        const b = st.regions[st.active];
        if (b && (b.w < 0.005 || b.h < 0.005)) {
            removeRegion(st, node, st.active);
            return;
        }
        sync(st, node);
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);
    canvas.addEventListener("pointerleave", () => {
        if (st.drag || (st.hover === -1 && st.hoverTag === -1)) return;
        st.hover = -1;
        st.hoverTag = -1;
        st.canvas.style.cursor = "crosshair";
        requestDraw(st);
    });

    canvas.addEventListener("dblclick", (e) => {
        e.stopPropagation();
        const hit = hitAt(st, pointerNorm(st, e), false);
        if (hit) {
            st.active = hit.index;
            renderInspector(st, node);
            st.descInput.focus();
        }
        requestDraw(st);
    });

    canvas.addEventListener("keydown", (e) => {
        if (st.drag || st.active < 0) return;
        if (e.key === "Delete" || e.key === "Backspace") {
            e.preventDefault();
            e.stopPropagation();
            removeRegion(st, node, st.active);
        }
    });
}

function destroyEditor(node) {
    const st = editors.get(node);
    if (!st) return;
    closePicker(st, true);
    st.widget?.onRemove?.();
    st.root?.remove();
}

// Our widget can outlive its slot in node.widgets (configure() replaces the widget
// list with the serialised one), which would leave an orphaned editor on screen.
function purgeEditors(node) {
    for (const widget of [...(node.widgets ?? [])]) {
        if (widget.name !== WIDGET_NAME) continue;
        widget.element?.remove();
        node.removeWidget?.(widget);
    }
}

function attach(node) {
    const previous = editors.get(node);
    if (previous && getWidget(node, WIDGET_NAME)) {
        readState(previous, node);
        return previous;
    }
    destroyEditor(node);
    purgeEditors(node);

    const st = {
        regions: [],
        active: -1,
        hover: -1,
        hoverTag: -1,
        tagRects: [],
        drag: null,
        dragging: false,
        bg: null,
        grid: true,
        dims: [1024, 1024],
        runDims: undefined,
        widgetDims: [1024, 1024],
        loadedFrom: undefined,
        syncedAt: 0,
        picker: null,
        pickerTimer: 0,
        colorMode: "hex",
        swatchDrag: null,
        swatchMovedAt: 0,
        needsFit: true,
        raf: 0,
    };
    editors.set(node, st);
    hideInternalWidgets(st, node);
    if (typeof node.properties?.bboxGrid === "boolean") st.grid = node.properties.bboxGrid;
    buildUi(st, node);

    const widget = node.addDOMWidget(WIDGET_NAME, "ppm_bounding_boxes", st.root, {
        serialize: false,
        hideOnZoom: false,
        surfaces: { canvas: "shown", vueNode: "never", panel: "never" },
        getMinHeight: () => widgetHeight(st, node),
        getMaxHeight: () => widgetHeight(st, node),
        onDraw: () => maintain(st, node),
    });
    widget.serialize = false;
    const onRemove = widget.onRemove?.bind(widget);
    widget.onRemove = () => {
        closePicker(st, true);
        onRemove?.();
        if (editors.get(node) === st) editors.delete(node);
    };
    st.widget = widget;

    readState(st, node);
    if (node.computeSize && node.setSize) {
        const [cw, ch] = node.computeSize();
        node.setSize([
            Math.max(cw, node.size[0], DEFAULT_NODE_W),
            Math.max(ch, node.size[1]),
        ]);
    }
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "PPM.CreateBoundingBoxes",
    setup() {
        // The frontend does not call node.onExecuted for this node.
        api.addEventListener("executed", ({ detail }) => {
            for (const [node, st] of editors) {
                if (String(node.id) === String(detail.node)) applyExecuted(st, node, detail.output);
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS || nodeType._ppmBboxEditor) return;
        nodeType._ppmBboxEditor = true;

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = origConfigure?.apply(this, args);
            if (editors.has(this)) attach(this);
            return result;
        };

        const origConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const result = origConnectionsChange?.apply(this, args);
            const st = editors.get(this);
            if (st) updateSurfaces(st, this);
            return result;
        };
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        if (node.graph) {
            attach(node);
            return;
        }
        // Runs from the node constructor, before the graph has this node.
        const onAdded = node.onAdded;
        node.onAdded = function (...args) {
            const result = onAdded?.apply(this, args);
            attach(this);
            return result;
        };
    },
});
