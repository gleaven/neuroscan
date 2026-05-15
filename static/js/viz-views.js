/**
 * NEUROSCAN — 2D Interpretability Visualization Views
 *
 * Three Canvas 2D view classes for mechanistic interpretability:
 *   1. HeatmapView   — Activation magnitudes: Y=layers, X=tokens
 *   2. AttentionView  — Attention patterns: seq×seq matrix per layer/head
 *   3. LogitLensView  — Model's evolving predictions layer-by-layer
 *
 * All classes share a uniform interface:
 *   init(canvas), setTokens(tokens), addLayer(layerData, nLayers),
 *   onComplete(data), clear(), resize(), dispose()
 */

// ── Canvas buffer sync ──────────────────────────────────────
// Canvas has two sizes: CSS display size and drawing buffer size.
// The buffer defaults to 300×150 and must be explicitly synced.
function syncCanvasSize(canvas) {
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (cw > 0 && ch > 0 && (canvas.width !== cw || canvas.height !== ch)) {
        canvas.width = cw;
        canvas.height = ch;
    }
}

// ── Color Utilities ─────────────────────────────────────────
function heatColor(value, min, max) {
    // Diverging colormap: cyan → black → amber/red
    if (max === min) return 'rgba(0, 200, 220, 0.3)';
    const t = Math.max(0, Math.min(1, (value - min) / (max - min))); // 0..1, clamped
    if (t < 0.5) {
        const s = t * 2;
        const r = Math.round(0 * (1 - s));
        const g = Math.round(229 * (1 - s));
        const b = Math.round(255 * (1 - s));
        return `rgb(${r},${g},${b})`;
    } else {
        const s = (t - 0.5) * 2;
        const r = Math.round(255 * s);
        const g = Math.round(170 * s * (1 - s * 0.4));
        const b = 0;
        return `rgb(${r},${g},${b})`;
    }
}

function purpleIntensity(probability) {
    const a = Math.min(1, probability * 2.5);
    return `rgba(0, 200, 220, ${(0.1 + a * 0.85).toFixed(2)})`;
}


// ── Special token detection ──────────────────────────────────
// Tokens that act as attention sinks (BOS/EOS/PAD) produce
// activation norms 5-100× larger than content tokens, which
// crushes the heatmap color scale.  We identify them by pattern
// so they can be excluded from normalization and dimmed visually.
const SPECIAL_TOKEN_RE = /^(<\|?(?:endoftext|bos|eos|pad|sep|cls|mask|unk|s|\/s)\|?>|<s>|<\/s>|\[CLS\]|\[SEP\]|\[PAD\]|\[MASK\]|\[UNK\])$/i;

function isSpecialToken(tok) {
    return SPECIAL_TOKEN_RE.test(tok);
}


// ═══════════════════════════════════════════════════════════════
// HeatmapView — Activation Heatmap (Y=layers, X=tokens)
// ═══════════════════════════════════════════════════════════════
class HeatmapView {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.tokens = [];
        this.layers = [];
        this.nLayers = 0;
        this.globalMin = Infinity;
        this.globalMax = -Infinity;
        this._onClickCb = null;
        this._specialCols = new Set();  // indices of special tokens
    }

    init(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    }

    setTokens(tokens) {
        this.tokens = tokens;
        // Pre-compute which columns are special tokens (attention sinks)
        this._specialCols = new Set();
        for (let i = 0; i < tokens.length; i++) {
            if (isSpecialToken(tokens[i])) this._specialCols.add(i);
        }
    }

    addLayer(layerData, nLayers) {
        this.nLayers = nLayers;
        if (!layerData.heatmap) return;

        this.layers.push({
            layer: layerData.layer,
            heatmap: layerData.heatmap,
        });

        this._recomputeBounds();
        this._draw();
    }

    onComplete() {
        this._recomputeBounds();
        this._draw();
    }

    clear() {
        this.layers = [];
        this.tokens = [];
        this.globalMin = Infinity;
        this.globalMax = -Infinity;
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    /** Recompute color bounds from CONTENT tokens only (P2/P98).
     *  Special tokens (BOS/EOS/PAD) are excluded from normalization
     *  so they don't crush the color scale for real content. */
    _recomputeBounds() {
        const contentVals = [];
        for (const ld of this.layers) {
            for (let i = 0; i < ld.heatmap.length; i++) {
                if (!this._specialCols.has(i)) contentVals.push(ld.heatmap[i]);
            }
        }
        // If ALL tokens are special (unlikely), fall back to everything
        const allVals = contentVals.length > 0 ? contentVals : this.layers.flatMap(ld => ld.heatmap);
        if (allVals.length === 0) return;

        allVals.sort((a, b) => a - b);
        const lo = Math.floor(allVals.length * 0.02);
        const hi = Math.min(allVals.length - 1, Math.floor(allVals.length * 0.98));
        this.globalMin = allVals[lo];
        this.globalMax = allVals[hi];
        // Fallback if range is degenerate
        if (this.globalMax <= this.globalMin) {
            this.globalMin = allVals[0];
            this.globalMax = allVals[allVals.length - 1];
        }
    }

    resize() {
        if (!this.canvas) return;
        syncCanvasSize(this.canvas);
        this._draw();
    }

    dispose() {
        this.canvas = null;
        this.ctx = null;
    }

    onClick(cb) {
        this._onClickCb = cb;
    }

    getCellAt(x, y) {
        if (!this.layers.length || !this.tokens.length) return null;
        const { marginLeft, marginTop, cellW, cellH } = this._layout();
        const col = Math.floor((x - marginLeft) / cellW);
        const row = Math.floor((y - marginTop) / cellH);
        if (col < 0 || col >= this.tokens.length || row < 0 || row >= this.layers.length) return null;
        const ld = this.layers[row];
        return {
            layer: ld.layer,
            tokenIdx: col,
            token: this.tokens[col],
            value: ld.heatmap[col],
        };
    }

    _layout() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const nTokens = this.tokens.length || 1;
        const nRows = this.nLayers || this.layers.length || 1;
        const marginLeft = 52;
        const marginBottom = 56;
        const marginTop = 22;           // room for token importance bar row
        const marginRight = 50;         // room for layer stats bars
        const cellW = (w - marginLeft - marginRight) / nTokens;
        const cellH = (h - marginBottom - marginTop) / nRows;
        return { w, h, nTokens, nRows, marginLeft, marginBottom, marginTop, marginRight, cellW, cellH };
    }

    _draw() {
        if (!this.ctx || !this.layers.length) return;
        syncCanvasSize(this.canvas);

        const ctx = this.ctx;
        const { w, h, nTokens, nRows, marginLeft, marginBottom, marginTop, marginRight, cellW, cellH } = this._layout();

        ctx.clearRect(0, 0, w, h);

        // Adaptive font sizes based on cell dimensions
        const rowFontSize = Math.max(9, Math.min(14, cellH * 0.5));
        const colFontSize = Math.max(9, Math.min(13, cellW * 0.4));

        // Draw cells — special tokens (BOS/EOS) are dimmed with a
        // diagonal hatch overlay so the user can focus on content tokens
        for (let row = 0; row < this.layers.length; row++) {
            const ld = this.layers[row];
            const y = marginTop + row * cellH;

            for (let col = 0; col < ld.heatmap.length; col++) {
                const x = marginLeft + col * cellW;
                const isSpecial = this._specialCols.has(col);
                if (isSpecial) {
                    // Dim flat color for special tokens — not color-scaled
                    ctx.fillStyle = 'rgba(60, 60, 80, 0.5)';
                    ctx.fillRect(x, y, cellW - 1, cellH - 1);
                    // Diagonal hatch to mark as "excluded"
                    ctx.strokeStyle = 'rgba(0, 200, 220, 0.15)';
                    ctx.lineWidth = 0.5;
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + cellW - 1, y + cellH - 1);
                    ctx.stroke();
                } else {
                    ctx.fillStyle = heatColor(ld.heatmap[col], this.globalMin, this.globalMax);
                    ctx.fillRect(x, y, cellW - 1, cellH - 1);
                }
            }

            // Row label
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.font = `${rowFontSize}px "Share Tech Mono"`;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(`L${ld.layer}`, marginLeft - 6, y + cellH / 2);
        }

        // Column labels (tokens) — at bottom, rotated
        // Special tokens are dimmed + italic to match the hatched cells
        ctx.save();
        ctx.font = `${colFontSize}px "Share Tech Mono"`;
        ctx.textAlign = 'left';
        for (let col = 0; col < this.tokens.length; col++) {
            const isSpecial = this._specialCols.has(col);
            ctx.fillStyle = isSpecial ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.55)';
            const x = marginLeft + col * cellW + cellW / 2;
            const y = marginTop + nRows * cellH + 6;
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(Math.PI / 4);
            const maxLen = Math.max(4, Math.floor(cellW / 7));
            const label = this.tokens[col].length > maxLen
                ? this.tokens[col].slice(0, maxLen - 1) + '\u2026'
                : this.tokens[col];
            ctx.fillText(label, 0, 0);
            ctx.restore();
        }
        ctx.restore();

        // ── Right margin: per-layer mean activation bars (content tokens only) ──
        if (this.layers.length > 1) {
            const barX = marginLeft + nTokens * cellW + 6;
            const barMaxW = marginRight - 14;
            // Compute per-row means, excluding special token columns
            const rowMeans = this.layers.map(ld => {
                let sum = 0, cnt = 0;
                for (let i = 0; i < ld.heatmap.length; i++) {
                    if (!this._specialCols.has(i)) { sum += ld.heatmap[i]; cnt++; }
                }
                return cnt > 0 ? sum / cnt : 0;
            });
            const meanMax = Math.max(...rowMeans, 0.001);

            for (let row = 0; row < this.layers.length; row++) {
                const y = marginTop + row * cellH;
                const barW = (rowMeans[row] / meanMax) * barMaxW;
                const barH = Math.max(2, cellH - 2);
                // Bar background
                ctx.fillStyle = 'rgba(255,255,255,0.03)';
                ctx.fillRect(barX, y, barMaxW, barH);
                // Bar fill
                const t = rowMeans[row] / meanMax;
                ctx.fillStyle = `rgba(${Math.round(255 * t)}, ${Math.round(170 * t * (1 - t * 0.4))}, 0, 0.7)`;
                ctx.fillRect(barX, y, barW, barH);
            }
            // Bar header
            ctx.fillStyle = 'rgba(255,255,255,0.3)';
            ctx.font = `${Math.max(7, Math.min(9, marginRight * 0.2))}px "Orbitron"`;
            ctx.textAlign = 'center';
            ctx.fillText('MEAN', barX + barMaxW / 2, marginTop - 5);
        }

        // ── Top margin: per-token importance mini-bars (content tokens only) ──
        if (this.layers.length > 1 && this.tokens.length > 0) {
            const barY = 2;
            const barMaxH = marginTop - 6;
            // Compute per-column sums, excluding special tokens
            const colSums = new Array(nTokens).fill(0);
            for (const ld of this.layers) {
                for (let col = 0; col < ld.heatmap.length && col < nTokens; col++) {
                    if (!this._specialCols.has(col)) colSums[col] += ld.heatmap[col];
                }
            }
            const colMax = Math.max(...colSums, 0.001);

            for (let col = 0; col < nTokens; col++) {
                const x = marginLeft + col * cellW;
                const isSpecial = this._specialCols.has(col);
                if (isSpecial) continue; // no importance bar for special tokens

                const barH = (colSums[col] / colMax) * barMaxH;
                const barW = Math.max(2, cellW - 1);
                // Draw from bottom of top margin upward
                const by = marginTop - 3 - barH;
                // Bar background
                ctx.fillStyle = 'rgba(255,255,255,0.03)';
                ctx.fillRect(x, marginTop - 3 - barMaxH, barW, barMaxH);
                // Bar fill
                const t = colSums[col] / colMax;
                ctx.fillStyle = `rgba(0, 200, 220, ${(0.2 + t * 0.7).toFixed(2)})`;
                ctx.fillRect(x, by, barW, barH);
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════
// AttentionView — Attention Pattern Matrix (seq × seq)
// ═══════════════════════════════════════════════════════════════
class AttentionView {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.tokens = [];
        this.layers = [];
        this.nLayers = 0;
        this.selectedLayer = 0;
        this.selectedHead = -1;
        this._headCache = {};
        this._currentPattern = null;
    }

    init(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    }

    setTokens(tokens) {
        this.tokens = tokens;
    }

    addLayer(layerData, nLayers) {
        this.nLayers = nLayers;
        if (!layerData.attention_summary) return;

        this.layers.push({
            layer: layerData.layer,
            attention_summary: layerData.attention_summary,
            n_heads: layerData.n_heads || 1,
        });

        if (this.layers.length === 1) {
            this.selectedLayer = layerData.layer;
            this._currentPattern = layerData.attention_summary;
            this._draw();
        }
    }

    onComplete() {
        this.selectLayer(this.selectedLayer);
    }

    clear() {
        this.layers = [];
        this.tokens = [];
        this._headCache = {};
        this._currentPattern = null;
        this.selectedLayer = 0;
        this.selectedHead = -1;
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    resize() {
        if (!this.canvas) return;
        syncCanvasSize(this.canvas);
        this._draw();
    }

    dispose() {
        this.canvas = null;
        this.ctx = null;
    }

    selectLayer(layerIdx) {
        this.selectedLayer = layerIdx;
        this.selectedHead = -1;
        const ld = this.layers.find(l => l.layer === layerIdx);
        if (ld) {
            this._currentPattern = ld.attention_summary;
            this._draw();
        }
    }

    setHeadPattern(pattern) {
        this._currentPattern = pattern;
        this._draw();
    }

    getNHeads() {
        const ld = this.layers.find(l => l.layer === this.selectedLayer);
        return ld ? ld.n_heads : 0;
    }

    getCellAt(x, y) {
        if (!this._currentPattern || !this.tokens.length) return null;
        const layout = this._layout();
        const col = Math.floor((x - layout.marginLeft) / layout.cellSize);
        const row = Math.floor((y - layout.marginTop) / layout.cellSize);
        const n = this._currentPattern.length;
        if (col < 0 || col >= n || row < 0 || row >= n) return null;
        return {
            fromToken: this.tokens[row],
            toToken: this.tokens[col],
            fromIdx: row,
            toIdx: col,
            weight: this._currentPattern[row][col],
        };
    }

    _layout() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const n = this._currentPattern ? this._currentPattern.length : 1;
        // Scale margins proportionally to canvas size
        const marginLeft = Math.max(50, Math.min(80, w * 0.08));
        const marginTop = Math.max(50, Math.min(80, h * 0.1));
        const marginRight = 20;
        const marginBottom = 20;
        const available = Math.min(w - marginLeft - marginRight, h - marginTop - marginBottom);
        const cellSize = Math.max(4, available / n);
        return { w, h, n, marginLeft, marginTop, marginRight, marginBottom, cellSize };
    }

    _draw() {
        if (!this.ctx || !this._currentPattern) return;
        syncCanvasSize(this.canvas);

        const ctx = this.ctx;
        const pattern = this._currentPattern;
        const { w, h, n, marginLeft, marginTop, cellSize } = this._layout();

        ctx.clearRect(0, 0, w, h);

        const labelFontSize = Math.max(9, Math.min(14, cellSize * 0.7));

        // Draw attention matrix
        for (let row = 0; row < n; row++) {
            for (let col = 0; col < n; col++) {
                const val = pattern[row][col];
                const alpha = Math.min(1, val * 3);
                ctx.fillStyle = `rgba(0, 200, 220, ${alpha.toFixed(3)})`;
                ctx.fillRect(
                    marginLeft + col * cellSize,
                    marginTop + row * cellSize,
                    cellSize - (cellSize > 6 ? 1 : 0),
                    cellSize - (cellSize > 6 ? 1 : 0)
                );
            }
        }

        // Row labels (source tokens)
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
        ctx.font = `${labelFontSize}px "Share Tech Mono"`;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i < Math.min(n, this.tokens.length); i++) {
            const maxLen = Math.max(3, Math.floor(marginLeft / 9));
            const label = this.tokens[i].length > maxLen
                ? this.tokens[i].slice(0, maxLen - 1) + '\u2026'
                : this.tokens[i];
            ctx.fillText(label, marginLeft - 5, marginTop + i * cellSize + cellSize / 2);
        }

        // Column labels (target tokens) — rotated
        ctx.save();
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
        ctx.font = `${labelFontSize}px "Share Tech Mono"`;
        ctx.textAlign = 'left';
        for (let i = 0; i < Math.min(n, this.tokens.length); i++) {
            const x = marginLeft + i * cellSize + cellSize / 2;
            const y = marginTop - 5;
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(-Math.PI / 4);
            const maxLen = Math.max(3, Math.floor(marginTop / 9));
            const label = this.tokens[i].length > maxLen
                ? this.tokens[i].slice(0, maxLen - 1) + '\u2026'
                : this.tokens[i];
            ctx.fillText(label, 0, 0);
            ctx.restore();
        }
        ctx.restore();

        // Axis titles
        const titleFontSize = Math.max(9, Math.min(12, w * 0.012));
        ctx.fillStyle = 'rgba(0, 200, 220, 0.4)';
        ctx.font = `${titleFontSize}px "Orbitron"`;
        ctx.textAlign = 'center';
        ctx.fillText('DESTINATION TOKEN \u2192', marginLeft + (n * cellSize) / 2, marginTop - marginTop * 0.7);
        ctx.save();
        ctx.translate(14, marginTop + (n * cellSize) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('SOURCE TOKEN \u2192', 0, 0);
        ctx.restore();

        // Layer/head indicator
        ctx.fillStyle = 'rgba(0, 200, 220, 0.35)';
        ctx.font = `${titleFontSize}px "Orbitron"`;
        ctx.textAlign = 'right';
        const headLabel = this.selectedHead === -1 ? 'Mean (all heads)' : `Head ${this.selectedHead}`;
        ctx.fillText(`Layer ${this.selectedLayer} \u2022 ${headLabel}`, w - 16, h - 12);
    }
}


// ═══════════════════════════════════════════════════════════════
// LogitLensView — Model's Evolving Predictions
// ═══════════════════════════════════════════════════════════════
class LogitLensView {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.tokens = [];
        this.layers = [];
        this.nLayers = 0;
    }

    init(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    }

    setTokens(tokens) {
        this.tokens = tokens;
    }

    addLayer(layerData, nLayers) {
        this.nLayers = nLayers;
        if (!layerData.logit_lens) return;

        this.layers.push({
            layer: layerData.layer,
            logit_lens: layerData.logit_lens,
        });

        this._draw();
    }

    onComplete() {
        this._draw();
    }

    clear() {
        this.layers = [];
        this.tokens = [];
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    resize() {
        if (!this.canvas) return;
        syncCanvasSize(this.canvas);
        this._draw();
    }

    dispose() {
        this.canvas = null;
        this.ctx = null;
    }

    getCellAt(x, y) {
        if (!this.layers.length) return null;
        const { marginLeft, marginTop, colWidth, rowH } = this._layout();
        const row = Math.floor((y - marginTop) / rowH);
        const col = Math.floor((x - marginLeft) / colWidth);
        if (row < 0 || row >= this.layers.length || col < 0 || col >= 5) return null;
        const ld = this.layers[row];
        const tok = ld.logit_lens.top_tokens[col];
        return {
            layer: ld.layer,
            rank: col + 1,
            token: tok.token,
            probability: tok.probability,
            matchesFinal: ld.logit_lens.matches_final && col === 0,
        };
    }

    _layout() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const nRows = this.nLayers || this.layers.length || 1;
        const marginLeft = 52;
        const marginTop = 32;
        // Reserve space for probability bars: proportional to width
        const barAreaWidth = Math.max(60, w * 0.1);
        const marginRight = barAreaWidth + 16;
        const marginBottom = 16;
        const colWidth = (w - marginLeft - marginRight) / 5;
        const rowH = (h - marginTop - marginBottom) / nRows;
        return { w, h, nRows, marginLeft, marginTop, marginRight, marginBottom, colWidth, rowH, barAreaWidth };
    }

    _draw() {
        if (!this.ctx || !this.layers.length) return;
        syncCanvasSize(this.canvas);

        const ctx = this.ctx;
        const { w, h, nRows, marginLeft, marginTop, marginRight, colWidth, rowH, barAreaWidth } = this._layout();

        ctx.clearRect(0, 0, w, h);

        // Adaptive font sizes
        const headerFontSize = Math.max(9, Math.min(13, w * 0.012));
        const rowLabelFontSize = Math.max(10, Math.min(14, rowH * 0.5));
        const tokenFontSize = Math.max(10, Math.min(16, Math.min(rowH * 0.5, colWidth * 0.14)));
        const probFontSize = Math.max(8, Math.min(12, Math.min(rowH * 0.35, colWidth * 0.1)));

        // Column headers: Rank 1-5
        ctx.fillStyle = 'rgba(0, 200, 220, 0.5)';
        ctx.font = `${headerFontSize}px "Orbitron"`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        for (let c = 0; c < 5; c++) {
            ctx.fillText(`RANK #${c + 1}`, marginLeft + c * colWidth + colWidth / 2, marginTop - 8);
        }
        // Prob bar header
        ctx.fillText('PROB', marginLeft + 5 * colWidth + barAreaWidth / 2, marginTop - 8);

        // Draw each layer row
        for (let row = 0; row < this.layers.length; row++) {
            const ld = this.layers[row];
            const y = marginTop + row * rowH;
            const ll = ld.logit_lens;

            // Row background for convergence highlight
            if (ll.matches_final) {
                ctx.fillStyle = 'rgba(0, 255, 136, 0.06)';
                ctx.fillRect(marginLeft, y, 5 * colWidth, rowH);
            }

            // Layer label
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.font = `${rowLabelFontSize}px "Share Tech Mono"`;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(`L${ld.layer}`, marginLeft - 6, y + rowH / 2);

            // Token cells
            for (let c = 0; c < ll.top_tokens.length; c++) {
                const tok = ll.top_tokens[c];
                const x = marginLeft + c * colWidth;

                // Cell background
                ctx.fillStyle = purpleIntensity(tok.probability);
                ctx.fillRect(x, y, colWidth - 1, rowH - 1);

                // Green border for convergence
                if (c === 0 && ll.matches_final) {
                    ctx.strokeStyle = 'rgba(0, 255, 136, 0.7)';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x + 1, y + 1, colWidth - 3, rowH - 3);
                }

                // Token text (centered vertically)
                ctx.fillStyle = tok.probability > 0.3 ? '#ffffff' : 'rgba(255,255,255,0.7)';
                ctx.font = `${tokenFontSize}px "Share Tech Mono"`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const maxLen = Math.max(3, Math.floor(colWidth / (tokenFontSize * 0.6)));
                const label = tok.token.length > maxLen
                    ? tok.token.slice(0, maxLen - 1) + '\u2026'
                    : tok.token;
                // Offset slightly up to make room for probability below
                ctx.fillText(label, x + colWidth / 2, y + rowH * 0.4);

                // Probability text below token
                ctx.fillStyle = 'rgba(255,255,255,0.45)';
                ctx.font = `${probFontSize}px "Share Tech Mono"`;
                ctx.textBaseline = 'middle';
                ctx.fillText(`${(tok.probability * 100).toFixed(1)}%`, x + colWidth / 2, y + rowH * 0.75);
            }

            // Probability bar (rank-1 probability)
            const topProb = ll.top_tokens[0].probability;
            const barX = marginLeft + 5 * colWidth + 8;
            const barW = barAreaWidth - 16;
            const barH = Math.max(4, rowH * 0.35);
            const barY = y + (rowH - barH) / 2;

            ctx.fillStyle = 'rgba(255,255,255,0.05)';
            ctx.fillRect(barX, barY, barW, barH);

            const fillColor = ll.matches_final
                ? 'rgba(0, 255, 136, 0.7)'
                : 'rgba(0, 200, 220, 0.6)';
            ctx.fillStyle = fillColor;
            ctx.fillRect(barX, barY, barW * topProb, barH);
        }
    }
}


// ===============================================================
// KVCacheView — KV-Cache Norm Heatmap  (Y=layers, X=tokens)
//
// Visualizes Key/Value cache tensor norms per layer and token
// position, highlighting attention sink positions and showing
// per-position influence scores as a bar chart.
// ===============================================================
class KVCacheView {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.tokens = [];
        this.layers = [];       // { layer, kv_cache: {...} }
        this.nLayers = 0;
        this.globalMin = Infinity;
        this.globalMax = -Infinity;
        this._onClickCb = null;
        this._specialCols = new Set();
        this._mode = 'k';       // 'k' or 'v' — which norms to display
        this._headIdx = -1;     // -1 = mean across heads
    }

    init(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    }

    setTokens(tokens) {
        this.tokens = tokens;
        this._specialCols = new Set();
        for (let i = 0; i < tokens.length; i++) {
            if (isSpecialToken(tokens[i])) this._specialCols.add(i);
        }
    }

    setMode(mode) {
        this._mode = mode;
        this._recomputeBounds();
        this._draw();
    }

    setHead(headIdx) {
        this._headIdx = headIdx;
        this._recomputeBounds();
        this._draw();
    }

    addLayer(layerData, nLayers) {
        this.nLayers = nLayers;
        if (!layerData.kv_cache) return;
        this.layers.push({
            layer: layerData.layer,
            kv_cache: layerData.kv_cache,
        });
        this._recomputeBounds();
        this._draw();
    }

    onComplete() {
        this._recomputeBounds();
        this._draw();
    }

    clear() {
        this.layers = [];
        this.tokens = [];
        this.globalMin = Infinity;
        this.globalMax = -Infinity;
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    resize() {
        if (!this.canvas) return;
        syncCanvasSize(this.canvas);
        this._draw();
    }

    dispose() { this.canvas = null; this.ctx = null; }

    onClick(cb) { this._onClickCb = cb; }

    // Return norm array for current mode + head selection
    _getNorms(ld) {
        const kv = ld.kv_cache;
        if (this._headIdx === -1) {
            return this._mode === 'k' ? kv.k_mean_norm : kv.v_mean_norm;
        }
        const perHead = this._mode === 'k' ? kv.k_norms : kv.v_norms;
        if (this._headIdx < perHead.length) {
            return perHead[this._headIdx];
        }
        return this._mode === 'k' ? kv.k_mean_norm : kv.v_mean_norm;
    }

    _recomputeBounds() {
        // P2/P98 normalization on content tokens (exclude special tokens)
        const contentVals = [];
        for (const ld of this.layers) {
            const norms = this._getNorms(ld);
            for (let i = 0; i < norms.length; i++) {
                if (!this._specialCols.has(i)) contentVals.push(norms[i]);
            }
        }
        const allVals = contentVals.length > 0 ? contentVals
            : this.layers.flatMap(ld => this._getNorms(ld));
        if (allVals.length === 0) return;
        allVals.sort((a, b) => a - b);
        const lo = Math.floor(allVals.length * 0.02);
        const hi = Math.min(allVals.length - 1, Math.floor(allVals.length * 0.98));
        this.globalMin = allVals[lo];
        this.globalMax = allVals[hi];
        if (this.globalMax <= this.globalMin) {
            this.globalMin = allVals[0];
            this.globalMax = allVals[allVals.length - 1];
        }
    }

    getCellAt(x, y) {
        if (!this.layers.length || !this.tokens.length) return null;
        const { marginLeft, marginTop, cellW, cellH } = this._layout();
        const col = Math.floor((x - marginLeft) / cellW);
        const row = Math.floor((y - marginTop) / cellH);
        if (col < 0 || col >= this.tokens.length || row < 0 || row >= this.layers.length) return null;
        const ld = this.layers[row];
        const norms = this._getNorms(ld);
        const isSink = ld.kv_cache.sink_positions.includes(col);
        return {
            layer: ld.layer,
            tokenIdx: col,
            token: this.tokens[col],
            value: norms[col] != null ? norms[col] : 0,
            isSink: isSink,
            influence: ld.kv_cache.influence_scores[col] || 0,
            memoryBytes: ld.kv_cache.memory_bytes,
        };
    }

    _layout() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const nTokens = this.tokens.length || 1;
        const nRows = this.nLayers || this.layers.length || 1;
        const marginLeft = 52;
        const marginBottom = 56;
        const marginTop = 40;   // room for influence bar chart
        const marginRight = 50;
        const cellW = (w - marginLeft - marginRight) / nTokens;
        const cellH = (h - marginBottom - marginTop) / nRows;
        return { w, h, nTokens, nRows, marginLeft, marginBottom, marginTop, marginRight, cellW, cellH };
    }

    _draw() {
        if (!this.ctx || !this.layers.length) return;
        syncCanvasSize(this.canvas);

        const ctx = this.ctx;
        const { w, h, nTokens, nRows, marginLeft, marginBottom, marginTop,
                marginRight, cellW, cellH } = this._layout();

        ctx.clearRect(0, 0, w, h);
        const rowFontSize = Math.max(9, Math.min(14, cellH * 0.5));
        const colFontSize = Math.max(9, Math.min(13, cellW * 0.4));

        // ── Heatmap cells ──
        for (let row = 0; row < this.layers.length; row++) {
            const ld = this.layers[row];
            const norms = this._getNorms(ld);
            const y = marginTop + row * cellH;
            const sinks = new Set(ld.kv_cache.sink_positions);

            for (let col = 0; col < norms.length; col++) {
                const x = marginLeft + col * cellW;
                const isSpecial = this._specialCols.has(col);
                const isSink = sinks.has(col);

                if (isSpecial) {
                    ctx.fillStyle = 'rgba(60, 60, 80, 0.5)';
                    ctx.fillRect(x, y, cellW - 1, cellH - 1);
                } else {
                    ctx.fillStyle = heatColor(norms[col], this.globalMin, this.globalMax);
                    ctx.fillRect(x, y, cellW - 1, cellH - 1);
                }

                // Sink indicator: red border
                if (isSink && !isSpecial) {
                    ctx.strokeStyle = 'rgba(255, 50, 80, 0.8)';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x + 1, y + 1, cellW - 3, cellH - 3);
                }
            }

            // Row label
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.font = `${rowFontSize}px "Share Tech Mono"`;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(`L${ld.layer}`, marginLeft - 6, y + cellH / 2);
        }

        // ── Column labels (rotated tokens at bottom) ──
        ctx.save();
        ctx.font = `${colFontSize}px "Share Tech Mono"`;
        ctx.textAlign = 'left';
        for (let col = 0; col < this.tokens.length; col++) {
            const isSpecial = this._specialCols.has(col);
            ctx.fillStyle = isSpecial ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.55)';
            const x = marginLeft + col * cellW + cellW / 2;
            const y = marginTop + nRows * cellH + 6;
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(Math.PI / 4);
            const maxLen = Math.max(4, Math.floor(cellW / 7));
            const label = this.tokens[col].length > maxLen
                ? this.tokens[col].slice(0, maxLen - 1) + '\u2026' : this.tokens[col];
            ctx.fillText(label, 0, 0);
            ctx.restore();
        }
        ctx.restore();

        // ── Top margin: influence score bars per token ──
        if (this.layers.length > 0 && this.tokens.length > 0) {
            const avgInfluence = new Array(nTokens).fill(0);
            for (const ld of this.layers) {
                const inf = ld.kv_cache.influence_scores;
                for (let i = 0; i < inf.length && i < nTokens; i++) {
                    avgInfluence[i] += inf[i] / this.layers.length;
                }
            }
            const infMax = Math.max(...avgInfluence, 0.001);
            const barMaxH = marginTop - 16;

            for (let col = 0; col < nTokens; col++) {
                if (this._specialCols.has(col)) continue;
                const x = marginLeft + col * cellW;
                const barH = (avgInfluence[col] / infMax) * barMaxH;
                const barW = Math.max(2, cellW - 1);
                const by = marginTop - 3 - barH;

                // Background
                ctx.fillStyle = 'rgba(255,255,255,0.03)';
                ctx.fillRect(x, marginTop - 3 - barMaxH, barW, barMaxH);
                // Fill — amber for influence
                const t = avgInfluence[col] / infMax;
                ctx.fillStyle = `rgba(255, 170, 0, ${(0.2 + t * 0.7).toFixed(2)})`;
                ctx.fillRect(x, by, barW, barH);
            }

            // Bar header
            ctx.fillStyle = 'rgba(255,170,0,0.4)';
            ctx.font = `${Math.max(7, Math.min(9, marginTop * 0.2))}px "Orbitron"`;
            ctx.textAlign = 'center';
            ctx.fillText('INFLUENCE', marginLeft + (nTokens * cellW) / 2, 8);
        }

        // ── Mode and head indicator (bottom-right) ──
        const titleFontSize = Math.max(9, Math.min(12, w * 0.012));
        ctx.fillStyle = 'rgba(0, 200, 220, 0.35)';
        ctx.font = `${titleFontSize}px "Orbitron"`;
        ctx.textAlign = 'right';
        const modeLabel = this._mode === 'k' ? 'KEY NORMS' : 'VALUE NORMS';
        const headLabel = this._headIdx === -1 ? 'Mean (all heads)' : `Head ${this._headIdx}`;
        ctx.fillText(`${modeLabel} \u2022 ${headLabel}`, w - 16, h - 12);

        // ── Sink legend (bottom-left) ──
        ctx.textAlign = 'left';
        ctx.fillStyle = 'rgba(255, 50, 80, 0.6)';
        ctx.fillText('\u25A0 SINK', 16, h - 12);
    }
}


// ── Export ───────────────────────────────────────────────────
window.HeatmapView = HeatmapView;
window.AttentionView = AttentionView;
window.LogitLensView = LogitLensView;
window.KVCacheView = KVCacheView;
