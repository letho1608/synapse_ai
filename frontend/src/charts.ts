// Small SVG charts for live data: a time-series line chart with crosshair +
// tooltip, a horizontal bar list and a heat cell scale. No chart library.
//
// Rules followed (dataviz skill): one y-axis per chart, 2px lines, >=8px
// markers, recessive grid, legend for >=2 series plus direct end labels for
// <=4, text in ink tokens (never the series color), color follows the entity.

export const SERIES = ['--series-1', '--series-2', '--series-3', '--series-4',
  '--series-5', '--series-6', '--series-7', '--series-8'];

export function seriesColor(slot: number): string {
  // A 9th node folds back to slot 8's hue rather than inventing one; the
  // legend and direct label still name it.
  return `var(${SERIES[Math.min(slot, SERIES.length - 1)]})`;
}

const SVGNS = 'http://www.w3.org/2000/svg';

function esc(v: unknown): string {
  return String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}

export interface Point { t: number; v: number | null }
export interface Series { id: string; label: string; color: string; points: Point[] }

export interface LineOptions {
  unit?: string;
  yMax?: number;              // fixed top (e.g. 100 for percentages)
  digits?: number;
  windowS?: number;           // visible time window ending now
  linearX?: string;           // x is a plain number (e.g. training step); value = axis name
  autoWindow?: boolean;       // shrink the window to the data (min 60 s, max windowS)
  dots?: boolean;             // draw a marker at every point (sparse series)
  empty?: string;
}

const PAD = { l: 40, r: 64, t: 12, b: 24 };

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

function ago(s: number): string {
  if (s < 1) return 'bây giờ';
  if (s < 60) return `-${Math.round(s)}s`;
  return `-${Math.round(s / 60)}m`;
}

/** Live line chart. Call update() with fresh series; hover state survives updates. */
export class LineChart {
  private svg: SVGSVGElement;
  private tip: HTMLDivElement;
  private legend: HTMLDivElement;
  private series: Series[] = [];
  private hoverX: number | null = null;
  private w = 600;
  private h: number;
  private ro: ResizeObserver;

  constructor(private host: HTMLElement, private opts: LineOptions = {}, height = 180) {
    this.h = height;
    host.classList.add('chart');
    host.innerHTML = '';
    this.legend = document.createElement('div');
    this.legend.className = 'chart-legend';
    this.svg = document.createElementNS(SVGNS, 'svg');
    this.svg.setAttribute('class', 'chart-svg');
    this.svg.setAttribute('role', 'img');
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    this.tip.hidden = true;
    host.append(this.legend, this.svg, this.tip);
    this.svg.addEventListener('pointermove', e => {
      const r = this.svg.getBoundingClientRect();
      this.hoverX = e.clientX - r.left;
      this.draw();
    });
    this.svg.addEventListener('pointerleave', () => { this.hoverX = null; this.draw(); });
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(host);
  }

  destroy() { this.ro.disconnect(); }

  update(series: Series[]) {
    this.series = series;
    this.draw();
  }

  private draw() {
    const o = this.opts;
    this.w = Math.max(240, this.host.clientWidth);
    const { w, h } = this;
    this.svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    this.svg.setAttribute('width', String(w));
    this.svg.setAttribute('height', String(h));
    const all = this.series.flatMap(s => s.points);
    const now = o.linearX ? Math.max(1, ...all.map(p => p.t)) : Date.now() / 1000;
    const firstT = Math.min(now, ...all.filter(p => p.v !== null).map(p => p.t));
    const win = o.linearX ? Math.max(1, now - firstT)
      : o.autoWindow ? Math.min(o.windowS ?? 600, Math.max(60, (now - firstT) * 1.15 + 5))
      : (o.windowS ?? 300);
    const t0 = now - win;
    const xText = (t: number) => o.linearX ? `${o.linearX} ${Math.round(t)}` : ago(now - t);
    const pts = this.series.flatMap(s => s.points.filter(p => p.t >= t0 && p.v !== null)) as Array<{ t: number; v: number }>;
    const digits = o.digits ?? 0;
    const unit = o.unit ?? '';

    this.legend.innerHTML = this.series.length >= 2
      ? this.series.map(s => `<span class="legend-item"><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join('')
      : '';

    if (!pts.length) {
      this.svg.innerHTML = `<text x="${w / 2}" y="${h / 2}" class="chart-empty" text-anchor="middle">${esc(o.empty || 'Chưa có dữ liệu')}</text>`;
      this.tip.hidden = true;
      return;
    }
    const yMax = o.yMax ?? niceMax(Math.max(...pts.map(p => p.v)) * 1.1);
    const x = (t: number) => PAD.l + ((t - t0) / win) * (w - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - Math.min(v, yMax) / yMax) * (h - PAD.t - PAD.b);

    let g = '';
    for (let i = 0; i <= 4; i++) {
      const v = (yMax * i) / 4;
      const yy = y(v);
      g += `<line class="${i === 0 ? 'chart-base' : 'chart-grid'}" x1="${PAD.l}" x2="${w - PAD.r}" y1="${yy}" y2="${yy}"/>`;
      if (i % 2 === 0) g += `<text class="chart-tick" x="${PAD.l - 6}" y="${yy + 4}" text-anchor="end">${v.toFixed(v < 10 && yMax <= 10 ? 1 : 0)}${unit}</text>`;
    }
    if (o.linearX) {
      for (let i = 0; i <= 4; i++) {
        const t = t0 + (win * i) / 4;
        g += `<text class="chart-tick" x="${x(t)}" y="${h - 6}" text-anchor="middle">${Math.round(t)}</text>`;
      }
    } else {
      const step = win >= 480 ? 120 : win >= 240 ? 60 : win >= 90 ? 30 : 15;
      for (let s = 0; s <= win; s += step) {
        g += `<text class="chart-tick" x="${x(now - s)}" y="${h - 6}" text-anchor="middle">${ago(s)}</text>`;
      }
    }

    const labels: Array<{ y: number; text: string; color: string }> = [];
    for (const s of this.series) {
      const vis = s.points.filter(p => p.t >= t0);
      // break the line at missing samples instead of drawing through them
      let d = '';
      let pen = false;
      for (const p of vis) {
        if (p.v === null) { pen = false; continue; }
        d += `${pen ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`;
        pen = true;
      }
      g += `<path class="chart-line" d="${d}" style="stroke:${s.color}"/>`;
      if (o.dots) {
        for (const p of vis) if (p.v !== null)
          g += `<circle class="chart-dot" cx="${x(p.t)}" cy="${y(p.v)}" r="4" style="fill:${s.color}"/>`;
      }
      const last = [...vis].reverse().find(p => p.v !== null);
      if (last && last.v !== null) {
        g += `<circle class="chart-end" cx="${x(last.t)}" cy="${y(last.v)}" r="4" style="fill:${s.color}"/>`;
        if (this.series.length <= 4) labels.push({ y: y(last.v), text: `${s.label} ${last.v.toFixed(digits)}${unit}`, color: s.color });
      }
    }
    // direct end labels, nudged apart so they never overlap
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
    for (const l of labels) g += `<text class="chart-label" x="${w - PAD.r + 8}" y="${l.y + 4}">${esc(l.text)}</text>`;

    // crosshair + tooltip at the nearest sample time
    this.tip.hidden = true;
    if (this.hoverX !== null && this.hoverX >= PAD.l && this.hoverX <= w - PAD.r) {
      const tH = t0 + ((this.hoverX - PAD.l) / (w - PAD.l - PAD.r)) * win;
      let best: number | null = null;
      for (const p of pts) if (best === null || Math.abs(p.t - tH) < Math.abs(best - tH)) best = p.t;
      if (best !== null) {
        const bx = x(best);
        g += `<line class="chart-cross" x1="${bx}" x2="${bx}" y1="${PAD.t}" y2="${h - PAD.b}"/>`;
        const rows = this.series.map(s => {
          const p = s.points.reduce<Point | null>((a, q) => (q.v !== null && (a === null || Math.abs(q.t - best!) < Math.abs(a.t - best!)) ? q : a), null);
          if (!p || p.v === null || Math.abs(p.t - best!) > (o.linearX ? win / 50 + 0.5 : 5)) return '';
          g += `<circle class="chart-hover" cx="${x(p.t)}" cy="${y(p.v)}" r="5" style="fill:${s.color}"/>`;
          return `<div><i style="background:${s.color}"></i>${esc(s.label)}<b>${p.v.toFixed(digits)}${unit}</b></div>`;
        }).join('');
        this.tip.innerHTML = `<div class="chart-tip-time">${xText(best)}</div>${rows}`;
        this.tip.hidden = !rows;
        const left = Math.min(bx + 12, w - 170);
        this.tip.style.left = `${Math.max(0, left)}px`;
        this.tip.style.top = `${PAD.t + 20}px`;
      }
    }
    this.svg.innerHTML = g;
    this.svg.setAttribute('aria-label', this.series.map(s => s.label).join(', '));
  }
}

/** Horizontal bars (one measure, one color). Values labelled at the bar end. */
export function barList(rows: Array<{ label: string; value: number }>, color = 'var(--series-1)', unit = ''): string {
  if (!rows.length) return '<div class="muted">Chưa có dữ liệu.</div>';
  const max = Math.max(...rows.map(r => r.value), 1);
  return `<div class="bar-list">${rows.map(r => `
    <div class="bar-row" title="${esc(r.label)}: ${r.value}${unit}">
      <span class="bar-label mono">${esc(r.label)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(100 * r.value / max).toFixed(1)}%;background:${color}"></span></span>
      <span class="bar-value">${r.value}${unit}</span>
    </div>`).join('')}</div>`;
}

/** Sequential one-hue scale for heat cells (dark surface: low recedes, high is bright). */
const SEQ = ['#104281', '#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef'];
export function heatColor(v: number, max: number): string {
  const i = Math.max(0, Math.min(SEQ.length - 1, Math.round((v / (max || 1)) * (SEQ.length - 1))));
  return SEQ[i];
}
