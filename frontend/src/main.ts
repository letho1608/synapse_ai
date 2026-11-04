import './style.css';
import { api, ApiError, getApiKey, setApiKey, streamChat } from './api';
import { live } from './live';
import { LineChart, barList, heatColor, seriesColor, type Series } from './charts';
import type {
  ActivityEntry, ClusterResources, Dataset, DistributedStatus, ModelsList, Pairing,
  RequestMetric, Settings, SystemInfo, TrainingJob,
} from './types';

type Page = 'dashboard' | 'cluster' | 'chat' | 'models' | 'training' | 'datasets' | 'logs' | 'settings';

const PAGES: Array<{ id: Page; title: string; subtitle: string; section: string; icon: string }> = [
  { id: 'dashboard', section: 'Cụm', title: 'Tổng quan', subtitle: 'Cụm, tốc độ sinh token và tài nguyên theo thời gian thực', icon: '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>' },
  { id: 'cluster', section: 'Cụm', title: 'Node & chia layer', subtitle: 'Mỗi node giữ một dải layer; hidden state đi node này sang node kế', icon: '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>' },
  { id: 'chat', section: 'Model', title: 'Chat', subtitle: 'Gửi prompt qua cụm, đo TTFT và tok/s thật', icon: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>' },
  { id: 'models', section: 'Model', title: 'Model', subtitle: 'Mỗi node chỉ tải phần trọng số của dải layer nó giữ', icon: '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>' },
  { id: 'training', section: 'Model', title: 'Huấn luyện', subtitle: 'Mỗi node train layer của mình (full hoặc LoRA)', icon: '<path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>' },
  { id: 'datasets', section: 'Model', title: 'Dataset', subtitle: 'File .json / .jsonl để huấn luyện', icon: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>' },
  { id: 'logs', section: 'Hệ thống', title: 'Log', subtitle: 'Hoạt động và log của node', icon: '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>' },
  { id: 'settings', section: 'Hệ thống', title: 'Cài đặt', subtitle: 'Cài đặt node, ghép máy (TLS), API key', icon: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>' },
];

function esc(v: unknown): string {
  return String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}

function num(v: number | null | undefined, digits = 1, unit = ''): string {
  return v === null || v === undefined || Number.isNaN(v) ? '—' : `${v.toFixed(digits)}${unit}`;
}

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function errorBox(e: unknown): string {
  if (e instanceof ApiError && e.forbidden) {
    return `<div class="notice notice-warn">Mục này chỉ mở được từ chính máy chạy node, hoặc khi đã nhập API key
      (<a data-nav="settings">Cài đặt</a>). Chi tiết: ${esc(e.message)}</div>`;
  }
  return `<div class="notice notice-error">${esc(e instanceof Error ? e.message : e)}</div>`;
}

async function attempt<T>(fn: () => Promise<T>): Promise<{ ok: true; v: T } | { ok: false; e: unknown }> {
  try { return { ok: true, v: await fn() }; } catch (e) { return { ok: false, e }; }
}

function toast(msg: string, kind: 'ok' | 'error' = 'ok') {
  const el = document.createElement('div');
  el.className = `toast toast-${kind}`;
  el.setAttribute('role', 'status');
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

function card(title: string, body: string, extra = '', cls = ''): string {
  return `<section class="glass-panel ${cls}"><div class="card-title-bar"><h3>${title}</h3>${extra}</div>
    <div class="card-body">${body}</div></section>`;
}

function pill(status: string): string {
  const k = /complete|success|ok|done/.test(status) ? 'ok' : /fail|error/.test(status) ? 'error' : 'warn';
  const icon = k === 'ok' ? '✓' : k === 'error' ? '✕' : '●';
  return `<span class="pill pill-${k}">${icon} ${esc(status)}</span>`;
}

interface ChatMsg { role: 'user' | 'ai'; text: string; meta?: string; error?: boolean }

/** Something mounted on a page that must be torn down when leaving it. */
interface Mounted { destroy(): void }

class App {
  private page: Page = (location.hash.slice(1) as Page) || 'dashboard';
  private info: SystemInfo | null = null;
  private mounted: Mounted[] = [];

  private chat: ChatMsg[] = [];
  private chatModel = '';
  private chatTemp = 0.7;
  private chatMax = 256;
  private chatSystem = '';
  private chatAbort: AbortController | null = null;
  private modelTab: 'downloaded' | 'catalog' | 'search' | 'fit' | 'checkpoints' = 'downloaded';
  private modelFilter = '';
  private searchQuery = '';
  private fitFilter = '';
  private fitPage = 1;
  private datasetPreview: { path: string; name: string; lines: string[]; edit: boolean } | null = null;
  private logFilter = '';
  private logFollow = true;

  constructor() {
    if (!PAGES.some(p => p.id === this.page)) this.page = 'dashboard';
    live.start();
    live.subscribe(() => this.refreshFooter());
    this.renderShell();
    this.show();
    document.addEventListener('click', e => {
      const nav = (e.target as HTMLElement).closest('[data-nav]') as HTMLElement | null;
      if (nav) { e.preventDefault(); this.go(nav.dataset.nav as Page); }
    });
    window.addEventListener('hashchange', () => {
      const p = location.hash.slice(1) as Page;
      if (p && p !== this.page && PAGES.some(x => x.id === p)) { this.page = p; this.renderShell(); this.show(); }
    });
  }

  private go(p: Page) {
    if (p === this.page) return;
    this.page = p;
    location.hash = p;
    this.renderShell();
    this.show();
  }

  // -- shell ------------------------------------------------------------------
  private renderShell() {
    const app = document.getElementById('app')!;
    const meta = PAGES.find(p => p.id === this.page)!;
    let section = '';
    const nav = PAGES.map(p => {
      const head = p.section !== section ? `<div class="nav-section-title">${p.section}</div>` : '';
      section = p.section;
      return `${head}<a class="nav-item ${p.id === this.page ? 'active' : ''}" data-nav="${p.id}" href="#${p.id}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">${p.icon}</svg>${p.title}</a>`;
    }).join('');
    app.innerHTML = `
      <aside class="sidebar">
        <div class="brand">
          <div class="brand-icon">⚡</div>
          <div class="brand-info"><h1>SYNAPSE AI</h1><span class="badge" id="brand-version">v${esc(this.info?.version || '…')}</span></div>
        </div>
        <nav class="nav-menu">${nav}</nav>
        <div class="sidebar-footer" id="sidebar-footer">Đang kết nối…</div>
      </aside>
      <div class="main-wrapper">
        <header class="top-nav">
          <div class="nav-left"><div>
            <div class="page-title">${meta.title}</div>
            <div class="page-subtitle">${meta.subtitle}</div>
          </div></div>
          <div class="nav-right">
            <span class="live-dot" id="live-dot" title="Số liệu cập nhật mỗi 3 giây">Trực tiếp</span>
            <button class="btn btn-glass" id="btn-refresh">Làm mới</button>
            ${this.page !== 'chat' ? '<button class="btn btn-primary" data-nav="chat">Chat</button>' : ''}
          </div>
        </header>
        <main class="content-container" id="content"><div class="loading">Đang tải…</div></main>
      </div>`;
    document.getElementById('btn-refresh')!.onclick = () => { live.tick(); this.show(); };
    this.refreshFooter();
  }

  private refreshFooter() {
    const el = document.getElementById('sidebar-footer');
    const dot = document.getElementById('live-dot');
    if (dot) dot.classList.toggle('offline', !!live.error && !live.cluster);
    if (!el) return;
    const d = live.dist;
    if (!d) {
      el.innerHTML = live.error
        ? '<div class="cluster-pill offline"><div class="cluster-dot"></div>Mất kết nối node</div>'
        : 'Đang kết nối…';
      return;
    }
    el.innerHTML = `
      <div class="cluster-pill"><div class="cluster-dot"></div>${d.partitions_count} node đang giữ layer</div>
      <div>Node: <b>${esc(d.node_id || '?')}</b>${d.is_master ? ' (master)' : ''}</div>
      <div>Master: ${esc(d.master_id || '—')}</div>
      <div>TLS: ${esc(d.tls_mode || 'off')}${d.api_key_required ? ' · API key' : ''}</div>`;
    if (!this.info) {
      this.info = {} as SystemInfo;
      api.systemInfo().then(i => {
        this.info = i;
        const b = document.getElementById('brand-version');
        if (b) b.textContent = `v${i.version}`;
      }).catch(() => { this.info = null; });
    }
  }

  private content(): HTMLElement { return document.getElementById('content')!; }

  private unmount() {
    this.mounted.forEach(m => { try { m.destroy(); } catch { /* already gone */ } });
    this.mounted = [];
  }

  private async show() {
    const page = this.page;
    this.unmount();
    const html = await this.view(page);
    if (page !== this.page) return;
    this.unmount();
    this.content().innerHTML = html;
    this.bind(page);
  }

  private view(page: Page): Promise<string> {
    switch (page) {
      case 'dashboard': return this.viewDashboard();
      case 'cluster': return this.viewCluster();
      case 'chat': return this.viewChat();
      case 'models': return this.viewModels();
      case 'training': return this.viewTraining();
      case 'datasets': return this.viewDatasets();
      case 'logs': return this.viewLogs();
      case 'settings': return this.viewSettings();
    }
  }

  private bind(page: Page) {
    if (page === 'dashboard') this.bindDashboard();
    if (page === 'cluster') this.bindCluster();
    if (page === 'chat') this.bindChat();
    if (page === 'models') this.bindModels();
    if (page === 'training') this.bindTraining();
    if (page === 'datasets') this.bindDatasets();
    if (page === 'logs') this.bindLogs();
    if (page === 'settings') this.bindSettings();
  }

  /** Run fn now and on every live tick until the page is left. */
  private onLive(fn: () => void) {
    fn();
    const off = live.subscribe(fn);
    this.mounted.push({ destroy: off });
  }

  /** three.js is loaded only by pages that show the 3D view. */
  private async mount3d(id: string) {
    const host = document.getElementById(id);
    if (!host) return;
    host.innerHTML = '<div class="loading center">Đang dựng sơ đồ 3D…</div>';
    const page = this.page;
    const { Cluster3D, nodeViews } = await import('./cluster3d');
    if (page !== this.page || !document.body.contains(host)) return;
    const viz = new Cluster3D(host);
    this.mounted.push(viz);
    this.onLive(() => viz.update(nodeViews(live.cluster, live.dist, n => live.slot(n)), live.busy()));
  }

  private nodeSeries(map: Map<string, Array<{ t: number; v: number | null }>>): Series[] {
    const names = new Map((live.cluster?.nodes || []).map(n => [n.node_id, n.name]));
    return live.nodeIds().map(id => ({
      id, label: names.get(id) || id, color: seriesColor(live.slot(id)), points: map.get(id) || [],
    }));
  }

  private requestSeries(key: 'tok_s' | 'ttft_s', label: string): Series[] {
    return [{
      id: key, label, color: seriesColor(0),
      points: live.requests.filter(r => !r.error).map(r => ({ t: r.at, v: r[key] })),
    }];
  }

  // -- dashboard --------------------------------------------------------------
  private async viewDashboard(): Promise<string> {
    const act = await attempt(() => api.activity());
    return `
      <div class="kpi-grid" id="kpis">${this.kpis()}</div>
      <div class="grid-viz">
        ${card('Cụm', '<div id="viz-dash" class="viz3d-host"></div>', '<a class="link" data-nav="cluster">Chi tiết →</a>', 'viz-card')}
        ${card('Trạng thái', `<div id="dash-cluster">${this.clusterSummary()}</div>`)}
      </div>
      <div class="grid-charts">
        ${card('Tốc độ sinh token (tok/s mỗi request)', '<div id="ch-toks"></div>')}
        ${card('Thời gian tới token đầu (TTFT, giây)', '<div id="ch-ttft"></div>')}
        ${card('CPU theo node (%)', '<div id="ch-cpu"></div>')}
        ${card('RAM theo node (%)', '<div id="ch-ram"></div>')}
      </div>
      ${card('Request gần đây', '<div id="dash-requests"></div>', '<span class="muted small">Đo ở node nhận request</span>')}
      ${card('Hoạt động gần đây', act.ok ? this.activityTable(act.v.slice(-8).reverse()) : errorBox(act.e), '<a class="link" data-nav="logs">Tất cả →</a>')}`;
  }

  private kpis(): string {
    const c = live.cluster;
    const d = live.dist;
    const ok = live.requests.filter(r => !r.error);
    const recent = ok.slice(-10);
    const tps = median(recent.map(r => r.tok_s).filter((v): v is number => v !== null));
    const ttft = median(recent.map(r => r.ttft_s).filter((v): v is number => v !== null));
    const failed = live.requests.length - ok.length;
    const s = live.stats;
    const kpi = (title: string, value: string, meta: string) =>
      `<div class="kpi-card glass-panel"><div class="kpi-header"><span class="kpi-title">${title}</span></div>
       <div class="kpi-value">${value}</div><div class="kpi-meta">${meta}</div></div>`;
    return [
      kpi('Node', c ? String(c.total_nodes) : '—', d ? `master ${esc(d.master_id || '—')} · ${d.partitions_count} giữ layer` : ''),
      kpi('Layer', c ? String(c.total_layers) : '—', esc(c?.strategy_name?.split(' (')[0] || '')),
      kpi('Tốc độ', tps === null ? '—' : `${tps.toFixed(1)} tok/s`, recent.length ? `trung vị ${recent.length} request gần nhất` : 'chưa có request'),
      kpi('TTFT', ttft === null ? '—' : `${(ttft * 1000).toFixed(0)} ms`, failed ? `${failed} request lỗi` : `${live.requests.length} request đã đo`),
      kpi('Máy này', s ? `${num(s.cpu, 0, '%')} CPU` : '—', s ? `RAM ${num(s.memory_used_gb, 1)} / ${num(s.memory_total_gb, 1)} GB` : ''),
    ].join('');
  }

  private clusterSummary(): string {
    const c = live.cluster;
    const d = live.dist;
    if (!c) return live.error ? errorBox(live.error) : '<div class="loading">Đang tải…</div>';
    return `<div class="kv-grid">
        <div><span>Master</span><b>${esc(d?.master_id || '—')}</b></div>
        <div><span>Chia layer theo</span><b>${esc(d?.partition_objective || '—')}</b></div>
        <div><span>Autoscale</span><b>${esc(d?.autoscale_mode || '—')}</b></div>
        <div><span>TLS</span><b>${esc(d?.tls_mode || '—')}</b></div>
        <div><span>Lượng tử</span><b>${esc(d?.quant_active || 'none')}</b></div>
        <div><span>Train nhiều node</span><b>${d?.multi_node_training ? 'có' : 'không'}</b></div>
      </div>
      ${this.layerBar(c)}
      ${d?.autoscale_hints?.length ? `<div class="notice notice-warn">Autoscale: ${d.autoscale_hints.map(h => esc(`${h.action} ${h.node || ''} (${h.reason})`)).join('; ')}</div>` : ''}`;
  }

  private requestTable(rows: RequestMetric[]): string {
    if (!rows.length) return '<div class="muted">Chưa có request nào kể từ khi mở trang. Gửi thử ở trang <a data-nav="chat">Chat</a>.</div>';
    return `<table class="data-table"><thead><tr><th>Giờ</th><th>Model</th><th class="right">Token</th>
      <th class="right">TTFT</th><th class="right">tok/s</th><th class="right">Tổng</th><th>Kết quả</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td class="mono">${new Date(r.at * 1000).toLocaleTimeString()}</td><td class="mono">${esc(r.model)}</td>
        <td class="right">${r.tokens}</td><td class="right">${r.ttft_s === null ? '—' : `${Math.round(r.ttft_s * 1000)} ms`}</td>
        <td class="right">${num(r.tok_s, 1)}</td><td class="right">${num(r.total_s, 1, ' s')}</td>
        <td>${r.error ? `<span class="pill pill-error" title="${esc(r.error)}">✕ lỗi</span>` : '<span class="pill pill-ok">✓ xong</span>'}</td></tr>`).join('')}
    </tbody></table>`;
  }

  private bindDashboard() {
    this.mount3d('viz-dash');
    const toks = new LineChart(document.getElementById('ch-toks')!, { unit: '', digits: 1, dots: true, windowS: 600, autoWindow: true, empty: 'Chưa có request' });
    const ttft = new LineChart(document.getElementById('ch-ttft')!, { unit: 's', digits: 2, dots: true, windowS: 600, autoWindow: true, empty: 'Chưa có request' });
    const cpu = new LineChart(document.getElementById('ch-cpu')!, { unit: '%', yMax: 100, autoWindow: true, windowS: 300 });
    const ram = new LineChart(document.getElementById('ch-ram')!, { unit: '%', yMax: 100, autoWindow: true, windowS: 300 });
    this.mounted.push(toks, ttft, cpu, ram);
    this.onLive(() => {
      toks.update(this.requestSeries('tok_s', 'tok/s'));
      ttft.update(this.requestSeries('ttft_s', 'TTFT'));
      cpu.update(this.nodeSeries(live.cpu));
      ram.update(this.nodeSeries(live.ram));
      const k = document.getElementById('kpis'); if (k) k.innerHTML = this.kpis();
      const s = document.getElementById('dash-cluster'); if (s) s.innerHTML = this.clusterSummary();
      const r = document.getElementById('dash-requests'); if (r) r.innerHTML = this.requestTable(live.requests.slice(-8).reverse());
    });
  }

  private layerBar(c: ClusterResources): string {
    const total = c.total_layers || 1;
    const nodes = c.nodes.filter(n => n.layer_start !== null).sort((a, b) => (a.layer_start! - b.layer_start!));
    if (!nodes.length) return '<div class="muted">Chưa có node nào giữ layer.</div>';
    return `<div class="layer-bar" role="img" aria-label="Chia layer">${nodes.map(n => {
      const w = ((n.layer_end! - n.layer_start! + 1) / total) * 100;
      return `<div class="layer-seg" style="width:${w}%;background:${seriesColor(live.slot(n.node_id))}"
        title="${esc(n.name)}: layer ${n.layer_start}–${n.layer_end}">${esc(n.name)} · ${n.layer_start}–${n.layer_end}</div>`;
    }).join('')}</div>`;
  }

  private activityTable(rows: ActivityEntry[]): string {
    if (!rows.length) return '<div class="muted">Chưa có hoạt động.</div>';
    return `<table class="data-table"><thead><tr><th>Giờ</th><th>Việc</th><th>Model</th><th>Kết quả</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td class="mono">${esc(r.time)}</td><td>${esc(r.action)}</td><td class="mono">${esc(r.model)}</td>
        <td>${pill(r.status)}</td></tr>`).join('')}
    </tbody></table>`;
  }

  // -- cluster ------------------------------------------------------------------
  private async viewCluster(): Promise<string> {
    const net = await attempt(() => api.networkNodes());
    const discovered = net.ok
      ? (net.v.data.length
        ? `<table class="data-table"><thead><tr><th>Node</th><th>Địa chỉ</th><th>Trạng thái</th></tr></thead><tbody>
            ${net.v.data.map(n => `<tr><td>${esc(n.name || n.node_id || n.id || '')}</td><td class="mono">${esc(n.address || n.ip || '')}</td>
              <td>${esc(n.status || '')}</td></tr>`).join('')}</tbody></table>`
        : `<div class="muted">Chưa phát hiện node nào khác qua ${esc(net.v.type)}.</div>`)
      : errorBox(net.e);
    return `
      ${card('Sơ đồ cụm', '<div id="viz-cluster" class="viz3d-host viz3d-tall"></div><div class="muted small">Kéo để xoay, cuộn để phóng to. Vòng quanh mỗi node là phần layer nó giữ; hạt sáng chạy nhanh khi đang có request.</div>', '<span class="muted small" id="cluster-strategy"></span>')}
      ${card('Chia layer', '<div id="cluster-bar"></div>')}
      ${card('Node', '<div id="cluster-nodes" class="scroll-x"></div>')}
      <div class="grid-2col">
        ${card('Độ trễ đo được (ms)', '<div id="cluster-latency" class="scroll-x"></div>')}
        ${card('Đường truyền (số lượt)', '<div id="cluster-transport"></div>')}
        ${card('Điều phối', '<div id="cluster-orch"></div>')}
        ${card(`Phát hiện qua mạng`, discovered)}
      </div>`;
  }

  private bindCluster() {
    this.mount3d('viz-cluster');
    this.onLive(() => {
      const c = live.cluster;
      const d = live.dist;
      const set = (id: string, html: string) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };
      if (!c) { set('cluster-nodes', live.error ? errorBox(live.error) : '<div class="loading">Đang tải…</div>'); return; }
      set('cluster-strategy', `${esc(c.strategy_name)} · ${c.total_layers} layer`);
      set('cluster-bar', this.layerBar(c));
      set('cluster-nodes', this.nodeTable(c, d));
      set('cluster-latency', this.latencyHeat(c));
      set('cluster-transport', barList(Object.entries(d?.transport_counts || {}).sort((a, b) => b[1] - a[1]).map(([label, value]) => ({ label, value }))));
      set('cluster-orch', this.orchestration(d));
    });
  }

  private nodeTable(c: ClusterResources, d: DistributedStatus | null): string {
    return `<table class="data-table"><thead><tr><th>Node</th><th>Chip</th><th class="right">RAM</th><th class="right">VRAM</th>
        <th class="right">CPU</th><th>Layer</th><th class="right">Tỉ lệ</th></tr></thead><tbody>
      ${c.nodes.map(n => `<tr>
        <td><span class="swatch" style="background:${seriesColor(live.slot(n.node_id))}"></span><b>${esc(n.name)}</b>
          ${n.is_self ? ' <span class="pill pill-muted">máy này</span>' : ''}
          ${n.node_id === d?.master_id ? ' <span class="pill pill-ok">master</span>' : ''}
          ${d?.standby_nodes?.includes(n.node_id) ? ' <span class="pill pill-warn">dự phòng</span>' : ''}</td>
        <td>${esc(n.chip)}${n.gpu_backend ? ` <span class="muted small">${esc(n.gpu_backend)}</span>` : ''}</td>
        <td class="right">${n.ram_used_gb !== null ? `${num(n.ram_used_gb)} / ` : ''}${num(n.ram_total_gb ?? n.ram_gb)} GB</td>
        <td class="right">${num(n.vram_gb)} GB</td><td class="right">${num(n.cpu_pct, 0, '%')}</td>
        <td class="mono">${n.layer_start === null ? '—' : `${n.layer_start}–${n.layer_end}`}</td>
        <td class="right">${num(n.share_pct, 0, '%')}</td></tr>`).join('')}
    </tbody></table>`;
  }

  private latencyHeat(c: ClusterResources): string {
    const ids = Object.keys(c.latency_matrix || {});
    if (ids.length < 2) return '<div class="muted">Cần ít nhất 2 node để đo độ trễ.</div>';
    const vals = ids.flatMap(a => ids.filter(b => a !== b).map(b => c.latency_matrix[a]?.[b] ?? 0));
    const max = Math.max(...vals, 1);
    return `<table class="heat-table"><thead><tr><th></th>${ids.map(i => `<th>${esc(i)}</th>`).join('')}</tr></thead><tbody>
      ${ids.map(a => `<tr><th>${esc(a)}</th>${ids.map(b => {
        if (a === b) return '<td class="heat-self">—</td>';
        const v = c.latency_matrix[a]?.[b];
        return v === undefined ? '<td>—</td>' : `<td style="background:${heatColor(v, max)}" title="${esc(a)} → ${esc(b)}: ${v.toFixed(2)} ms">${v.toFixed(1)}</td>`;
      }).join('')}</tr>`).join('')}
      </tbody></table><div class="muted small">Ô càng sáng, độ trễ càng cao (cao nhất ${max.toFixed(1)} ms).</div>`;
  }

  private orchestration(d: DistributedStatus | null): string {
    if (!d) return '<div class="loading">Đang tải…</div>';
    const peers = d.peer_versions && Object.keys(d.peer_versions).length
      ? Object.entries(d.peer_versions).map(([k, v]) => `<li>${esc(k)}: bridge v${esc(v.bridge_version)} · ${esc(v.synapse_version)}</li>`).join('')
      : '<li class="muted">Chưa có peer</li>';
    return `<div class="kv-grid">
        <div><span>Master</span><b>${esc(d.master_id || '—')}</b></div>
        <div><span>Node này</span><b>${esc(d.node_id || '—')}${d.is_master ? ' (master)' : ''}</b></div>
        <div><span>Chia layer theo</span><b>${esc(d.partition_objective || '—')}</b></div>
        <div><span>Autoscale</span><b>${esc(d.autoscale_mode || '—')}${d.autoscale_objective ? ` → ${esc(d.autoscale_objective)}` : ''}</b></div>
        <div><span>Bridge</span><b>v${esc(d.bridge_version ?? '—')}</b></div>
        <div><span>TLS</span><b>${esc(d.tls_mode || '—')}</b></div>
      </div>
      ${d.recovery_degraded ? `<div class="notice notice-warn">Phục hồi checkpoint thiếu trọng số: ${esc(JSON.stringify(d.recovery_degraded))}</div>` : ''}
      <h4>Gợi ý autoscale</h4><ul class="plain">${d.autoscale_hints?.length ? d.autoscale_hints.map(h => `<li>${esc(h.action)} ${esc(h.node || '')} (${esc(h.reason)})</li>`).join('') : '<li class="muted">Không có</li>'}</ul>
      <h4>Phiên bản peer</h4><ul class="plain">${peers}</ul>`;
  }

  // -- chat -------------------------------------------------------------------------
  private async viewChat(): Promise<string> {
    const models = await attempt(() => api.models());
    const names = models.ok ? models.v.downloaded.map(m => m.name) : [];
    if (!this.chatModel) {
      const s = await attempt(() => api.settings());
      this.chatModel = (s.ok && s.v.default_model) || names[0] || '';
    }
    if (this.chatModel && !names.includes(this.chatModel)) names.unshift(this.chatModel);
    const bubbles = this.chat.length ? this.chat.map(m => this.bubble(m)).join('')
      : '<div class="muted center">Chưa có tin nhắn. Câu trả lời đi qua mọi node đang giữ layer của model.</div>';
    return `
      <div class="chat-layout">
        <div class="chat-container glass-panel">
          <div class="chat-toolbar">
            <label>Model <input list="chat-models" id="chat-model" value="${esc(this.chatModel)}" placeholder="repo id trên Hugging Face"></label>
            <datalist id="chat-models">${names.map(n => `<option value="${esc(n)}">`).join('')}</datalist>
            <button class="btn btn-glass" id="chat-clear">Xoá hội thoại</button>
          </div>
          <div class="chat-messages" id="chat-messages" aria-live="polite">${bubbles}</div>
          <div class="chat-input-bar">
            <textarea id="chat-input" rows="2" placeholder="Nhập prompt… (Enter gửi, Shift+Enter xuống dòng)"></textarea>
            <button class="btn btn-primary" id="chat-send">${this.chatAbort ? 'Dừng' : 'Gửi'}</button>
          </div>
        </div>
        <aside class="chat-side">
          ${card('Tham số', `<div class="form-grid one">
            <label>Nhiệt độ (temperature) <input id="chat-temp" type="number" step="0.1" min="0" max="2" value="${this.chatTemp}"></label>
            <label>Số token tối đa <input id="chat-max" type="number" min="1" max="4096" value="${this.chatMax}"></label>
            <label>System prompt <textarea id="chat-system" rows="3" placeholder="Để trống = dùng cài đặt node">${esc(this.chatSystem)}</textarea></label>
          </div>`)}
          ${card('Tốc độ các câu trả lời', '<div id="ch-chat"></div>')}
        </aside>
      </div>`;
  }

  private bubble(m: ChatMsg): string {
    return `<div class="chat-bubble ${m.role}${m.error ? ' error' : ''}"><div class="chat-text">${esc(m.text) || '<span class="muted">…</span>'}</div>
      ${m.meta ? `<div class="chat-meta">${esc(m.meta)}</div>` : ''}</div>`;
  }

  private bindChat() {
    const input = document.getElementById('chat-input') as HTMLTextAreaElement;
    const send = document.getElementById('chat-send') as HTMLButtonElement;
    const modelEl = document.getElementById('chat-model') as HTMLInputElement;
    const tempEl = document.getElementById('chat-temp') as HTMLInputElement;
    const maxEl = document.getElementById('chat-max') as HTMLInputElement;
    const sysEl = document.getElementById('chat-system') as HTMLTextAreaElement;
    const box = document.getElementById('chat-messages')!;
    box.scrollTop = box.scrollHeight;
    modelEl.onchange = () => { this.chatModel = modelEl.value.trim(); };
    tempEl.onchange = () => { this.chatTemp = Number(tempEl.value) || 0; };
    maxEl.onchange = () => { this.chatMax = Math.max(1, Number(maxEl.value) || 256); };
    sysEl.onchange = () => { this.chatSystem = sysEl.value; };
    const chart = new LineChart(document.getElementById('ch-chat')!, { digits: 1, dots: true, windowS: 900, autoWindow: true, empty: 'Chưa có câu trả lời' }, 150);
    this.mounted.push(chart);
    this.onLive(() => chart.update(this.requestSeries('tok_s', 'tok/s')));
    document.getElementById('chat-clear')!.onclick = () => { this.chat = []; this.show(); };
    const go = async () => {
      if (this.chatAbort) { this.chatAbort.abort(); return; }
      const text = input.value.trim();
      if (!text) return;
      this.chatModel = modelEl.value.trim();
      input.value = '';
      this.chat.push({ role: 'user', text });
      const reply: ChatMsg = { role: 'ai', text: '' };
      this.chat.push(reply);
      const history = this.chat.filter(m => m !== reply && !m.error).map(m => ({ role: m.role === 'ai' ? 'assistant' : 'user', content: m.text }));
      if (this.chatSystem.trim()) history.unshift({ role: 'system', content: this.chatSystem.trim() });
      this.chatAbort = new AbortController();
      this.renderChatMessages();
      send.textContent = 'Dừng';
      try {
        const st = await streamChat(this.chatModel, history, delta => {
          reply.text += delta;
          this.updateLastBubble(reply);
        }, this.chatAbort.signal, { temperature: this.chatTemp, max_tokens: this.chatMax });
        const secs = st.totalMs / 1000;
        const tps = st.tokens > 1 && st.ttftMs !== null && st.totalMs > st.ttftMs
          ? (st.tokens - 1) / ((st.totalMs - st.ttftMs) / 1000) : null;
        reply.meta = `TTFT ${st.ttftMs === null ? '—' : Math.round(st.ttftMs) + ' ms'} · ${st.tokens || '?'} token trong ${secs.toFixed(1)} s`
          + (tps ? ` · ${tps.toFixed(1)} tok/s` : '');
        live.tick();
      } catch (e) {
        if ((e as Error).name === 'AbortError') reply.meta = 'Đã dừng';
        else { reply.error = true; reply.text = reply.text || errMsg(e); }
      } finally {
        this.chatAbort = null;
        if (this.page === 'chat') { this.renderChatMessages(); send.textContent = 'Gửi'; input.focus(); }
      }
    };
    send.onclick = go;
    input.onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); go(); } };
    input.focus();
  }

  private renderChatMessages() {
    const box = document.getElementById('chat-messages');
    if (!box) return;
    box.innerHTML = this.chat.map(m => this.bubble(m)).join('');
    box.scrollTop = box.scrollHeight;
  }

  private updateLastBubble(m: ChatMsg) {
    const box = document.getElementById('chat-messages');
    const last = box?.lastElementChild?.querySelector('.chat-text');
    if (!box || !last) return;
    last.textContent = m.text;
    box.scrollTop = box.scrollHeight;
  }

  // -- models -------------------------------------------------------------------------
  private async viewModels(): Promise<string> {
    const tabs: Array<[typeof this.modelTab, string]> = [
      ['downloaded', 'Đã tải'], ['catalog', 'Danh mục'], ['search', 'Tìm trên Hugging Face'],
      ['fit', 'Vừa máy này'], ['checkpoints', 'Checkpoint & gộp LoRA'],
    ];
    const head = card('Tải model', `
        <div class="form-row">
          <input id="pull-name" placeholder="repo id, ví dụ HuggingFaceTB/SmolLM2-135M-Instruct">
          <label class="check"><input type="checkbox" id="pull-full"> Tải cả repo</label>
          <button class="btn btn-primary" id="pull-btn">Tải</button>
        </div>
        <div class="muted small">Mặc định chỉ tải config, tokenizer và các file trọng số chứa dải layer node này đang giữ.
          Các node khác tự tải phần của chúng khi được giao layer.</div>
        <div id="download-progress"></div>`);
    const tabBar = `<div class="tabs" role="tablist">${tabs.map(([id, label]) =>
      `<button class="tab ${id === this.modelTab ? 'active' : ''}" role="tab" aria-selected="${id === this.modelTab}" data-tab="${id}">${label}</button>`).join('')}</div>`;
    return `${head}${tabBar}<div id="model-tab">${await this.modelTabBody()}</div>`;
  }

  private async modelTabBody(): Promise<string> {
    const pullBtn = (n: string) => `<button class="btn btn-glass btn-sm" data-pull-model="${esc(n)}">Tải</button>`;
    if (this.modelTab === 'downloaded' || this.modelTab === 'catalog') {
      const r = await attempt(() => api.models());
      if (!r.ok) return errorBox(r.e);
      const m: ModelsList = r.v;
      if (this.modelTab === 'downloaded') {
        return card(`Đã có trên máy này (${m.downloaded.length})`, m.downloaded.length
          ? `<div class="scroll-x"><table class="data-table"><thead><tr><th>Model</th><th>Tham số</th><th>Kiến trúc</th><th></th></tr></thead><tbody>
            ${m.downloaded.map(x => `<tr><td class="mono">${esc(x.name)}</td><td>${esc(x.parameter_size || '—')}</td><td>${esc(x.architecture || '—')}</td>
              <td class="right"><button class="btn btn-glass btn-sm" data-chat-model="${esc(x.name)}">Chat</button>
              <button class="btn btn-danger btn-sm" data-delete-model="${esc(x.name)}">Xoá</button></td></tr>`).join('')}
            </tbody></table></div>` : '<div class="muted">Chưa có model nào trong cache.</div>');
      }
      const f = this.modelFilter.toLowerCase();
      const avail = m.available.filter(x => !f || x.name.toLowerCase().includes(f) || (x.family || '').toLowerCase().includes(f)).slice(0, 80);
      return card(`Danh mục (${m.available.length})`, `<div class="scroll-x"><table class="data-table"><thead><tr><th>Model</th><th>Tham số</th><th>Bộ nhớ</th><th>Họ</th><th></th></tr></thead><tbody>
          ${avail.map(x => `<tr><td class="mono">${esc(x.name)}</td><td>${esc(x.parameter_size || '—')}</td><td>${esc(x.memory || '—')}</td>
            <td>${esc(x.family || '—')}</td><td class="right">${pullBtn(x.name)}</td></tr>`).join('')}
          </tbody></table></div>${m.available.length > avail.length ? `<div class="muted small">Hiện ${avail.length} model đầu; gõ để lọc.</div>` : ''}`,
        `<input id="model-filter" class="search" placeholder="Lọc theo tên / họ…" value="${esc(this.modelFilter)}">`);
    }
    if (this.modelTab === 'search') {
      const form = `<form class="form-row" id="hf-search"><input id="hf-q" placeholder="Tìm repo, ví dụ qwen2.5 instruct" value="${esc(this.searchQuery)}">
        <button class="btn btn-primary" type="submit">Tìm</button></form>`;
      if (!this.searchQuery) return card('Tìm trên Hugging Face', form);
      const r = await attempt(() => api.searchModels(this.searchQuery));
      return card('Tìm trên Hugging Face', form + (r.ok ? (r.v.length
        ? `<div class="scroll-x"><table class="data-table"><thead><tr><th>Repo</th><th>Loại</th><th class="right">Lượt tải</th><th class="right">Thích</th><th></th></tr></thead><tbody>
          ${r.v.map(x => `<tr><td class="mono">${esc(x.repo || x.name)}</td><td>${esc(x.pipeline_tag || '—')}</td>
            <td class="right">${(x.downloads ?? 0).toLocaleString()}</td><td class="right">${x.likes ?? 0}</td>
            <td class="right">${x.isDownloaded ? '<span class="pill pill-ok">✓ đã có</span>' : pullBtn(x.repo || x.name)}</td></tr>`).join('')}
          </tbody></table></div>` : '<div class="muted">Không có kết quả.</div>') : errorBox(r.e)));
    }
    if (this.modelTab === 'fit') {
      const r = await attempt(() => api.compatibility(this.fitPage, this.fitFilter));
      if (!r.ok) return errorBox(r.e);
      const c = r.v;
      const opts = ['', 'Perfect', 'Good', 'Marginal', 'Too Tight'];
      const fitPill = (f: string) => `<span class="pill pill-${f === 'Perfect' || f === 'Good' ? 'ok' : f === 'Marginal' ? 'warn' : 'error'}">${f === 'Perfect' || f === 'Good' ? '✓' : f === 'Marginal' ? '!' : '✕'} ${esc(f)}</span>`;
      return card(`Model vừa máy này (${c.total})`, `
        <div class="muted small">Ngân sách bộ nhớ ${num(c.budget_gb, 1)} GB · RAM ${num(c.hardware.ram_total_gb, 1)} GB (còn ${num(c.hardware.ram_available_gb, 1)} GB)
          ${c.hardware.gpu ? ` · ${esc(c.hardware.gpu)} ${num(c.hardware.gpu_vram_gb, 1)} GB` : ''}. Chia cho nhiều node thì chạy được model lớn hơn.</div>
        <div class="scroll-x"><table class="data-table"><thead><tr><th>Model</th><th>Tham số</th><th class="right">Cần</th><th>Mức vừa</th><th class="right">Ước tính</th><th></th></tr></thead><tbody>
          ${c.models.map(x => `<tr><td class="mono">${esc(x.name)}</td><td>${esc(x.parameters)}</td><td class="right">${num(x.memory_needed_gb, 1, ' GB')}</td>
            <td>${fitPill(x.fit)}</td><td class="right">${num(x.estimated_speed, 0, ' tok/s')}</td><td class="right">${pullBtn(x.name)}</td></tr>`).join('')}
        </tbody></table></div>
        <div class="pager"><button class="btn btn-glass btn-sm" data-fit-page="${c.page - 1}" ${c.page <= 1 ? 'disabled' : ''}>←</button>
          <span>Trang ${c.page} / ${c.total_pages}</span>
          <button class="btn btn-glass btn-sm" data-fit-page="${c.page + 1}" ${c.page >= c.total_pages ? 'disabled' : ''}>→</button></div>`,
        `<select id="fit-filter">${opts.map(o => `<option value="${o}" ${o === this.fitFilter ? 'selected' : ''}>${o || 'Mọi mức'}</option>`).join('')}</select>`);
    }
    const [cps, models] = await Promise.all([attempt(() => api.checkpoints()), attempt(() => api.models())]);
    const names = models.ok ? models.v.downloaded.map(m => m.name) : [];
    const list = cps.ok ? (cps.v.length
      ? `<table class="data-table"><thead><tr><th></th><th>Tên</th><th>Loại</th><th>Tạo lúc</th></tr></thead><tbody>
        ${cps.v.map(c => `<tr><td>${c.type === 'adapter' ? `<input type="checkbox" name="adapter" value="${esc(c.name)}" aria-label="Chọn ${esc(c.name)}">` : ''}</td>
          <td class="mono">${esc(c.name)}</td><td>${c.type === 'adapter' ? 'LoRA adapter' : 'model đầy đủ'}</td><td>${esc(String(c.created || ''))}</td></tr>`).join('')}
        </tbody></table>` : '<div class="muted">Chưa có checkpoint. Chạy một job ở trang Huấn luyện để tạo.</div>') : errorBox(cps.e);
    return card('Checkpoint', `<form id="merge-form">${list}
      ${cps.ok && cps.v.some(c => c.type === 'adapter') ? `<div class="form-grid">
        <label>Model gốc<input name="base_model" list="merge-models" required value="${esc(names[0] || '')}"></label>
        <datalist id="merge-models">${names.map(n => `<option value="${esc(n)}">`).join('')}</datalist>
        <label>Tên model sau khi gộp<input name="output_name" required placeholder="my-model-merged"></label>
        <div class="span-2"><button class="btn btn-primary" type="submit">Gộp adapter đã chọn vào model gốc</button></div>
      </div>` : ''}</form>`);
  }

  private bindModels() {
    const pull = async (name: string, full: boolean) => {
      if (!name) return;
      toast(`Đang tải ${name}…`);
      try { const r = await api.pullModel(name, full); toast(r.message || 'Đã tải'); this.show(); }
      catch (e) { toast(errMsg(e), 'error'); }
    };
    document.getElementById('pull-btn')!.onclick = () =>
      pull((document.getElementById('pull-name') as HTMLInputElement).value.trim(),
        (document.getElementById('pull-full') as HTMLInputElement).checked);
    this.content().querySelectorAll<HTMLElement>('[data-tab]').forEach(b => b.onclick = () => {
      this.modelTab = b.dataset.tab as typeof this.modelTab; this.show();
    });
    const rebindTab = async () => {
      const el = document.getElementById('model-tab');
      if (!el) return;
      el.innerHTML = '<div class="loading">Đang tải…</div>';
      el.innerHTML = await this.modelTabBody();
      this.bindModelTab(pull, rebindTab);
    };
    this.bindModelTab(pull, rebindTab);
    const tick = window.setInterval(() => this.updateDownloadProgress(), 2000);
    this.mounted.push({ destroy: () => clearInterval(tick) });
    this.updateDownloadProgress();
  }

  private bindModelTab(pull: (n: string, full: boolean) => void, rerender: () => void) {
    const root = document.getElementById('model-tab')!;
    root.querySelectorAll<HTMLElement>('[data-pull-model]').forEach(b => b.onclick = () => pull(b.dataset.pullModel!, false));
    root.querySelectorAll<HTMLElement>('[data-chat-model]').forEach(b => b.onclick = () => { this.chatModel = b.dataset.chatModel!; this.go('chat'); });
    root.querySelectorAll<HTMLElement>('[data-delete-model]').forEach(b => b.onclick = async () => {
      const name = b.dataset.deleteModel!;
      if (!confirm(`Xoá ${name} khỏi cache Hugging Face của máy này?`)) return;
      try { const r = await api.deleteModel(name); toast(r.message); rerender(); }
      catch (e) { toast(errMsg(e), 'error'); }
    });
    const filter = document.getElementById('model-filter') as HTMLInputElement | null;
    if (filter) filter.oninput = () => {
      this.modelFilter = filter.value;
      clearTimeout((filter as any)._t);
      (filter as any)._t = setTimeout(async () => {
        await rerender();
        const f = document.getElementById('model-filter') as HTMLInputElement;
        f?.focus(); f?.setSelectionRange(f.value.length, f.value.length);
      }, 250);
    };
    const search = document.getElementById('hf-search') as HTMLFormElement | null;
    if (search) search.onsubmit = e => { e.preventDefault(); this.searchQuery = (document.getElementById('hf-q') as HTMLInputElement).value.trim(); rerender(); };
    const fit = document.getElementById('fit-filter') as HTMLSelectElement | null;
    if (fit) fit.onchange = () => { this.fitFilter = fit.value; this.fitPage = 1; rerender(); };
    root.querySelectorAll<HTMLButtonElement>('[data-fit-page]').forEach(b => b.onclick = () => { this.fitPage = Number(b.dataset.fitPage); rerender(); });
    const merge = document.getElementById('merge-form') as HTMLFormElement | null;
    if (merge) merge.onsubmit = async e => {
      e.preventDefault();
      const fd = new FormData(merge);
      const adapters = fd.getAll('adapter').map(String);
      if (!adapters.length) { toast('Chọn ít nhất một adapter', 'error'); return; }
      toast('Đang gộp…');
      try {
        const r = await api.mergeModel(String(fd.get('base_model')), adapters, String(fd.get('output_name')));
        toast(r.message || `Đã gộp${r.output_path ? `: ${r.output_path}` : ''}`); rerender();
      } catch (err) { toast(errMsg(err), 'error'); }
    };
  }

  private async updateDownloadProgress() {
    const el = document.getElementById('download-progress');
    if (!el) return;
    const r = await attempt(() => api.downloadProgress());
    if (!r.ok) { el.innerHTML = ''; return; }
    const rows = Object.entries(r.v.progress || {});
    el.innerHTML = rows.map(([node, p]: [string, any]) => {
      const pct = p.total_bytes ? (100 * (p.downloaded_bytes || 0) / p.total_bytes) : 0;
      return `<div class="progress-row"><span>${esc(node)}: ${esc(p.repo_id || '')} ${esc(p.status || '')} ${pct ? `${pct.toFixed(0)}%` : ''}</span>
        <div class="progress-bar-container" role="progressbar" aria-valuenow="${pct.toFixed(0)}" aria-valuemin="0" aria-valuemax="100"><div class="progress-fill" style="width:${pct.toFixed(1)}%"></div></div></div>`;
    }).join('');
  }

  // -- training ----------------------------------------------------------------------
  private async viewTraining(): Promise<string> {
    const [models, datasets] = await Promise.all([attempt(() => api.models()), attempt(() => api.datasets())]);
    const names = models.ok ? models.v.downloaded.map(m => m.name) : [];
    const ds: Dataset[] = datasets.ok ? datasets.v : [];
    const multi = (live.dist?.partitions_count ?? 1) > 1;
    return `
      <div class="grid-2col">
        ${card('Bắt đầu huấn luyện', `
          ${datasets.ok ? '' : errorBox(datasets.e)}
          ${ds.length ? '' : '<div class="notice notice-warn">Chưa có dataset. Tạo hoặc tải lên ở trang <a data-nav="datasets">Dataset</a>.</div>'}
          <form id="train-form" class="form-grid">
            <label>Model<input name="model" list="train-models" required value="${esc(names[0] || '')}"></label>
            <datalist id="train-models">${names.map(n => `<option value="${esc(n)}">`).join('')}</datalist>
            <label>Dataset<select name="dataset" required>${ds.map(d => `<option value="${esc(d.path)}">${esc(d.name)} (${d.sample_count ?? '?'} mẫu)</option>`).join('')}</select></label>
            <label>Epoch<input name="epochs" type="number" min="1" value="1"></label>
            <label>Batch<input name="batch_size" type="number" min="1" value="1"></label>
            <label>Learning rate<input name="learning_rate" value="1e-4"></label>
            <label>Max length<input name="max_length" type="number" min="16" max="2048" value="128"></label>
            <label>Max step (0 = hết)<input name="max_steps" type="number" min="0" value="0"></label>
            <label>Warmup step<input name="warmup_steps" type="number" min="0" value="0"></label>
            <label>Lưu mỗi (step)<input name="save_every" type="number" min="0" value="0"></label>
            <label>Tên model đầu ra<input name="output_model_name" placeholder="tự đặt nếu để trống"></label>
            <label class="check span-2"><input type="checkbox" name="use_lora" checked> LoRA (chỉ train adapter, chạy được cả trên trọng số int8/int4)</label>
            <fieldset class="span-2" id="lora-fields">
              <label>r<input name="lora_r" type="number" min="1" value="8"></label>
              <label>alpha<input name="lora_alpha" type="number" min="1" value="16"></label>
              <label>dropout<input name="lora_dropout" type="number" step="0.01" min="0" max="0.9" value="0.05"></label>
              <label class="span-3">Module<input name="lora_target_modules" value="q_proj,k_proj,v_proj,o_proj"></label>
            </fieldset>
            <label class="check span-2"><input type="checkbox" name="distributed" ${multi ? 'checked' : ''} ${multi ? '' : 'disabled'}>
              Chia tải qua các node (pipeline)${multi ? '' : ' — cần cụm từ 2 node'}</label>
            <div class="span-2"><button class="btn btn-primary" type="submit" ${ds.length ? '' : 'disabled'}>Bắt đầu</button></div>
          </form>
          <div class="muted small">Mỗi node train đúng dải layer nó đang phục vụ; hidden state đi xuôi, gradient đi ngược giữa các node.</div>`)}
        <div>
          ${card('Trạng thái', '<div id="train-status"><div class="loading">Đang tải…</div></div>')}
          ${card('Loss theo bước', '<div id="ch-loss"></div>')}
        </div>
      </div>`;
  }

  private bindTraining() {
    const form = document.getElementById('train-form') as HTMLFormElement;
    const lora = form.elements.namedItem('use_lora') as HTMLInputElement;
    const fields = document.getElementById('lora-fields')!;
    const sync = () => { fields.style.display = lora.checked ? '' : 'none'; };
    lora.onchange = sync; sync();
    const loss = new LineChart(document.getElementById('ch-loss')!, { linearX: 'bước', digits: 3, empty: 'Chưa có loss' });
    this.mounted.push(loss);
    form.onsubmit = async e => {
      e.preventDefault();
      const fd = new FormData(form);
      const body: Record<string, unknown> = {
        model: fd.get('model'), dataset: fd.get('dataset'),
        epochs: Number(fd.get('epochs')), batch_size: Number(fd.get('batch_size')),
        learning_rate: String(fd.get('learning_rate') || ''), max_length: Number(fd.get('max_length')),
        max_steps: Number(fd.get('max_steps')), warmup_steps: Number(fd.get('warmup_steps')),
        save_every: Number(fd.get('save_every')), use_lora: lora.checked,
      };
      const out = String(fd.get('output_model_name') || '').trim();
      if (out) body.output_model_name = out;
      if (lora.checked) {
        body.lora_r = Number(fd.get('lora_r')); body.lora_alpha = Number(fd.get('lora_alpha'));
        body.lora_dropout = Number(fd.get('lora_dropout')); body.lora_target_modules = fd.get('lora_target_modules');
      }
      const dist = (form.elements.namedItem('distributed') as HTMLInputElement).checked;
      try {
        const r = await (dist ? api.startDistributedTraining(body) : api.startTraining(body));
        toast(r.message || 'Đã bắt đầu huấn luyện'); update();
      } catch (err) { toast(errMsg(err), 'error'); }
    };
    const update = async () => {
      const el = document.getElementById('train-status');
      if (!el) return;
      const r = await attempt(() => api.trainingStatus());
      if (!r.ok) { el.innerHTML = errorBox(r.e); return; }
      el.innerHTML = this.trainingStatus(r.v.job);
      const h = r.v.job?.loss_history || [];
      loss.update([{ id: 'loss', label: 'loss', color: seriesColor(1), points: h.map(([t, v]) => ({ t, v })) }]);
    };
    update();
    const t = window.setInterval(update, 2000);
    this.mounted.push({ destroy: () => clearInterval(t) });
  }

  private trainingStatus(j: TrainingJob | null): string {
    if (!j) return '<div class="muted">Chưa có job nào.</div>';
    const pct = j.progress ?? 0;
    return `
      <div class="kv-grid">
        <div><span>Trạng thái</span><b>${pill(j.status)}</b></div>
        <div><span>Kiểu</span><b>${j.use_lora ? 'LoRA' : 'Full'}</b></div>
        <div><span>Model</span><b class="mono">${esc(j.model)}</b></div>
        <div><span>Bước</span><b>${j.current_step ?? 0}${j.max_steps ? ` / ${j.max_steps}` : ''}</b></div>
        <div><span>Epoch</span><b>${(j.current_epoch ?? 0) + (j.status === 'completed' ? 0 : 1)} / ${j.epochs}</b></div>
        <div><span>Loss</span><b>${j.loss ?? '—'}</b></div>
      </div>
      <div class="progress-bar-container" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><div class="progress-fill" style="width:${pct}%"></div></div>
      <div class="muted small">${pct}% · ${esc(j.dataset)}${j.started_at ? ` · bắt đầu ${esc(j.started_at)}` : ''}${j.completed_at ? ` · xong lúc ${esc(j.completed_at)}` : ''}</div>
      ${j.error ? `<div class="notice notice-error">${esc(j.error)}</div>` : ''}`;
  }

  // -- datasets ------------------------------------------------------------------------
  private async viewDatasets(): Promise<string> {
    const r = await attempt(() => api.datasets());
    if (!r.ok) return errorBox(r.e);
    const ds = r.v;
    const p = this.datasetPreview;
    return `
      <div class="grid-3col">
        ${card('Tải lên', `<div class="form-row"><input type="file" id="ds-file" accept=".json,.jsonl" aria-label="Chọn file dataset"><button class="btn btn-primary" id="ds-upload">Tải lên</button></div>
          <div class="muted small">Mỗi dòng một mẫu: {"messages": [...]}, {"instruction", "output"} hoặc {"text"}.</div>`)}
        ${card('Tạo mới', `<form id="ds-create" class="form-grid one">
            <label>Tên file<input name="name" required placeholder="my-data.jsonl"></label>
            <label>Mẫu<select name="template"><option value="messages">Hội thoại (messages)</option><option value="alpaca">Instruction / output</option>
              <option value="text">Văn bản (text)</option><option value="empty">Trống</option></select></label>
            <button class="btn btn-primary" type="submit">Tạo</button></form>`)}
        ${card('Dùng file có sẵn', `<form id="ds-link" class="form-grid one">
            <label>Đường dẫn trên máy node<input name="path" required placeholder="D:\\data\\train.jsonl"></label>
            <label>Tên hiển thị<input name="name" placeholder="tuỳ chọn"></label>
            <button class="btn btn-glass" type="submit">Liên kết</button></form>`)}
      </div>
      ${card(`Dataset (${ds.length})`, ds.length ? `<div class="scroll-x"><table class="data-table"><thead><tr><th>Tên</th><th class="right">Mẫu</th><th class="right">Dung lượng</th><th>Tạo lúc</th><th></th></tr></thead><tbody>
          ${ds.map(d => `<tr><td class="mono">${esc(d.name)}${d.type === 'link' ? ' <span class="pill pill-muted">liên kết</span>' : ''}</td>
            <td class="right">${d.sample_count ?? '—'}</td><td class="right">${num(d.size_mb, 2, ' MB')}</td><td>${esc(d.created)}</td>
            <td class="right nowrap">
              <button class="btn btn-glass btn-sm" data-ds-act="preview" data-path="${esc(d.path)}" data-name="${esc(d.name)}">Xem</button>
              <button class="btn btn-glass btn-sm" data-ds-act="edit" data-path="${esc(d.path)}" data-name="${esc(d.name)}">Sửa</button>
              <button class="btn btn-glass btn-sm" data-ds-act="validate" data-path="${esc(d.path)}" data-name="${esc(d.name)}">Kiểm tra</button>
              <button class="btn btn-glass btn-sm" data-ds-act="download" data-path="${esc(d.path)}" data-name="${esc(d.name)}">Tải về</button>
              <button class="btn btn-glass btn-sm" data-ds-act="rename" data-path="${esc(d.path)}" data-name="${esc(d.name)}">Đổi tên</button>
              <button class="btn btn-danger btn-sm" data-ds-act="delete" data-path="${esc(d.path)}" data-name="${esc(d.name)}">Xoá</button></td></tr>`).join('')}
        </tbody></table></div>` : '<div class="muted">Chưa có dataset.</div>')}
      <div id="ds-validate"></div>
      ${p ? card(p.edit ? `Sửa ${esc(p.name)}` : `Xem trước ${esc(p.name)}`, p.edit
        ? `<textarea id="ds-editor" class="editor" spellcheck="false" aria-label="Nội dung dataset">${esc(p.lines.join('\n'))}</textarea>
           <div class="form-row"><button class="btn btn-primary" id="ds-save">Lưu</button><button class="btn btn-glass" id="ds-close">Đóng</button></div>`
        : `<pre class="terminal-window">${esc(p.lines.join('\n'))}</pre><div class="form-row"><button class="btn btn-glass" id="ds-close">Đóng</button></div>`) : ''}`;
  }

  private bindDatasets() {
    const run = async (fn: () => Promise<unknown>, ok: string, refresh = true) => {
      try { await fn(); toast(ok); if (refresh) this.show(); } catch (e) { toast(errMsg(e), 'error'); }
    };
    document.getElementById('ds-upload')!.onclick = () => {
      const f = (document.getElementById('ds-file') as HTMLInputElement).files?.[0];
      if (!f) { toast('Chọn file trước', 'error'); return; }
      run(() => api.uploadDataset(f), `Đã tải lên ${f.name}`);
    };
    const create = document.getElementById('ds-create') as HTMLFormElement;
    create.onsubmit = e => {
      e.preventDefault();
      const fd = new FormData(create);
      let name = String(fd.get('name')).trim();
      if (!/\.jsonl?$/.test(name)) name += '.jsonl';
      run(() => api.createDataset(name, String(fd.get('template'))), `Đã tạo ${name}`);
    };
    const link = document.getElementById('ds-link') as HTMLFormElement;
    link.onsubmit = e => {
      e.preventDefault();
      const fd = new FormData(link);
      run(() => api.linkDataset(String(fd.get('path')).trim(), String(fd.get('name') || '').trim()), 'Đã liên kết');
    };
    this.content().querySelectorAll<HTMLElement>('[data-ds-act]').forEach(b => b.onclick = async () => {
      const path = b.dataset.path!, name = b.dataset.name!;
      switch (b.dataset.dsAct) {
        case 'preview':
        case 'edit':
          try {
            const edit = b.dataset.dsAct === 'edit';
            this.datasetPreview = { path, name, edit, lines: await api.previewDataset(path, edit ? 100000 : 30) };
            await this.show();
            document.getElementById(edit ? 'ds-editor' : 'ds-close')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
          } catch (e) { toast(errMsg(e), 'error'); }
          break;
        case 'validate': {
          const el = document.getElementById('ds-validate')!;
          const r = await attempt(() => api.validateDataset(path));
          el.innerHTML = r.ok ? card(`Kiểm tra ${esc(name)}`, `
            <div class="kv-grid"><div><span>Định dạng</span><b>${esc(r.v.format_detected || '—')}</b></div>
              <div><span>Mẫu hợp lệ</span><b>${r.v.valid_count ?? 0}${r.v.total_lines !== undefined ? ` / ${r.v.total_lines}` : ''}</b></div>
              <div><span>Lỗi</span><b>${(r.v.errors || []).length}</b></div></div>
            ${(r.v.errors || []).length ? `<ul class="plain">${(r.v.errors as any[]).slice(0, 20).map(er => `<li>${typeof er === 'string' ? esc(er) : `Dòng ${esc(er.line)}: ${esc(er.message)}`}</li>`).join('')}</ul>` : '<div class="notice notice-ok">✓ Không có lỗi.</div>'}`)
            : errorBox(r.e);
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
          break;
        }
        case 'download':
          try { await api.downloadDataset(path, name); } catch (e) { toast(errMsg(e), 'error'); }
          break;
        case 'rename': {
          const nn = prompt(`Tên mới cho ${name}:`, name);
          if (nn && nn !== name) run(() => api.renameDataset(path, nn), 'Đã đổi tên');
          break;
        }
        case 'delete':
          if (confirm(`Xoá dataset ${name}?`)) { this.datasetPreview = null; run(() => api.deleteDataset(path), 'Đã xoá'); }
          break;
      }
    });
    document.getElementById('ds-close')?.addEventListener('click', () => { this.datasetPreview = null; this.show(); });
    document.getElementById('ds-save')?.addEventListener('click', () => {
      const p = this.datasetPreview!;
      const content = (document.getElementById('ds-editor') as HTMLTextAreaElement).value;
      run(() => api.updateDataset(p.path, content), 'Đã lưu', false);
    });
  }

  // -- logs ------------------------------------------------------------------------------
  private async viewLogs(): Promise<string> {
    const act = await attempt(() => api.activity());
    return `
      ${card('Hoạt động', `<div class="scroll-x">${act.ok ? this.activityTable(act.v.slice().reverse()) : errorBox(act.e)}</div>`)}
      ${card('Log node', '<pre class="terminal-window tall" id="term">Đang tải…</pre>', `<div class="form-row tight">
          <input id="log-filter" class="search" placeholder="Lọc dòng…" value="${esc(this.logFilter)}">
          <label class="check"><input type="checkbox" id="log-follow" ${this.logFollow ? 'checked' : ''}> Theo dõi</label></div>`)}`;
  }

  private bindLogs() {
    const term = document.getElementById('term')!;
    const filter = document.getElementById('log-filter') as HTMLInputElement;
    const follow = document.getElementById('log-follow') as HTMLInputElement;
    let lines: string[] = [];
    const render = () => {
      const f = this.logFilter.toLowerCase();
      term.textContent = (f ? lines.filter(l => l.toLowerCase().includes(f)) : lines).slice(-600).join('\n') || '(trống)';
      if (this.logFollow) term.scrollTop = term.scrollHeight;
    };
    const load = async () => {
      const r = await attempt(() => api.terminalLogs());
      if (!r.ok) { term.outerHTML = errorBox(r.e); return; }
      lines = r.v;
      render();
    };
    filter.oninput = () => { this.logFilter = filter.value; render(); };
    follow.onchange = () => { this.logFollow = follow.checked; render(); };
    load();
    const t = window.setInterval(() => { if (this.logFollow) load(); }, 3000);
    this.mounted.push({ destroy: () => clearInterval(t) });
  }

  // -- settings ---------------------------------------------------------------------------
  private async viewSettings(): Promise<string> {
    const [s, p] = await Promise.all([attempt(() => api.settings()), attempt(() => api.pairing())]);
    const sel = (name: string, cur: string, opts: Array<[string, string]>) =>
      `<select name="${name}">${opts.map(([v, l]) => `<option value="${v}"${v === cur ? ' selected' : ''}>${l}</option>`).join('')}</select>`;
    const settings = s.ok ? (() => {
      const v: Settings = s.v;
      return `<form id="settings-form" class="form-grid">
        <label class="span-2">Model mặc định<input name="default_model" value="${esc(v.default_model)}"></label>
        <label>Chia layer theo${sel('lacp_priority', v.lacp_priority, [['throughput', 'Thông lượng (nhiều request)'], ['latency', 'Độ trễ (một request)']])}</label>
        <label>Lượng tử trọng số${sel('load_quantization', v.load_quantization, [['auto', 'Tự động (khi thiếu bộ nhớ)'], ['none', 'Không'], ['int8', 'int8'], ['int4', 'int4']])}</label>
        <label>Truyền hidden state${sel('transport_quantization', v.transport_quantization, [['int8', 'int8 theo hàng'], ['none', 'Chính xác (fp32)']])}</label>
        <label>Draft lookahead<input name="lookahead_window" type="number" min="0" max="16" value="${v.lookahead_window}"></label>
        <label>Tìm node${sel('discovery_module', v.discovery_module, [['udp', 'UDP (LAN)'], ['tailscale', 'Tailscale'], ['manual', 'File cấu hình']])}</label>
        <div class="span-2"><button class="btn btn-primary" type="submit">Lưu</button>
          <span class="muted small">Đổi cách tìm node sẽ khởi động lại node.</span></div>
      </form>`;
    })() : errorBox(s.e);

    const pairing = p.ok ? (() => {
      const v: Pairing = p.v;
      return `
        <div class="kv-grid"><div><span>TLS</span><b>${esc(v.tls_mode)}</b></div><div><span>Đã ghép</span><b>${v.paired ? 'có' : 'chưa'}</b></div></div>
        ${v.error ? `<div class="notice notice-error">${esc(v.error)}</div>` : ''}
        ${v.restart_needed ? '<div class="notice notice-warn">Khởi động lại node để áp dụng thay đổi ghép máy.</div>' : ''}
        ${v.code ? `<div class="code-box"><span class="mono" id="pair-code">${esc(v.code)}</span>
            <button class="btn btn-glass btn-sm" id="pair-copy">Sao chép</button></div>
          <div class="muted small">Nhập mã này ở trang Cài đặt của các máy khác. Máy không có mã không kết nối được với cụm.</div>
          <div class="form-row"><button class="btn btn-danger" id="pair-forget">Bỏ ghép (về plaintext)</button></div>`
        : `<div class="muted small">Không bắt buộc trên LAN tin cậy: các node tự tìm nhau. Ghép máy để mã hoá đường truyền (Tailscale, mạng lạ).</div>
           <div class="form-row"><button class="btn btn-primary" id="pair-create">Tạo mã cho cụm mới</button></div>
           <div class="form-row"><input id="pair-join-code" placeholder="XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX" aria-label="Mã ghép máy">
           <button class="btn btn-glass" id="pair-join">Tham gia cụm</button></div>`}`;
    })() : errorBox(p.e);

    const key = getApiKey();
    const hw = this.info?.hardware;
    return `
      <div class="grid-2col">
        <div>
          ${card('Node', settings)}
          ${card('Máy này', hw ? `<div class="kv-grid">
              <div><span>CPU</span><b>${esc(hw.cpu)} (${hw.cpu_cores} nhân)</b></div><div><span>RAM</span><b>${num(hw.ram_total_gb, 1, ' GB')}</b></div>
              <div><span>GPU</span><b>${esc(hw.gpu || 'không')}</b></div><div><span>Backend</span><b>${esc(hw.backend)}</b></div>
              <div><span>Hệ điều hành</span><b>${esc(hw.os)}</b></div><div><span>Tìm node</span><b>${esc(this.info?.network_discovery || '—')}</b></div>
            </div>` : '<div class="muted">Chưa đọc được.</div>')}
        </div>
        <div>
          ${card('Ghép máy (TLS)', pairing)}
          ${card('API key', `
            <div class="muted small">Node đặt SYNAPSE_API_KEY thì mọi lệnh cần key. Không đặt key thì các thao tác quản trị chỉ được phép
              từ chính máy chạy node. Key chỉ lưu trong trình duyệt này.</div>
            <div class="form-row"><input id="api-key" type="password" value="${esc(key)}" placeholder="SYNAPSE_API_KEY" aria-label="API key">
              <button class="btn btn-glass" id="api-key-save">Lưu</button>${key ? '<button class="btn btn-danger" id="api-key-clear">Xoá</button>' : ''}</div>`)}
          ${card('Tắt node', `<div class="muted small">Dừng node trên máy này. Các node khác sẽ chia lại layer không có nó.</div>
            <div class="form-row"><button class="btn btn-danger" id="node-quit">Tắt node</button></div>`)}
        </div>
      </div>`;
  }

  private bindSettings() {
    const form = document.getElementById('settings-form') as HTMLFormElement | null;
    if (form) form.onsubmit = async e => {
      e.preventDefault();
      const fd = new FormData(form);
      try {
        const r = await api.saveSettings({
          default_model: String(fd.get('default_model') || ''), lacp_priority: String(fd.get('lacp_priority')),
          load_quantization: String(fd.get('load_quantization')), transport_quantization: String(fd.get('transport_quantization')),
          lookahead_window: Number(fd.get('lookahead_window')), discovery_module: String(fd.get('discovery_module')),
        });
        toast(r.restarted ? 'Đã lưu; node đang khởi động lại' : 'Đã lưu');
      } catch (err) { toast(errMsg(err), 'error'); }
    };
    const act = async (fn: () => Promise<unknown>, ok: string) => {
      try { await fn(); toast(ok); this.show(); } catch (e) { toast(errMsg(e), 'error'); }
    };
    const byId = (id: string) => document.getElementById(id);
    byId('pair-create')?.addEventListener('click', () => act(() => api.pair('create'), 'Đã tạo mã; khởi động lại node để bật TLS'));
    byId('pair-join')?.addEventListener('click', () =>
      act(() => api.pair('join', (byId('pair-join-code') as HTMLInputElement).value), 'Đã lưu mã; khởi động lại node'));
    byId('pair-forget')?.addEventListener('click', () => {
      if (confirm('Bỏ ghép? Node sẽ nói chuyện plaintext sau khi khởi động lại.')) act(() => api.pair('forget'), 'Đã bỏ ghép');
    });
    byId('pair-copy')?.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(byId('pair-code')!.textContent || ''); toast('Đã sao chép'); }
      catch { toast('Không sao chép được: hãy chọn mã và copy tay', 'error'); }
    });
    byId('api-key-save')?.addEventListener('click', () => { setApiKey((byId('api-key') as HTMLInputElement).value.trim()); toast('Đã lưu key'); this.show(); });
    byId('api-key-clear')?.addEventListener('click', () => { setApiKey(''); toast('Đã xoá key'); this.show(); });
    byId('node-quit')?.addEventListener('click', async () => {
      if (!confirm('Tắt node trên máy này?')) return;
      try { await api.quit(); } catch { /* the connection drops as the node exits */ }
      toast('Đã gửi lệnh tắt node');
    });
  }
}

new App();
