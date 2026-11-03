import './style.css';
import { api, ApiError, getApiKey, setApiKey, streamChat } from './api';
import type {
  ActivityEntry, ClusterResources, Dataset, ModelsList, Pairing,
  Settings, SystemInfo, TrainingStatus,
} from './types';

type Page = 'dashboard' | 'cluster' | 'chat' | 'models' | 'training' | 'datasets' | 'logs' | 'settings';

const PAGES: Array<{ id: Page; title: string; subtitle: string; section: string; icon: string }> = [
  { id: 'dashboard', section: 'Cụm', title: 'Tổng quan', subtitle: 'Tài nguyên máy này và trạng thái cụm', icon: '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>' },
  { id: 'cluster', section: 'Cụm', title: 'Node & chia layer', subtitle: 'Mỗi node giữ một dải layer; hidden state đi node này sang node kế', icon: '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>' },
  { id: 'chat', section: 'Model', title: 'Chat', subtitle: 'Gửi prompt qua cụm, đo TTFT và tok/s thật', icon: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>' },
  { id: 'models', section: 'Model', title: 'Model', subtitle: 'Mỗi node chỉ tải phần trọng số của dải layer nó giữ', icon: '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>' },
  { id: 'training', section: 'Model', title: 'Huấn luyện', subtitle: 'Mỗi node train layer của mình (full hoặc LoRA)', icon: '<path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>' },
  { id: 'datasets', section: 'Model', title: 'Dataset', subtitle: 'File .json / .jsonl để huấn luyện', icon: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>' },
  { id: 'logs', section: 'Hệ thống', title: 'Log', subtitle: 'Hoạt động và log của node', icon: '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>' },
  { id: 'settings', section: 'Hệ thống', title: 'Cài đặt', subtitle: 'Cài đặt node, ghép máy (TLS), API key', icon: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>' },
];

const LIVE_PAGES: Page[] = ['dashboard', 'cluster', 'training', 'logs', 'models'];

function esc(v: unknown): string {
  return String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}

function num(v: number | null | undefined, digits = 1, unit = ''): string {
  return v === null || v === undefined || Number.isNaN(v) ? '—' : `${v.toFixed(digits)}${unit}`;
}

function errorBox(e: unknown): string {
  if (e instanceof ApiError && e.forbidden) {
    return `<div class="notice notice-warn">Mục này chỉ mở được từ chính máy chạy node, hoặc khi đã nhập API key
      (<a data-nav="settings">Cài đặt</a>). Chi tiết: ${esc(e.message)}</div>`;
  }
  return `<div class="notice notice-error">${esc(e instanceof Error ? e.message : e)}</div>`;
}

/** Run a loader; a failure becomes an error box instead of breaking the page. */
async function attempt<T>(fn: () => Promise<T>): Promise<{ ok: true; v: T } | { ok: false; e: unknown }> {
  try { return { ok: true, v: await fn() }; } catch (e) { return { ok: false, e }; }
}

function toast(msg: string, kind: 'ok' | 'error' = 'ok') {
  const el = document.createElement('div');
  el.className = `toast toast-${kind}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

interface ChatMsg { role: 'user' | 'ai'; text: string; meta?: string; error?: boolean }

class App {
  private page: Page = (location.hash.slice(1) as Page) || 'dashboard';
  private info: SystemInfo | null = null;

  private chat: ChatMsg[] = [];
  private chatModel = '';
  private chatAbort: AbortController | null = null;
  private modelFilter = '';
  private datasetPreview: { path: string; lines: string[] } | null = null;
  private busy = false;

  constructor() {
    if (!PAGES.some(p => p.id === this.page)) this.page = 'dashboard';
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
    setInterval(() => {
      if (LIVE_PAGES.includes(this.page) && !this.busy && document.visibilityState === 'visible') this.refreshLive();
    }, 4000);
    setInterval(() => this.refreshFooter(), 8000);
    this.refreshFooter();
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
      return `${head}<a class="nav-item ${p.id === this.page ? 'active' : ''}" data-nav="${p.id}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">${p.icon}</svg>${p.title}</a>`;
    }).join('');
    app.innerHTML = `
      <aside class="sidebar">
        <div class="brand">
          <div class="brand-icon">⚡</div>
          <div class="brand-info"><h1>SYNAPSE AI</h1><span class="badge" id="brand-version">v${esc(this.info?.version || '…')}</span></div>
        </div>
        <div class="nav-menu">${nav}</div>
        <div class="sidebar-footer" id="sidebar-footer">Đang kết nối…</div>
      </aside>
      <div class="main-wrapper">
        <header class="top-nav">
          <div class="nav-left"><div>
            <div class="page-title">${meta.title}</div>
            <div class="page-subtitle">${meta.subtitle}</div>
          </div></div>
          <div class="nav-right">
            <button class="btn btn-glass" id="btn-refresh">Làm mới</button>
            ${this.page !== 'chat' ? '<button class="btn btn-primary" data-nav="chat">Chat</button>' : ''}
          </div>
        </header>
        <div class="content-container" id="content"><div class="loading">Đang tải…</div></div>
      </div>`;
    document.getElementById('btn-refresh')!.onclick = () => this.show();
  }

  private async refreshFooter() {
    const r = await attempt(() => api.distributed());
    const el = document.getElementById('sidebar-footer');
    if (!el) return;
    if (!r.ok) { el.innerHTML = `<div class="cluster-pill offline"><div class="cluster-dot"></div>Mất kết nối node</div>`; return; }

    const d = r.v;
    el.innerHTML = `
      <div class="cluster-pill"><div class="cluster-dot"></div>${d.partitions_count} node đang giữ layer</div>
      <div>Node: <b>${esc(d.node_id || '?')}</b>${d.is_master ? ' (master)' : ''}</div>
      <div>Master: ${esc(d.master_id || '—')}</div>
      <div>TLS: ${esc(d.tls_mode || 'off')}${d.api_key_required ? ' · API key' : ''}</div>`;
    if (!this.info) {
      const i = await attempt(() => api.systemInfo());
      if (i.ok) { this.info = i.v; const b = document.getElementById('brand-version'); if (b) b.textContent = `v${i.v.version}`; }
    }
  }

  private content(): HTMLElement { return document.getElementById('content')!; }

  private async show() {
    const page = this.page;
    const html = await this.view(page);
    if (page !== this.page) return; // navigated away meanwhile
    this.content().innerHTML = html;
    this.bind(page);
  }

  private async refreshLive() {
    const page = this.page;
    if (page === 'training') { await this.updateTrainingStatus(); return; }
    if (page === 'models') { await this.updateDownloadProgress(); return; }
    const html = await this.view(page);
    if (page === this.page && !this.busy) { this.content().innerHTML = html; this.bind(page); }
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
    if (page === 'chat') this.bindChat();
    if (page === 'models') this.bindModels();
    if (page === 'training') this.bindTraining();
    if (page === 'datasets') this.bindDatasets();
    if (page === 'settings') this.bindSettings();
  }

  // -- dashboard --------------------------------------------------------------
  private async viewDashboard(): Promise<string> {
    const [stats, cluster, dist, act] = await Promise.all([
      attempt(() => api.systemStats()), attempt(() => api.cluster()),
      attempt(() => api.distributed()), attempt(() => api.activity()),
    ]);
    const kpi = (title: string, value: string, meta: string) =>
      `<div class="kpi-card glass-panel"><div class="kpi-header"><span class="kpi-title">${title}</span></div>
       <div class="kpi-value">${value}</div><div class="kpi-meta">${meta}</div></div>`;
    let top = '';
    if (stats.ok) {
      const s = stats.v;
      top = `<div class="kpi-grid">
        ${kpi('CPU', num(s.cpu, 0, '%'), `${s.process_count ?? '—'} tiến trình`)}
        ${kpi('RAM', `${num(s.memory_used_gb, 1)} / ${num(s.memory_total_gb, 1)} GB`, num(s.memory, 0, '%'))}
        ${kpi('GPU', esc(s.gpu_name || s.gpu || '—'), s.gpu_utilization === null ? 'không đo được mức dùng' : num(s.gpu_utilization, 0, '%'))}
        ${kpi('Uptime máy', esc(s.uptime), `Disk ${num(s.disk_percent, 0, '%')}`)}
      </div>`;
    } else top = errorBox(stats.e);

    let clusterBox = '';
    if (cluster.ok) {
      const c = cluster.v;
      const d = dist.ok ? dist.v : null;
      clusterBox = `<div class="glass-panel"><div class="card-title-bar"><h3>Cụm</h3><a class="link" data-nav="cluster">Chi tiết →</a></div>
        <div class="card-body">
          <div class="kv-grid">
            <div><span>Node</span><b>${c.total_nodes}</b></div>
            <div><span>Tổng layer</span><b>${c.total_layers}</b></div>
            <div><span>Master</span><b>${esc(d?.master_id || '—')}</b></div>
            <div><span>Chia layer theo</span><b>${esc(d?.partition_objective || '—')}</b></div>
            <div><span>Autoscale</span><b>${esc(d?.autoscale_mode || '—')}</b></div>
            <div><span>TLS</span><b>${esc(d?.tls_mode || '—')}</b></div>
          </div>
          ${this.layerBar(c)}
        </div></div>`;
    } else clusterBox = errorBox(cluster.e);

    const actBox = `<div class="glass-panel"><div class="card-title-bar"><h3>Hoạt động gần đây</h3><a class="link" data-nav="logs">Tất cả →</a></div>
      <div class="card-body">${act.ok ? this.activityTable(act.v.slice(-8).reverse()) : errorBox(act.e)}</div></div>`;
    return `${top}<div class="grid-2col">${clusterBox}${actBox}</div>`;
  }

  private layerBar(c: ClusterResources): string {
    const total = c.total_layers || 1;
    const nodes = c.nodes.filter(n => n.layer_start !== null).sort((a, b) => (a.layer_start! - b.layer_start!));
    if (!nodes.length) return '<div class="muted">Chưa có node nào giữ layer.</div>';
    const colors = ['var(--primary)', 'var(--tertiary)', 'var(--accent-green)', 'var(--accent-amber)', 'var(--accent-cyan)'];
    return `<div class="layer-bar">${nodes.map((n, i) => {
      const w = ((n.layer_end! - n.layer_start! + 1) / total) * 100;
      return `<div class="layer-seg" style="width:${w}%;background:${colors[i % colors.length]}"
        title="${esc(n.name)}: layer ${n.layer_start}–${n.layer_end}">${esc(n.name)} · ${n.layer_start}–${n.layer_end}</div>`;
    }).join('')}</div>`;
  }

  private activityTable(rows: ActivityEntry[]): string {
    if (!rows.length) return '<div class="muted">Chưa có hoạt động.</div>';
    return `<table class="data-table"><thead><tr><th>Giờ</th><th>Việc</th><th>Model</th><th>Kết quả</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td>${esc(r.time)}</td><td>${esc(r.action)}</td><td>${esc(r.model)}</td>
        <td><span class="pill pill-${r.status === 'success' ? 'ok' : r.status === 'failed' ? 'error' : 'muted'}">${esc(r.status)}</span></td></tr>`).join('')}
    </tbody></table>`;
  }

  // -- cluster ------------------------------------------------------------------
  private async viewCluster(): Promise<string> {
    const [cluster, dist] = await Promise.all([attempt(() => api.cluster()), attempt(() => api.distributed())]);
    if (!cluster.ok) return errorBox(cluster.e);
    const c = cluster.v;
    const d = dist.ok ? dist.v : null;
    const stages = c.nodes.filter(n => n.layer_start !== null).sort((a, b) => a.layer_start! - b.layer_start!);
    const flow = stages.length ? `<div class="pipeline-flow-diagram">${stages.map((n, i) => `
        ${i ? '<div class="pipeline-arrow">→</div>' : ''}
        <div class="pipeline-node-box">
          <div class="node-box-header"><span class="node-name">${esc(n.name)}</span>
            ${n.node_id === d?.master_id ? '<span class="node-role-badge">master</span>' : ''}</div>
          <div class="node-shard-badge">layer ${n.layer_start}–${n.layer_end}</div>
          <div class="node-stat-row"><span>${esc(n.chip)}</span></div>
        </div>`).join('')}</div>` : '<div class="muted">Chưa có node nào giữ layer.</div>';

    const table = `<table class="data-table"><thead><tr><th>Node</th><th>Địa chỉ</th><th>Chip</th><th>RAM</th><th>VRAM</th>
        <th>CPU</th><th>Layer</th><th>Tỉ lệ</th></tr></thead><tbody>
      ${c.nodes.map(n => `<tr>
        <td><b>${esc(n.name)}</b>${n.is_self ? ' <span class="pill pill-muted">máy này</span>' : ''}
          ${d?.standby_nodes?.includes(n.node_id) ? ' <span class="pill pill-warn">dự phòng</span>' : ''}</td>
        <td class="mono">${esc(n.ip)}</td><td>${esc(n.chip)}</td>
        <td>${n.ram_used_gb !== null ? `${num(n.ram_used_gb)} / ` : ''}${num(n.ram_total_gb ?? n.ram_gb)} GB</td>
        <td>${num(n.vram_gb)} GB</td><td>${num(n.cpu_pct, 0, '%')}</td>
        <td class="mono">${n.layer_start === null ? '—' : `${n.layer_start}–${n.layer_end}`}</td>
        <td>${num(n.share_pct, 0, '%')}</td></tr>`).join('')}
    </tbody></table>`;

    const ids = Object.keys(c.latency_matrix || {});
    const lat = ids.length ? `<table class="data-table"><thead><tr><th></th>${ids.map(i => `<th>${esc(i)}</th>`).join('')}</tr></thead><tbody>
      ${ids.map(a => `<tr><th>${esc(a)}</th>${ids.map(b => `<td class="mono">${a === b ? '—' : num(c.latency_matrix[a]?.[b], 2, ' ms')}</td>`).join('')}</tr>`).join('')}
      </tbody></table>` : '<div class="muted">Chưa đo độ trễ giữa các node (cụm một node).</div>';

    const hints = d?.autoscale_hints?.length
      ? d.autoscale_hints.map(h => `<li>${esc(h.action)} ${esc(h.node || '')} (${esc(h.reason)})</li>`).join('') : '<li class="muted">Không có</li>';
    const transport = d?.transport_counts && Object.keys(d.transport_counts).length
      ? Object.entries(d.transport_counts).map(([k, v]) => `<li><span class="mono">${esc(k)}</span>: ${v}</li>`).join('')
      : '<li class="muted">Chưa có lượt truyền</li>';
    const peers = d?.peer_versions && Object.keys(d.peer_versions).length
      ? Object.entries(d.peer_versions).map(([k, v]) => `<li>${esc(k)}: bridge v${esc(v.bridge_version)} · ${esc(v.synapse_version)}</li>`).join('')
      : '<li class="muted">Chưa có peer</li>';

    return `
      <div class="glass-panel"><div class="card-title-bar"><h3>Pipeline</h3><span class="muted">${esc(c.strategy_name)} · ${c.total_layers} layer</span></div>
        <div class="card-body">${flow}${this.layerBar(c)}</div></div>
      <div class="glass-panel"><div class="card-title-bar"><h3>Node (${c.total_nodes})</h3></div><div class="card-body scroll-x">${table}</div></div>
      <div class="grid-2col">
        <div class="glass-panel"><div class="card-title-bar"><h3>Độ trễ đo được</h3></div><div class="card-body scroll-x">${lat}</div></div>
        <div class="glass-panel"><div class="card-title-bar"><h3>Điều phối</h3></div><div class="card-body">
          ${d ? `<div class="kv-grid">
            <div><span>Master</span><b>${esc(d.master_id || '—')}</b></div>
            <div><span>Chia layer theo</span><b>${esc(d.partition_objective || '—')}</b></div>
            <div><span>Autoscale</span><b>${esc(d.autoscale_mode || '—')}${d.autoscale_objective ? ` → ${esc(d.autoscale_objective)}` : ''}</b></div>
            <div><span>Lượng tử</span><b>${esc(d.quant_active || 'none')}</b></div>
          </div>
          ${d.recovery_degraded ? `<div class="notice notice-warn">Phục hồi checkpoint thiếu trọng số: ${esc(JSON.stringify(d.recovery_degraded))}</div>` : ''}
          <h4>Gợi ý autoscale</h4><ul class="plain">${hints}</ul>
          <h4>Đường truyền</h4><ul class="plain">${transport}</ul>
          <h4>Phiên bản peer</h4><ul class="plain">${peers}</ul>` : errorBox((dist as any).e)}
        </div></div>
      </div>`;
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
      <div class="chat-container glass-panel">
        <div class="chat-toolbar">
          <label>Model <input list="chat-models" id="chat-model" value="${esc(this.chatModel)}" placeholder="repo id trên Hugging Face"></label>
          <datalist id="chat-models">${names.map(n => `<option value="${esc(n)}">`).join('')}</datalist>
          <button class="btn btn-glass" id="chat-clear">Xoá hội thoại</button>
        </div>
        <div class="chat-messages" id="chat-messages">${bubbles}</div>
        <div class="chat-input-bar">
          <textarea id="chat-input" rows="2" placeholder="Nhập prompt… (Enter gửi, Shift+Enter xuống dòng)"></textarea>
          <button class="btn btn-primary" id="chat-send">${this.chatAbort ? 'Dừng' : 'Gửi'}</button>
        </div>
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
    const box = document.getElementById('chat-messages')!;
    box.scrollTop = box.scrollHeight;
    modelEl.onchange = () => { this.chatModel = modelEl.value.trim(); };
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
      this.chatAbort = new AbortController();
      this.busy = true;
      this.renderChatMessages();
      send.textContent = 'Dừng';
      try {
        const st = await streamChat(this.chatModel, history, delta => {
          reply.text += delta;
          this.updateLastBubble(reply);
        }, this.chatAbort.signal);
        const secs = st.totalMs / 1000;
        const tps = st.tokens && st.ttftMs !== null && secs * 1000 > st.ttftMs
          ? st.tokens / ((st.totalMs - st.ttftMs) / 1000) : null;
        reply.meta = `TTFT ${st.ttftMs === null ? '—' : Math.round(st.ttftMs) + ' ms'} · ${st.tokens || '?'} token trong ${secs.toFixed(1)} s`
          + (tps ? ` · ${tps.toFixed(1)} tok/s` : '');
      } catch (e) {
        if ((e as Error).name === 'AbortError') reply.meta = 'Đã dừng';
        else { reply.error = true; reply.text = reply.text || (e instanceof Error ? e.message : String(e)); }
      } finally {
        this.chatAbort = null;
        this.busy = false;
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
    const r = await attempt(() => api.models());
    if (!r.ok) return errorBox(r.e);
    const m: ModelsList = r.v;
    const f = this.modelFilter.toLowerCase();
    const avail = m.available.filter(x => !f || x.name.toLowerCase().includes(f) || (x.family || '').toLowerCase().includes(f)).slice(0, 60);
    return `
      <div class="glass-panel"><div class="card-title-bar"><h3>Tải model</h3></div><div class="card-body">
        <div class="form-row">
          <input id="pull-name" placeholder="repo id, ví dụ HuggingFaceTB/SmolLM2-135M-Instruct">
          <label class="check"><input type="checkbox" id="pull-full"> Tải cả repo</label>
          <button class="btn btn-primary" id="pull-btn">Tải</button>
        </div>
        <div class="muted small">Mặc định chỉ tải config, tokenizer và các file trọng số chứa dải layer node này đang giữ.
          Các node khác tự tải phần của chúng khi được giao layer.</div>
        <div id="download-progress"></div>
      </div></div>
      <div class="glass-panel"><div class="card-title-bar"><h3>Đã có trên máy này (${m.downloaded.length})</h3></div><div class="card-body scroll-x">
        ${m.downloaded.length ? `<table class="data-table"><thead><tr><th>Model</th><th>Tham số</th><th>Kiến trúc</th><th></th></tr></thead><tbody>
          ${m.downloaded.map(x => `<tr><td class="mono">${esc(x.name)}</td><td>${esc(x.parameter_size || '—')}</td><td>${esc(x.architecture || '—')}</td>
            <td class="right"><button class="btn btn-glass btn-sm" data-chat-model="${esc(x.name)}">Chat</button>
            <button class="btn btn-danger btn-sm" data-delete-model="${esc(x.name)}">Xoá</button></td></tr>`).join('')}
        </tbody></table>` : '<div class="muted">Chưa có model nào trong cache.</div>'}
      </div></div>
      <div class="glass-panel"><div class="card-title-bar"><h3>Danh mục (${m.available.length})</h3>
        <input id="model-filter" class="search" placeholder="Lọc theo tên / họ…" value="${esc(this.modelFilter)}"></div>
        <div class="card-body scroll-x"><table class="data-table"><thead><tr><th>Model</th><th>Tham số</th><th>Bộ nhớ</th><th>Họ</th><th></th></tr></thead><tbody>
          ${avail.map(x => `<tr><td class="mono">${esc(x.name)}</td><td>${esc(x.parameter_size || '—')}</td><td>${esc(x.memory || '—')}</td>
            <td>${esc(x.family || '—')}</td><td class="right"><button class="btn btn-glass btn-sm" data-pull-model="${esc(x.name)}">Tải</button></td></tr>`).join('')}
        </tbody></table>${m.available.length > avail.length ? `<div class="muted small">Hiện ${avail.length} model đầu; gõ để lọc.</div>` : ''}</div></div>`;
  }

  private bindModels() {
    const pull = async (name: string, full: boolean) => {
      if (!name) return;
      this.busy = true;
      toast(`Đang tải ${name}…`);
      try { const r = await api.pullModel(name, full); toast(r.message || 'Đã tải'); this.show(); }
      catch (e) { toast(e instanceof Error ? e.message : String(e), 'error'); }
      finally { this.busy = false; }
    };
    document.getElementById('pull-btn')!.onclick = () =>
      pull((document.getElementById('pull-name') as HTMLInputElement).value.trim(),
        (document.getElementById('pull-full') as HTMLInputElement).checked);
    const filter = document.getElementById('model-filter') as HTMLInputElement;
    filter.oninput = () => {
      this.modelFilter = filter.value;
      clearTimeout((filter as any)._t);
      (filter as any)._t = setTimeout(async () => { await this.show(); const f = document.getElementById('model-filter') as HTMLInputElement; f?.focus(); f?.setSelectionRange(f.value.length, f.value.length); }, 250);
    };
    this.content().querySelectorAll<HTMLElement>('[data-pull-model]').forEach(b => b.onclick = () => pull(b.dataset.pullModel!, false));
    this.content().querySelectorAll<HTMLElement>('[data-chat-model]').forEach(b => b.onclick = () => { this.chatModel = b.dataset.chatModel!; this.go('chat'); });
    this.content().querySelectorAll<HTMLElement>('[data-delete-model]').forEach(b => b.onclick = async () => {
      const name = b.dataset.deleteModel!;
      if (!confirm(`Xoá ${name} khỏi cache Hugging Face của máy này?`)) return;
      try { const r = await api.deleteModel(name); toast(r.message); this.show(); }
      catch (e) { toast(e instanceof Error ? e.message : String(e), 'error'); }
    });
    this.updateDownloadProgress();
  }

  private async updateDownloadProgress() {
    const el = document.getElementById('download-progress');
    if (!el) return;
    const r = await attempt(() => api.downloadProgress());
    if (!r.ok) { el.innerHTML = ''; return; }
    const rows = Object.entries(r.v.progress || {});
    el.innerHTML = rows.length ? rows.map(([node, p]: [string, any]) => {
      const pct = p.total_bytes ? (100 * (p.downloaded_bytes || 0) / p.total_bytes) : 0;
      return `<div class="progress-row"><span>${esc(node)}: ${esc(p.repo_id || '')} ${esc(p.status || '')}</span>
        <div class="progress-bar-container"><div class="progress-fill" style="width:${pct.toFixed(1)}%"></div></div></div>`;
    }).join('') : '';
  }

  // -- training ----------------------------------------------------------------------
  private async viewTraining(): Promise<string> {
    const [models, datasets] = await Promise.all([attempt(() => api.models()), attempt(() => api.datasets())]);
    const names = models.ok ? models.v.downloaded.map(m => m.name) : [];
    const ds: Dataset[] = datasets.ok ? datasets.v : [];
    return `
      <div class="grid-2col">
        <div class="glass-panel"><div class="card-title-bar"><h3>Bắt đầu huấn luyện</h3></div><div class="card-body">
          ${datasets.ok ? '' : errorBox(datasets.e)}
          <form id="train-form" class="form-grid">
            <label>Model<input name="model" list="train-models" required value="${esc(names[0] || '')}"></label>
            <datalist id="train-models">${names.map(n => `<option value="${esc(n)}">`).join('')}</datalist>
            <label>Dataset<select name="dataset" required>${ds.map(d => `<option value="${esc(d.path)}">${esc(d.name)} (${d.sample_count ?? '?'} mẫu)</option>`).join('')}</select></label>
            <label>Epoch<input name="epochs" type="number" min="1" value="1"></label>
            <label>Batch<input name="batch_size" type="number" min="1" value="1"></label>
            <label>Learning rate<input name="learning_rate" value="1e-4"></label>
            <label>Max length<input name="max_length" type="number" min="16" max="2048" value="128"></label>
            <label>Max step (0 = hết)<input name="max_steps" type="number" min="0" value="0"></label>
            <label class="check span-2"><input type="checkbox" name="use_lora" checked> LoRA (chỉ train adapter, chạy được cả trên trọng số int8/int4)</label>
            <fieldset class="span-2" id="lora-fields">
              <label>r<input name="lora_r" type="number" min="1" value="8"></label>
              <label>alpha<input name="lora_alpha" type="number" min="1" value="16"></label>
              <label>dropout<input name="lora_dropout" type="number" step="0.01" min="0" max="0.9" value="0.05"></label>
              <label class="span-3">Module<input name="lora_target_modules" value="q_proj,k_proj,v_proj,o_proj"></label>
            </fieldset>
            <div class="span-2"><button class="btn btn-primary" type="submit">Bắt đầu</button></div>
          </form>
          <div class="muted small">Mỗi node train đúng dải layer nó đang phục vụ; hidden state đi xuôi, gradient đi ngược giữa các node.</div>
        </div></div>
        <div class="glass-panel"><div class="card-title-bar"><h3>Trạng thái</h3></div><div class="card-body" id="train-status">Đang tải…</div></div>
      </div>`;
  }

  private bindTraining() {
    const form = document.getElementById('train-form') as HTMLFormElement;
    const lora = form.elements.namedItem('use_lora') as HTMLInputElement;
    const fields = document.getElementById('lora-fields')!;
    const sync = () => { fields.style.display = lora.checked ? '' : 'none'; };
    lora.onchange = sync; sync();
    form.onsubmit = async e => {
      e.preventDefault();
      const fd = new FormData(form);
      const body: Record<string, unknown> = {
        model: fd.get('model'), dataset: fd.get('dataset'),
        epochs: Number(fd.get('epochs')), batch_size: Number(fd.get('batch_size')),
        learning_rate: String(fd.get('learning_rate') || ''), max_length: Number(fd.get('max_length')),
        max_steps: Number(fd.get('max_steps')), use_lora: lora.checked,
      };
      if (lora.checked) {
        body.lora_r = Number(fd.get('lora_r')); body.lora_alpha = Number(fd.get('lora_alpha'));
        body.lora_dropout = Number(fd.get('lora_dropout')); body.lora_target_modules = fd.get('lora_target_modules');
      }
      try { await api.startTraining(body); toast('Đã bắt đầu huấn luyện'); this.updateTrainingStatus(); }
      catch (err) { toast(err instanceof Error ? err.message : String(err), 'error'); }
    };
    this.updateTrainingStatus();
  }

  private async updateTrainingStatus() {
    const el = document.getElementById('train-status');
    if (!el) return;
    const r = await attempt(() => api.trainingStatus());
    if (!r.ok) { el.innerHTML = errorBox(r.e); return; }
    const t: TrainingStatus = r.v;
    const j = t.job;
    if (!j) { el.innerHTML = '<div class="muted">Chưa có job nào.</div>'; return; }
    const pct = j.progress ?? 0;
    el.innerHTML = `
      <div class="kv-grid">
        <div><span>Trạng thái</span><b><span class="pill pill-${j.status === 'completed' ? 'ok' : j.status === 'failed' ? 'error' : 'warn'}">${esc(j.status)}</span></b></div>
        <div><span>Kiểu</span><b>${j.use_lora ? 'LoRA' : 'Full'}</b></div>
        <div><span>Model</span><b class="mono">${esc(j.model)}</b></div>
        <div><span>Bước</span><b>${j.current_step ?? 0}</b></div>
        <div><span>Epoch</span><b>${(j.current_epoch ?? 0) + (j.status === 'completed' ? 0 : 1)} / ${j.epochs}</b></div>
        <div><span>Loss</span><b>${j.loss ?? '—'}</b></div>
      </div>
      <div class="progress-bar-container"><div class="progress-fill" style="width:${pct}%"></div></div>
      <div class="muted small">${pct}% · ${esc(j.dataset)}${j.completed_at ? ` · xong lúc ${esc(j.completed_at)}` : ''}</div>
      ${j.error ? `<div class="notice notice-error">${esc(j.error)}</div>` : ''}`;
  }

  // -- datasets ------------------------------------------------------------------------
  private async viewDatasets(): Promise<string> {
    const r = await attempt(() => api.datasets());
    if (!r.ok) return errorBox(r.e);
    const ds = r.v;
    return `
      <div class="glass-panel"><div class="card-title-bar"><h3>Tải lên</h3></div><div class="card-body">
        <div class="form-row"><input type="file" id="ds-file" accept=".json,.jsonl"><button class="btn btn-primary" id="ds-upload">Tải lên</button></div>
        <div class="muted small">Mỗi dòng một mẫu: {"messages": [...]}, {"instruction", "output"} hoặc {"text"}.</div>
      </div></div>
      <div class="glass-panel"><div class="card-title-bar"><h3>Dataset (${ds.length})</h3></div><div class="card-body scroll-x">
        ${ds.length ? `<table class="data-table"><thead><tr><th>Tên</th><th>Mẫu</th><th>Dung lượng</th><th>Tạo lúc</th><th></th></tr></thead><tbody>
          ${ds.map(d => `<tr><td class="mono">${esc(d.name)}</td><td>${d.sample_count ?? '—'}</td><td>${num(d.size_mb, 2, ' MB')}</td><td>${esc(d.created)}</td>
            <td class="right"><button class="btn btn-glass btn-sm" data-preview="${esc(d.path)}">Xem</button>
            <button class="btn btn-danger btn-sm" data-delete-ds="${esc(d.path)}" data-name="${esc(d.name)}">Xoá</button></td></tr>`).join('')}
        </tbody></table>` : '<div class="muted">Chưa có dataset.</div>'}
      </div></div>
      ${this.datasetPreview ? `<div class="glass-panel"><div class="card-title-bar"><h3>Xem trước</h3></div>
        <div class="card-body"><pre class="terminal-window">${esc(this.datasetPreview.lines.join('\n'))}</pre></div></div>` : ''}`;
  }

  private bindDatasets() {
    document.getElementById('ds-upload')!.onclick = async () => {
      const f = (document.getElementById('ds-file') as HTMLInputElement).files?.[0];
      if (!f) { toast('Chọn file trước', 'error'); return; }
      try { const r = await api.uploadDataset(f); toast(`Đã tải lên ${r.name}`); this.show(); }
      catch (e) { toast(e instanceof Error ? e.message : String(e), 'error'); }
    };
    this.content().querySelectorAll<HTMLElement>('[data-preview]').forEach(b => b.onclick = async () => {
      try { this.datasetPreview = { path: b.dataset.preview!, lines: await api.previewDataset(b.dataset.preview!) }; this.show(); }
      catch (e) { toast(e instanceof Error ? e.message : String(e), 'error'); }
    });
    this.content().querySelectorAll<HTMLElement>('[data-delete-ds]').forEach(b => b.onclick = async () => {
      if (!confirm(`Xoá dataset ${b.dataset.name}?`)) return;
      try { await api.deleteDataset(b.dataset.deleteDs!); toast('Đã xoá'); this.datasetPreview = null; this.show(); }
      catch (e) { toast(e instanceof Error ? e.message : String(e), 'error'); }
    });
  }

  // -- logs ------------------------------------------------------------------------------
  private async viewLogs(): Promise<string> {
    const [act, term] = await Promise.all([attempt(() => api.activity()), attempt(() => api.terminalLogs())]);
    return `
      <div class="glass-panel"><div class="card-title-bar"><h3>Hoạt động</h3></div>
        <div class="card-body scroll-x">${act.ok ? this.activityTable(act.v.slice().reverse()) : errorBox(act.e)}</div></div>
      <div class="glass-panel"><div class="card-title-bar"><h3>Log node</h3></div><div class="card-body">
        ${term.ok ? `<pre class="terminal-window" id="term">${esc(term.v.slice(-400).join('\n'))}</pre>` : errorBox(term.e)}
      </div></div>`;
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
        : `<div class="form-row"><button class="btn btn-primary" id="pair-create">Tạo mã cho cụm mới</button></div>
           <div class="form-row"><input id="pair-join-code" placeholder="XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX">
           <button class="btn btn-glass" id="pair-join">Tham gia cụm</button></div>`}`;
    })() : errorBox(p.e);

    const key = getApiKey();
    return `
      <div class="grid-2col">
        <div class="glass-panel"><div class="card-title-bar"><h3>Node</h3></div><div class="card-body">${settings}</div></div>
        <div>
          <div class="glass-panel"><div class="card-title-bar"><h3>Ghép máy (TLS)</h3></div><div class="card-body">${pairing}</div></div>
          <div class="glass-panel"><div class="card-title-bar"><h3>API key</h3></div><div class="card-body">
            <div class="muted small">Node đặt SYNAPSE_API_KEY thì mọi lệnh cần key. Không đặt key thì các thao tác quản trị chỉ được phép
              từ chính máy chạy node. Key chỉ lưu trong trình duyệt này.</div>
            <div class="form-row"><input id="api-key" type="password" value="${esc(key)}" placeholder="SYNAPSE_API_KEY">
              <button class="btn btn-glass" id="api-key-save">Lưu</button>${key ? '<button class="btn btn-danger" id="api-key-clear">Xoá</button>' : ''}</div>
          </div></div>
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
      } catch (err) { toast(err instanceof Error ? err.message : String(err), 'error'); }
    };
    const act = async (fn: () => Promise<unknown>, ok: string) => {
      try { await fn(); toast(ok); this.show(); } catch (e) { toast(e instanceof Error ? e.message : String(e), 'error'); }
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
  }
}

new App();
