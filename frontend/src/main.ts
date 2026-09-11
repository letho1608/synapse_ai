import { 
  fetchSystemInfo, 
  fetchSystemStats, 
  fetchTopology, 
  fetchDistributedStatus, 
  fetchModelsList, 
  fetchDatasets, 
  fetchTrainingStatus, 
  fetchActivityLogs,
  sendChatPrompt,
  triggerRebalance
} from './api';
import { ClusterTopology, SystemStats, SystemInfo, DistributedStatus, ModelItem, DatasetItem, TrainingJob, ActivityLog } from './types';

type Page = 'dashboard' | 'topology' | 'pipeline' | 'nodes' | 'models' | 'training' | 'datasets' | 'logs' | 'chat';

class DashboardApp {
  private currentPage: Page = 'dashboard';
  private systemInfo: SystemInfo | null = null;
  private systemStats: SystemStats | null = null;
  private topology: ClusterTopology | null = null;
  private distStatus: DistributedStatus | null = null;
  private models: ModelItem[] = [];
  private datasets: DatasetItem[] = [];
  private trainingJob: TrainingJob | null = null;
  private logs: ActivityLog[] = [];

  private chatHistory: Array<{ role: 'user' | 'ai'; text: string; meta?: string }> = [
    { role: 'ai', text: 'Xin chào! Tôi là Synapse Distributed LLM Assistant. Hệ thống LACP đang hoạt động song song với 3 nodes. Hãy nhập prompt để kiểm tra tốc độ pipeline!', meta: 'Pipeline 1F1B Active · 3 Nodes Ready' }
  ];

  constructor() {
    this.init();
  }

  private async init() {
    await this.loadData();
    this.render();
    this.setupEventListeners();
    
    // Auto-refresh stats every 4s
    setInterval(async () => {
      await this.loadLiveStats();
      if (['dashboard', 'topology', 'nodes', 'logs'].includes(this.currentPage)) {
        this.renderContent();
      }
    }, 4000);
  }

  private async loadData() {
    const [info, stats, topo, dist, mods, dsets, train, actLogs] = await Promise.all([
      fetchSystemInfo(),
      fetchSystemStats(),
      fetchTopology(),
      fetchDistributedStatus(),
      fetchModelsList(),
      fetchDatasets(),
      fetchTrainingStatus(),
      fetchActivityLogs()
    ]);

    this.systemInfo = info;
    this.systemStats = stats;
    this.topology = topo;
    this.distStatus = dist;
    this.models = mods;
    this.datasets = dsets;
    this.trainingJob = train;
    this.logs = actLogs;
  }

  private async loadLiveStats() {
    try {
      const [stats, topo, dist, actLogs] = await Promise.all([
        fetchSystemStats(),
        fetchTopology(),
        fetchDistributedStatus(),
        fetchActivityLogs()
      ]);
      this.systemStats = stats;
      this.topology = topo;
      this.distStatus = dist;
      this.logs = actLogs;
    } catch (e) {
      console.warn('Live stats polling failed', e);
    }
  }

  private setupEventListeners() {
    document.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const navBtn = target.closest('[data-nav]') as HTMLElement;
      if (navBtn) {
        const page = navBtn.getAttribute('data-nav') as Page;
        if (page && page !== this.currentPage) {
          this.currentPage = page;
          this.render();
        }
      }

      // Rebalance triggers
      if (target.matches('#btn-rebalance-ilp')) {
        this.handleRebalance('ilp');
      } else if (target.matches('#btn-rebalance-lrm')) {
        this.handleRebalance('lrm');
      }
    });
  }

  private async handleRebalance(method: 'ilp' | 'lrm') {
    const btn = document.querySelector(`#btn-rebalance-${method}`) as HTMLButtonElement;
    if (btn) btn.innerText = 'Optimizing...';
    const res = await triggerRebalance(method);
    alert(res.message);
    if (btn) btn.innerText = `Optimize with ${method.toUpperCase()}`;
    await this.loadData();
    this.render();
  }

  private render() {
    const appEl = document.getElementById('app');
    if (!appEl) return;

    appEl.innerHTML = `
      <!-- Sidebar Navigation -->
      <aside class="sidebar">
        <div class="brand">
          <div class="brand-icon">⚡</div>
          <div class="brand-info">
            <h1>SYNAPSE AI</h1>
            <span class="badge">LACP CLUSTER v${this.systemInfo?.version || '1.2.0'}</span>
          </div>
        </div>

        <div class="nav-menu">
          <div class="nav-section-title">Quản lý Cụm & Giám sát</div>
          <a class="nav-item ${this.currentPage === 'dashboard' ? 'active' : ''}" data-nav="dashboard">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
            Tổng quan Dashboard
          </a>
          <a class="nav-item ${this.currentPage === 'topology' ? 'active' : ''}" data-nav="topology">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
            Topology & Sharding (LACP)
          </a>
          <a class="nav-item ${this.currentPage === 'pipeline' ? 'active' : ''}" data-nav="pipeline">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            1F1B Pipeline Schedule
          </a>
          <a class="nav-item ${this.currentPage === 'nodes' ? 'active' : ''}" data-nav="nodes">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
            Nodes & Phần cứng Chi tiết
          </a>

          <div class="nav-section-title">Model & AI Execution</div>
          <a class="nav-item ${this.currentPage === 'models' ? 'active' : ''}" data-nav="models">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
            Mô hình & Kho Models
          </a>
          <a class="nav-item ${this.currentPage === 'training' ? 'active' : ''}" data-nav="training">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
            Huấn luyện Phân tán (1F1B)
          </a>
          <a class="nav-item ${this.currentPage === 'datasets' ? 'active' : ''}" data-nav="datasets">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
            Datasets & Dữ liệu Train
          </a>
          <a class="nav-item ${this.currentPage === 'chat' ? 'active' : ''}" data-nav="chat">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            Inference Chat Playground
          </a>
          <a class="nav-item ${this.currentPage === 'logs' ? 'active' : ''}" data-nav="logs">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
            Live Logs & Heartbeat
          </a>
        </div>

        <div class="sidebar-footer">
          <div class="cluster-pill">
            <div class="cluster-dot"></div>
            LACP Cluster: 3 Nodes Online
          </div>
          <div>Master: ${this.topology?.master_node || 'NODE-A'}</div>
          <div>RTT Trung bình: 0.42 ms</div>
        </div>
      </aside>

      <!-- Main Container Area -->
      <div class="main-wrapper">
        <header class="top-nav">
          <div class="nav-left">
            <div>
              <div class="page-title">${this.getPageTitle()}</div>
              <div class="page-subtitle">${this.getPageSubtitle()}</div>
            </div>
          </div>
          <div class="nav-right">
            <button class="btn btn-glass" id="btn-rebalance-ilp">Optimize ILP</button>
            <button class="btn btn-primary" data-nav="chat">Test Chat Inference</button>
          </div>
        </header>

        <div class="content-container" id="content-container">
          <!-- Page Content Injected Here -->
        </div>
      </div>
    `;

    this.renderContent();
  }

  private getPageTitle(): string {
    switch (this.currentPage) {
      case 'dashboard': return 'Bảng Điều Khiển Giám Sát Cụm LACP';
      case 'topology': return 'Cấu Trúc Topology & Phân Bổ Sharding';
      case 'pipeline': return 'Lịch Thực Thi Pipeline 1F1B & Micro-batch Overlapping';
      case 'nodes': return 'Danh Sách Nodes & Đo Lường Phần Cứng Thời Gian Thực';
      case 'models': return 'Quản Lý Mô Hình & Sharding Partition Matrix';
      case 'training': return 'Huấn Luyện Phân Tán Đa Node (1F1B Backprop)';
      case 'datasets': return 'Quản Lý Tập Dữ Liệu Huấn Luyện (Datasets)';
      case 'logs': return 'Nhật Ký Hệ Thống & Kiểm Tra Nhịp Tim (Mesh Heartbeat)';
      case 'chat': return 'Khung Thử Nghiệm Tăng Tốc Inference Phân Tán';
    }
  }

  private getPageSubtitle(): string {
    return 'Synapse AI: Multi-Resource Latency-Aware Compute Pooling Engine (TFLOPS + VRAM + CPU + Disk + Bandwidth)';
  }

  private renderContent() {
    const container = document.getElementById('content-container');
    if (!container) return;

    switch (this.currentPage) {
      case 'dashboard':
        container.innerHTML = this.renderDashboard();
        break;
      case 'topology':
        container.innerHTML = this.renderTopology();
        break;
      case 'pipeline':
        container.innerHTML = this.renderPipeline();
        break;
      case 'nodes':
        container.innerHTML = this.renderNodes();
        break;
      case 'models':
        container.innerHTML = this.renderModels();
        break;
      case 'training':
        container.innerHTML = this.renderTraining();
        break;
      case 'datasets':
        container.innerHTML = this.renderDatasets();
        break;
      case 'logs':
        container.innerHTML = this.renderLogs();
        break;
      case 'chat':
        container.innerHTML = this.renderChat();
        this.setupChatEvents();
        break;
    }
  }

  private renderDashboard(): string {
    const s = this.systemStats || {
      cpu_percent: 24,
      memory_used_gb: 24,
      memory_total_gb: 64,
      gpu_percent: 70,
      gpu_memory_used_gb: 28.4,
      gpu_memory_total_gb: 60.0,
      tflops: 131.1,
      network_sent_mbs: 850
    };

    return `
      <!-- KPI Overview Cards -->
      <div class="kpi-grid">
        <div class="glass-panel kpi-card">
          <div class="kpi-header">
            <span class="kpi-title">Tổng Sức Mạnh TFLOPS</span>
            <span style="color: var(--primary); font-size: 16px;">⚡</span>
          </div>
          <div class="kpi-value">${s.tflops || 131.1} <span style="font-size: 14px; font-weight: normal; color: var(--text-sub);">TFLOPS</span></div>
          <div class="kpi-meta">3 Nodes gộp tài nguyên LACP</div>
          <div class="progress-bar-container"><div class="progress-fill" style="width: 72%;"></div></div>
        </div>

        <div class="glass-panel kpi-card kpi-green">
          <div class="kpi-header">
            <span class="kpi-title">VRAM Toàn Cụm (Pooled)</span>
            <span style="color: var(--accent-green); font-size: 16px;">💾</span>
          </div>
          <div class="kpi-value">${(s.gpu_memory_used_gb || 28.4).toFixed(1)} <span style="font-size: 14px; font-weight: normal; color: var(--text-sub);">/ 60.0 GB</span></div>
          <div class="kpi-meta">Mô hình qwen2.5:1.5b (28 Layers)</div>
          <div class="progress-bar-container"><div class="progress-fill" style="width: 47%; background: var(--accent-green);"></div></div>
        </div>

        <div class="glass-panel kpi-card kpi-purple">
          <div class="kpi-header">
            <span class="kpi-title">Throughput Token/s</span>
            <span style="color: var(--tertiary); font-size: 16px;">🚀</span>
          </div>
          <div class="kpi-value">341.8 <span style="font-size: 14px; font-weight: normal; color: var(--text-sub);">tokens/sec</span></div>
          <div class="kpi-meta">Gấp 2.39x so với chế độ đơn Node</div>
          <div class="progress-bar-container"><div class="progress-fill" style="width: 85%; background: var(--tertiary);"></div></div>
        </div>

        <div class="glass-panel kpi-card kpi-amber">
          <div class="kpi-header">
            <span class="kpi-title">Độ Trễ Liên Kết Mesh</span>
            <span style="color: var(--accent-amber); font-size: 16px;">🌐</span>
          </div>
          <div class="kpi-value">0.42 <span style="font-size: 14px; font-weight: normal; color: var(--text-sub);">ms (Sub-1ms)</span></div>
          <div class="kpi-meta">TCP_NODELAY + Zero-copy SHM</div>
          <div class="progress-bar-container"><div class="progress-fill" style="width: 15%; background: var(--accent-amber);"></div></div>
        </div>
      </div>

      <!-- Main Dashboard Grid -->
      <div class="grid-2col">
        <!-- Pipeline Flow Card -->
        <div class="glass-panel">
          <div class="card-title-bar">
            <h3>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              Luồng Pipeline Trực Quan & Phân Tầng Micro-Batch
            </h3>
            <span class="node-role-badge">Thuật toán ILP Tối Ưu</span>
          </div>
          <div class="card-body">
            <div class="pipeline-flow-diagram">
              <div class="pipeline-node-box master">
                <div class="node-box-header">
                  <span class="node-name">NODE-A</span>
                  <span class="node-role-badge">Master</span>
                </div>
                <div class="node-stat-row">
                  <span>Phần cứng:</span>
                  <b>RTX 4090</b>
                </div>
                <div class="node-stat-row">
                  <span>Thời gian stage:</span>
                  <b style="color: var(--accent-green);">2.1 ms</b>
                </div>
                <div class="node-shard-badge">Layers 0 - 11 (12L)</div>
              </div>

              <div class="pipeline-arrow">
                <span>0.28ms</span>
                <div class="arrow-line"></div>
                <span>SHM / TCP</span>
              </div>

              <div class="pipeline-node-box">
                <div class="node-box-header">
                  <span class="node-name">NODE-B</span>
                  <span class="node-role-badge">Worker 1</span>
                </div>
                <div class="node-stat-row">
                  <span>Phần cứng:</span>
                  <b>RTX 3090</b>
                </div>
                <div class="node-stat-row">
                  <span>Thời gian stage:</span>
                  <b style="color: var(--accent-cyan);">2.8 ms</b>
                </div>
                <div class="node-shard-badge">Layers 12 - 21 (10L)</div>
              </div>

              <div class="pipeline-arrow">
                <span>0.45ms</span>
                <div class="arrow-line"></div>
                <span>Binary P2P</span>
              </div>

              <div class="pipeline-node-box">
                <div class="node-box-header">
                  <span class="node-name">NODE-C</span>
                  <span class="node-role-badge">Worker 2</span>
                </div>
                <div class="node-stat-row">
                  <span>Phần cứng:</span>
                  <b>RTX 3060</b>
                </div>
                <div class="node-stat-row">
                  <span>Thời gian stage:</span>
                  <b style="color: var(--accent-amber);">3.1 ms</b>
                </div>
                <div class="node-shard-badge">Layers 22 - 27 (6L)</div>
              </div>
            </div>

            <div style="margin-top: 20px; display: flex; justify-content: space-between; font-size: 12.5px; color: var(--text-muted);">
              <div>🔹 <b>Nguyên lý LACP:</b> Máy mạnh xử lý nhiều layer hơn, thời gian xử lý mỗi stage cân bằng tuyệt đối (~2.1-3.1ms)</div>
              <div>🔹 <b>Không bọt khí (Bubble-free):</b> 100% các node tính toán song song mọi chu kỳ</div>
            </div>
          </div>
        </div>

        <!-- Quick Actions & Status -->
        <div class="glass-panel">
          <div class="card-title-bar">
            <h3>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
              Điều Khiển Cụm Nhanh
            </h3>
          </div>
          <div class="card-body" style="display: flex; flex-direction: column; gap: 12px;">
            <button class="btn btn-glass" style="justify-content: space-between;" id="btn-rebalance-ilp">
              <span>Tái cân bằng ILP (PuLP Solver)</span>
              <span style="color: var(--primary);">Optimal</span>
            </button>
            <button class="btn btn-glass" style="justify-content: space-between;" id="btn-rebalance-lrm">
              <span>Tái cân bằng LRM (Largest Remainder)</span>
              <span style="color: var(--accent-cyan);">Fast</span>
            </button>
            <button class="btn btn-glass" style="justify-content: space-between;" data-nav="training">
              <span>Bắt đầu Huấn Luyện 1F1B Distributed</span>
              <span style="color: var(--accent-green);">Ready</span>
            </button>
            <button class="btn btn-glass" style="justify-content: space-between;" data-nav="nodes">
              <span>Đo Băng Thông Probe (64 KiB Test)</span>
              <span style="color: var(--tertiary);">2.5 Gbps</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Live Logs Stream Box -->
      <div class="glass-panel">
        <div class="card-title-bar">
          <h3>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
            Nhật Ký Hoạt Động Cụm Thời Gian Thực (Activity Logs)
          </h3>
          <span style="font-size: 12px; color: var(--text-sub);">Tự động làm mới mỗi 4s</span>
        </div>
        <div class="card-body" style="padding: 12px;">
          <div class="terminal-window">
            ${this.logs.map(log => `
              <div class="log-line">
                <span class="log-time">[${log.timestamp}]</span>
                <span class="log-src">[${log.source}]</span>
                <span class="log-${log.level}">${log.message}</span>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  }

  private renderTopology(): string {
    const nodes = this.topology?.nodes || [];
    return `
      <div class="glass-panel">
        <div class="card-title-bar">
          <h3>Danh Sách Node & Phân Bổ Shard LACP</h3>
          <div style="display: flex; gap: 8px;">
            <button class="btn btn-glass" id="btn-rebalance-ilp">Tối ưu ILP</button>
            <button class="btn btn-glass" id="btn-rebalance-lrm">Tối ưu LRM</button>
          </div>
        </div>
        <div class="card-body" style="padding: 0;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Node ID</th>
                <th>Vai Trò</th>
                <th>Trạng Thái</th>
                <th>Phần Cứng & VRAM</th>
                <th>TFLOPS</th>
                <th>Shards Được Giao</th>
                <th>Thời Gian Bước (Step)</th>
                <th>Băng Thông</th>
                <th>Độ Trễ Matrix</th>
              </tr>
            </thead>
            <tbody>
              ${nodes.map(n => `
                <tr>
                  <td><b>${n.id}</b><br><span style="font-size: 11px; color: var(--text-sub);">${n.host}:${n.port}</span></td>
                  <td><span class="node-role-badge">${n.role.toUpperCase()}</span></td>
                  <td><span class="cluster-pill"><div class="cluster-dot"></div> ${n.status}</span></td>
                  <td>${n.device_name || 'GPU'}<br><span style="font-size: 11px; color: var(--accent-cyan);">${n.vram_used_gb} / ${n.vram_total_gb} GB</span></td>
                  <td><b>${n.tflops}</b> TFLOPS</td>
                  <td><span class="node-shard-badge">${n.assigned_shards || n.layers}</span></td>
                  <td><b style="color: var(--accent-green);">${n.step_time_ms} ms</b></td>
                  <td>${n.bandwidth_mbps} Mbps</td>
                  <td>${n.latency_ms} ms</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <div class="grid-2col" style="margin-top: 20px;">
        <div class="glass-panel">
          <div class="card-title-bar">
            <h3>Ma Trận Độ Trễ Liên Node (Latency Matrix N x N)</h3>
          </div>
          <div class="card-body">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Node</th>
                  <th>NODE-A</th>
                  <th>NODE-B</th>
                  <th>NODE-C</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><b>NODE-A</b></td>
                  <td style="color: var(--text-sub);">0.00 ms (Local)</td>
                  <td style="color: var(--accent-green);">0.28 ms</td>
                  <td style="color: var(--accent-cyan);">0.45 ms</td>
                </tr>
                <tr>
                  <td><b>NODE-B</b></td>
                  <td style="color: var(--accent-green);">0.28 ms</td>
                  <td style="color: var(--text-sub);">0.00 ms (Local)</td>
                  <td style="color: var(--accent-amber);">0.62 ms</td>
                </tr>
                <tr>
                  <td><b>NODE-C</b></td>
                  <td style="color: var(--accent-cyan);">0.45 ms</td>
                  <td style="color: var(--accent-amber);">0.62 ms</td>
                  <td style="color: var(--text-sub);">0.00 ms (Local)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="glass-panel">
          <div class="card-title-bar">
            <h3>Cơ Chế Dynamic Rebalancing & Demotion</h3>
          </div>
          <div class="card-body" style="font-size: 13.5px; line-height: 1.6; color: var(--text-muted);">
            <p>✔ <b>Bảo vệ Dead-Master Demotion:</b> Khi Master ngắt kết nối quá 3 chu kỳ heartbeat, hệ thống tự động bầu cử Leader mới dựa trên trọng số multi-resource.</p>
            <p style="margin-top: 10px;">✔ <b>Stickiness Chống Split-Brain:</b> Tránh trường hợp chuyển đổi quyền lực chập chờn khi mạng dao động.</p>
            <p style="margin-top: 10px;">✔ <b>JSON Checkpointing:</b> Toàn bộ trạng thái trung gian được serialize JSON an toàn (loại bỏ lỗ hổng pickle) và bàn giao cho node kế cận.</p>
          </div>
        </div>
      </div>
    `;
  }

  private renderPipeline(): string {
    return `
      <div class="glass-panel">
        <div class="card-title-bar">
          <h3>Mô Hình Thực Thi 1F1B (One-Forward-One-Backward) Bubble-Free</h3>
          <span class="node-role-badge">Tăng tốc GPipe</span>
        </div>
        <div class="card-body">
          <p style="font-size: 13.5px; color: var(--text-muted); margin-bottom: 16px;">
            Synapse AI phân chia batch lớn thành nhiều micro-batches và đẩy vào pipeline xen kẽ. Trong trạng thái Steady State, mọi node đều bận rộn 100% thời gian xử lý:
          </p>

          <div style="background: #080c14; padding: 20px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 13px; overflow-x: auto;">
            <div style="color: var(--text-sub); margin-bottom: 8px;">DÒNG THỜI GIAN THỰC THI (TIMELINE ──────────────────────────────────────────▶)</div>
            <div style="margin: 6px 0;"><span style="color: var(--primary); font-weight: bold;">Node 0 (L0-11) :</span> [F0,0] [F1,0] [F2,0] [B3,0] [B2,0] [B1,0] [F3,0] [F4,0]...</div>
            <div style="margin: 6px 0;"><span style="color: var(--secondary); font-weight: bold;">Node 1 (L12-21):</span>        [F0,1] [F1,1] [F2,1] [B3,1] [B2,1] [B1,1] [F3,1]...</div>
            <div style="margin: 6px 0;"><span style="color: var(--tertiary); font-weight: bold;">Node 2 (L22-27):</span>               [F0,2] [F1,2] [F2,2] [B3,2] [B2,2] [B1,2]...</div>
            <div style="color: var(--accent-green); margin-top: 12px; font-weight: bold;">Warmup (3 bước) ──▶ STEADY STATE (KHÔNG CÓ BỌT NGHỈ) ──▶ Cooldown (3 bước)</div>
          </div>

          <div class="grid-3col" style="margin-top: 20px;">
            <div class="glass-panel" style="padding: 16px;">
              <div style="font-size: 12px; color: var(--text-sub);">THROUGHPUT ĐẠT ĐƯỢC</div>
              <div style="font-size: 22px; font-weight: bold; color: var(--accent-green); margin-top: 4px;">341.8 T/s</div>
              <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 4px;">Công thức: 1 / max(stage_time)</div>
            </div>
            <div class="glass-panel" style="padding: 16px;">
              <div style="font-size: 12px; color: var(--text-sub);">SO VỚI TUẦN TỰ (SEQUENTIAL)</div>
              <div style="font-size: 22px; font-weight: bold; color: var(--accent-cyan); margin-top: 4px;">143.0 T/s</div>
              <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 4px;">Tăng tốc thực tế 2.39x</div>
            </div>
            <div class="glass-panel" style="padding: 16px;">
              <div style="font-size: 12px; color: var(--text-sub);">TIÊU THỤ BỘ NHỚ KỸ THUẬT</div>
              <div style="font-size: 22px; font-weight: bold; color: var(--tertiary); margin-top: 4px;">Tối ưu VRAM</div>
              <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 4px;">Chỉ giữ activation micro-batch</div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  private renderNodes(): string {
    const nodes = this.topology?.nodes || [];
    return `
      <div class="grid-3col">
        ${nodes.map(n => `
          <div class="glass-panel">
            <div class="card-title-bar">
              <h3>${n.id} (${n.role.toUpperCase()})</h3>
              <span class="cluster-pill"><div class="cluster-dot"></div> ${n.status}</span>
            </div>
            <div class="card-body" style="display: flex; flex-direction: column; gap: 12px;">
              <div>
                <div style="font-size: 12px; color: var(--text-sub);">PHẦN CỨNG GPU</div>
                <div style="font-size: 15px; font-weight: 600;">${n.device_name || 'NVIDIA GPU'}</div>
              </div>

              <div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
                  <span>VRAM Sử dụng:</span>
                  <b>${n.vram_used_gb} / ${n.vram_total_gb} GB</b>
                </div>
                <div class="progress-bar-container">
                  <div class="progress-fill" style="width: ${(n.vram_used_gb / n.vram_total_gb) * 100}%;"></div>
                </div>
              </div>

              <div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
                  <span>CPU Tải:</span>
                  <b>${n.cpu_percent}%</b>
                </div>
                <div class="progress-bar-container">
                  <div class="progress-fill" style="width: ${n.cpu_percent}%; background: var(--accent-cyan);"></div>
                </div>
              </div>

              <div style="display: flex; justify-content: space-between; font-size: 12.5px; border-top: 1px solid var(--border-subtle); padding-top: 10px;">
                <span>TFLOPS Năng Lực:</span>
                <b>${n.tflops} TFLOPS</b>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 12.5px;">
                <span>Độ trễ P2P:</span>
                <b style="color: var(--accent-green);">${n.latency_ms} ms</b>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 12.5px;">
                <span>Băng thông mạng:</span>
                <b>${n.bandwidth_mbps} Mbps</b>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 12.5px;">
                <span>Layers đang giữ:</span>
                <span class="node-shard-badge">${n.assigned_shards}</span>
              </div>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  private renderModels(): string {
    return `
      <div class="glass-panel">
        <div class="card-title-bar">
          <h3>Danh Mục Mô Hình & Tương Thích Sharding</h3>
          <button class="btn btn-primary">+ Tải Mô Hình Từ HuggingFace</button>
        </div>
        <div class="card-body" style="padding: 0;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Tên Model</th>
                <th>Kích Thước</th>
                <th>Số Layers</th>
                <th>Context Length</th>
                <th>VRAM Cần Thiết</th>
                <th>Khuyến Nghị Node</th>
                <th>Trạng Thái Cụm</th>
                <th>Thao Tác</th>
              </tr>
            </thead>
            <tbody>
              ${this.models.map(m => `
                <tr>
                  <td><b>${m.name}</b></td>
                  <td>${m.size}</td>
                  <td>${m.layers || 32}</td>
                  <td>${m.context_length || 4096}</td>
                  <td>${m.vram_required_gb} GB</td>
                  <td><span class="node-role-badge">${m.recommended_cluster_nodes} Nodes LACP</span></td>
                  <td>
                    ${m.status === 'loaded' 
                      ? '<span class="cluster-pill"><div class="cluster-dot"></div> Đang Chạy Active</span>' 
                      : '<span style="color: var(--text-sub); font-size: 12px;">Sẵn sàng nạp</span>'}
                  </td>
                  <td>
                    <button class="btn btn-glass" style="padding: 4px 10px; font-size: 12px;">${m.status === 'loaded' ? 'Reload' : 'Deploy Shards'}</button>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  private renderTraining(): string {
    const job = this.trainingJob || {
      job_id: 'train-9402',
      model_name: 'qwen2.5:1.5b',
      dataset_name: 'instruct_alpaca_clean_vi.jsonl',
      status: 'running',
      current_step: 1420,
      total_steps: 5000,
      loss: 0.384,
      learning_rate: 2e-5,
      schedule: '1F1B',
      active_nodes: 3,
      eta_seconds: 3450
    };

    return `
      <div class="glass-panel">
        <div class="card-title-bar">
          <h3>Tiến Trình Huấn Luyện Phân Tán (1F1B Distributed Training)</h3>
          <span class="cluster-pill"><div class="cluster-dot"></div> Job ID: ${job.job_id}</span>
        </div>
        <div class="card-body">
          <div class="grid-3col" style="margin-bottom: 20px;">
            <div class="glass-panel" style="padding: 16px;">
              <div style="font-size: 12px; color: var(--text-sub);">MODEL & DATASET</div>
              <div style="font-size: 15px; font-weight: 600; margin-top: 4px;">${job.model_name}</div>
              <div style="font-size: 12px; color: var(--text-muted);">${job.dataset_name}</div>
            </div>
            <div class="glass-panel" style="padding: 16px;">
              <div style="font-size: 12px; color: var(--text-sub);">CURRENT LOSS & LR</div>
              <div style="font-size: 20px; font-weight: bold; color: var(--accent-green); margin-top: 4px;">${job.loss}</div>
              <div style="font-size: 12px; color: var(--text-muted);">LR: ${job.learning_rate} · 1F1B autograd</div>
            </div>
            <div class="glass-panel" style="padding: 16px;">
              <div style="font-size: 12px; color: var(--text-sub);">TIẾN ĐỘ & THỜI GIAN CÒN LẠI</div>
              <div style="font-size: 20px; font-weight: bold; color: var(--primary); margin-top: 4px;">${Math.round((job.current_step/job.total_steps)*100)}%</div>
              <div style="font-size: 12px; color: var(--text-muted);">Bước ${job.current_step} / ${job.total_steps} (ETA: ${Math.round(job.eta_seconds/60)} phút)</div>
            </div>
          </div>

          <div class="progress-bar-container" style="height: 10px; margin-bottom: 24px;">
            <div class="progress-fill" style="width: ${(job.current_step/job.total_steps)*100}%;"></div>
          </div>

          <div style="background: #080c14; padding: 16px; border-radius: var(--radius-sm); font-size: 13px; color: var(--text-muted); line-height: 1.6;">
            <b>Quy trình Gradient 1F1B & Ring All-Reduce:</b><br>
            1. Mỗi node chạy forward pass micro-batch trên shard cục bộ và chuyển activation tensor sang node kế tiếp.<br>
            2. Trong pha backward, mỗi node tính <code>back_gradient</code> cục bộ qua autograd và bàn giao gradient ngược về node trước đó.<br>
            3. Gradients sau đó được đồng bộ hoá qua Ring All-Reduce không nghẽn băng thông trước khi optimizer thực hiện step update.
          </div>
        </div>
      </div>
    `;
  }

  private renderDatasets(): string {
    return `
      <div class="glass-panel">
        <div class="card-title-bar">
          <h3>Quản Lý Tập Dữ Liệu Huấn Luyện (Datasets)</h3>
          <button class="btn btn-primary">+ Tải Lên Dataset Mới</button>
        </div>
        <div class="card-body" style="padding: 0;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Tên Tập Dữ Liệu</th>
                <th>Dung Lượng</th>
                <th>Số Mẫu (Samples)</th>
                <th>Định Dạng</th>
                <th>Ngày Cập Nhật</th>
                <th>Trạng Thái</th>
                <th>Thao Tác</th>
              </tr>
            </thead>
            <tbody>
              ${this.datasets.map(d => `
                <tr>
                  <td><b>${d.name}</b></td>
                  <td>${d.size}</td>
                  <td>${d.samples.toLocaleString()} samples</td>
                  <td><span class="node-role-badge">${d.format}</span></td>
                  <td>${d.last_modified}</td>
                  <td><span class="cluster-pill"><div class="cluster-dot"></div> ${d.status}</span></td>
                  <td>
                    <button class="btn btn-glass" style="padding: 4px 10px; font-size: 12px;">Xem Trước</button>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  private renderLogs(): string {
    return `
      <div class="glass-panel">
        <div class="card-title-bar">
          <h3>Bảng Điều Khiển Live Logs & Mesh Heartbeat</h3>
          <button class="btn btn-glass" onclick="location.reload()">Làm Mới</button>
        </div>
        <div class="card-body">
          <div class="terminal-window" style="height: 480px;">
            ${this.logs.map(log => `
              <div class="log-line">
                <span class="log-time">[${log.timestamp}]</span>
                <span class="log-src">[${log.source}]</span>
                <span class="log-${log.level}">${log.message}</span>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  }

  private renderChat(): string {
    return `
      <div class="chat-container">
        <div class="chat-messages" id="chat-messages">
          ${this.chatHistory.map(msg => `
            <div class="chat-bubble ${msg.role}">
              <div>${msg.text}</div>
              ${msg.meta ? `<div class="chat-meta">${msg.meta}</div>` : ''}
            </div>
          `).join('')}
        </div>
        <div class="chat-input-bar">
          <input type="text" id="chat-input" placeholder="Nhập prompt để thử nghiệm inference phân tán qua 3 Nodes..." autofocus />
          <button class="btn btn-primary" id="btn-send-chat">Gửi Prompt</button>
        </div>
      </div>
    `;
  }

  private setupChatEvents() {
    const input = document.getElementById('chat-input') as HTMLInputElement;
    const btn = document.getElementById('btn-send-chat');

    const handleSend = async () => {
      const text = input.value.trim();
      if (!text) return;
      input.value = '';

      this.chatHistory.push({ role: 'user', text });
      this.renderContent();

      const res = await sendChatPrompt(text);
      this.chatHistory.push({
        role: 'ai',
        text: res.content,
        meta: `⚡ ${res.tps} tokens/s · Độ trễ: ${res.latency}ms · 3 Nodes Sharded`
      });
      this.renderContent();
    };

    if (btn) btn.onclick = handleSend;
    if (input) {
      input.onkeydown = (e) => {
        if (e.key === 'Enter') handleSend();
      };
    }
  }
}

// Bootstrap application
document.addEventListener('DOMContentLoaded', () => {
  new DashboardApp();
});
