// Synapse AI — Primary Centralized State Store
// ALL data flows through this single store. Pages read only, never compute derived state.
document.addEventListener('alpine:init', function () {
  Alpine.store('app', {
    // === Navigation ===
    currentPage: 'dashboard',
    sidebarOpen: false,

    // === Global loading state ===
    _initialized: false,

    // === Local System Stats (from /v1/system/stats) ===
    stats: {
      cpu: 0,
      memory: 0,
      memory_used_gb: 0,
      memory_total_gb: 0,
      uptime: '-',
      gpu: 'N/A',
      gpu_name: null,
      gpu_vram_gb: null,
      unified_memory: false,
      gpu_utilization: null,
      gpu_vram_used_mb: null,
      gpu_power_w: null,
      gpu_temp_c: null,
      models_count: 0,
      datasets_count: 0,
      _lastFetch: null,
      _loading: false,
      get isStale() { return Synapse.Utils.isStale(this._lastFetch); }
    },

    // === Tailscale Nodes ===
    tailscaleNodes: [],
    tailscale_lastFetch: null,
    get tailscale_synapse_count() {
      return this.tailscaleNodes.filter(n => n.is_synapse_node).length;
    },
    get tailscale_stale() {
      return Synapse.Utils.isStale(this.tailscale_lastFetch);
    },

    // === Distributed Status ===
    distributedStatus: {
      partitions_count: 0,
      multi_node_inference: false,
      multi_node_training: false,
      message: ''
    },

    // === Activity Feed ===
    activities: [],

    // === Models (centralized, no duplicate fetch) ===
    models: {
      downloaded: [],
      available: [],
      compatibility: {},
      _loading: false,
      _lastFetch: null,
      get all() {
        const downloadedNames = new Set(this.downloaded.map(m => m.name));
        const filteredAvailable = this.available.filter(m => !downloadedNames.has(m.name));
        return [...this.downloaded, ...filteredAvailable];
      },
      get downloadedModels() { return this.downloaded; }
    },

    // === Datasets ===
    datasets: {
      items: [],
      _loading: false,
      _lastFetch: null
    },

    // === Training (single job state — NO duplicate) ===
    training: {
      status: 'idle',
      job: null,
      _loading: false,
      get isRunning() { return this.status === 'started'; },
      get isCompleted() { return this.status === 'completed'; },
      get isFailed() { return this.status === 'failed'; },
      get progress() { return this.job ? (this.job.progress || 0) : 0; }
    },

    // === Cluster State (centralized — NO page recomputes totals) ===
    cluster: {
      nodes: [],
      strategy: '',
      totalLayers: 32,
      peerGraph: {},
      trainingJob: null,
      _loading: false,
      _lastFetch: null,
      _error: null,
      // Pre-computed totals — updated when nodes change
      get totals() {
        const n = this.nodes;
        return {
          vram_gb:    n.reduce((s, x) => s + (x.vram_gb || 0), 0),
          ram_gb:     n.reduce((s, x) => s + (x.ram_gb || 0), 0),
          tfpops:     n.reduce((s, x) => s + (x.flops_fp16 || 0), 0),
          cpu_cores:  n.reduce((s, x) => s + (x.cpu_cores || 0), 0),
          disk_gb:    n.reduce((s, x) => s + (x.disk_gb || 0), 0),
          nodeCount:  n.length
        };
      },
      get selfNode() { return this.nodes.find(n => n.is_self) || null; },
      get isStale() { return Synapse.Utils.isStale(this._lastFetch); }
    },

    // === Settings ===
    settings: {
      default_model: '',
      theme: 'light',
      _loading: false
    },

    // === Service Status ===
    modelStatus: {
      enabled: false,
      base_url: null,
      models: []
    },

    // === System Info ===
    systemInfo: {
      project: 'Synapse AI',
      version: '-',
      hardware: {}
    },

    // === Terminal Log ===
    terminal: {
      lines: [],
      _maxLines: Synapse.CONFIG.TERMINAL_MAX_LINES,
      get lineCount() { return this.lines.length; }
    },

    // === Init: fires when Alpine first initializes ===
    init() {
      if (this._initialized) return;
      this._initialized = true;
      this.settings.theme = localStorage.getItem('synapse_theme') || 'light';
      this.applyTheme(this.settings.theme);
      this.fetchSystemInfo();
      Synapse.Services.boot();
    },

    applyTheme(theme) {
      if (!theme) return;
      this.settings.theme = theme;
      localStorage.setItem('synapse_theme', theme);
      document.documentElement.classList.toggle('theme-dark', theme === 'dark');
      document.body.classList.toggle('theme-dark', theme === 'dark');
    },

    navigate(pageId, opts) {
      this.currentPage = pageId;
      this.sidebarOpen = false;
      const page = Synapse.PAGES.find(p => p.id === pageId);
      if (page && page.refetch.length > 0) {
        Synapse.Services.refetchForPage(page.refetch);
      }
    },

    getPageTitle() {
      const page = Synapse.PAGES.find(p => p.id === this.currentPage);
      return page ? page.label : 'Synapse AI';
    },

    // =============================================
    // API FETCH METHODS (write to store, not to DOM)
    // =============================================

    async fetchStats() {
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.STATS);
        this.stats.cpu              = data.cpu ?? 0;
        this.stats.memory           = data.memory ?? 0;
        this.stats.memory_used_gb   = data.memory_used_gb ?? 0;
        this.stats.memory_total_gb  = data.memory_total_gb ?? 0;
        this.stats.uptime           = data.uptime ?? '-';
        this.stats.gpu              = data.gpu ?? 'N/A';
        this.stats.gpu_name         = data.gpu_name ?? null;
        this.stats.gpu_vram_gb      = data.gpu_vram_gb ?? null;
        this.stats.unified_memory   = data.unified_memory ?? false;
        this.stats.gpu_utilization  = data.gpu_utilization ?? null;
        this.stats.gpu_vram_used_mb = data.gpu_vram_used_mb ?? null;
        this.stats.gpu_power_w      = data.gpu_power_w ?? null;
        this.stats.gpu_temp_c       = data.gpu_temp_c ?? null;
        this.stats.models_count     = this.models.all.length > 0 ? this.models.all.length : (data.models_count || 0);
        this.stats.datasets_count   = this.datasets.items.length > 0 ? this.datasets.items.length : (data.datasets_count || 0);
        this.stats._lastFetch       = Date.now();
        
        const ws = Alpine.store('ws');
        if (ws) ws.connected = true;
      } catch (e) { 
        const ws = Alpine.store('ws');
        if (ws) ws.connected = false;
      }
    },

    async fetchModelStatus() {
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.MODEL_STATUS);
        this.modelStatus = {
          enabled:  !!data.enabled,
          base_url: data.base_url || null,
          models:   data.models || []
        };
      } catch (e) { /* silent */ }
    },

    async fetchSystemInfo() {
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.SYSTEM_INFO);
        this.systemInfo = {
          project:  data.project || 'Synapse AI',
          version:  data.version || '-',
          hardware: data.hardware || {}
        };
      } catch (e) { /* silent */ }
    },

    async fetchActivity() {
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.ACTIVITY);
        this.activities = Synapse.Normalizers.activity(data.data);
      } catch (e) { /* silent */ }
    },

    async fetchTerminalLogs() {
      if (this.terminal._lastFetch && Date.now() - this.terminal._lastFetch < 600) return;
      this.terminal._lastFetch = Date.now();
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.LOGS_TERMINAL);
        this.terminal.lines = data.data || [];
      } catch (e) { /* silent */ }
    },

    async fetchModelList() {
      this.models._loading = true;
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.MODEL_LIST);
        if (!data) return;
        const normalized = Synapse.Normalizers.modelList(data);
        // Merge compatibility
        try {
          const compatData = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.MODEL_COMPAT);
          if (compatData && Array.isArray(compatData.models)) {
            const map = {};
            compatData.models.forEach(m => { map[m.name] = m; });
            normalized.downloaded = normalized.downloaded.map(m => ({ ...m, ...map[m.name] }));
            normalized.available = normalized.available.map(m => ({ ...m, ...map[m.name] }));
          }
        } catch (e) { /* ignore compat errors */ }
        this.models.downloaded = normalized.downloaded;
        this.models.available = normalized.available;
        this.models._lastFetch = Date.now();
        this.stats.models_count = normalized.downloaded.length;
      } catch (e) {
        this.models.downloaded = [];
        this.models.available = [];
      } finally {
        this.models._loading = false;
      }
    },

    async fetchDatasets() {
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.DATASETS);
        this.datasets.items = Synapse.Normalizers.datasets(data.data);
        this.datasets._lastFetch = Date.now();
        this.stats.datasets_count = this.datasets.items.length;
      } catch (e) { /* silent */ }
    },

    async fetchTailscaleNodes() {
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.TAILSCALE_NODES);
        this.tailscaleNodes = (data.data || []).map(n => Synapse.Normalizers.tailscaleNode(n));
        this.tailscale_lastFetch = Date.now();
      } catch (e) { this.tailscaleNodes = []; }
    },

    async fetchDistributedStatus() {
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.DISTRIBUTED_STATUS);
        this.distributedStatus = {
          partitions_count:     data.partitions_count ?? 0,
          multi_node_inference: !!data.multi_node_inference,
          multi_node_training:  !!data.multi_node_training,
          message:              data.message || ''
        };
      } catch (e) {
        this.distributedStatus = { partitions_count: 0, multi_node_inference: false, multi_node_training: false, message: '' };
      }
    },

    async fetchSettings() {
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.SETTINGS);
        this.settings.default_model = data.default_model || '';
        return this.settings;
      } catch (e) { return this.settings; }
    },

    async fetchClusterResources() {
      this.cluster._loading = true;
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.CLUSTER_RESOURCES);
        const nodes = Synapse.Normalizers.clusterNodes(data.nodes);
        if (!Synapse.Utils.deepEqual(nodes, this.cluster.nodes)) {
          this.cluster.nodes = nodes;
          this.cluster.strategy = data.strategy_name || 'Unknown';
          this.cluster.totalLayers = data.total_layers || 32;
          this.cluster.peerGraph = data.peer_graph || {};
          this.cluster.trainingJob = Synapse.Normalizers.trainingJob(data.training_job);
          this.cluster._lastFetch = Date.now();
        }
      } catch (e) { this.cluster._error = e.message; }
      finally { this.cluster._loading = false; }
    },

    async fetchTrainingStatus() {
      try {
        const data = await Synapse.API.Client.get(Synapse.API.ENDPOINTS.TRAINING_STATUS);
        this.training.status = data.status || 'idle';
        this.training.job = Synapse.Normalizers.trainingJob(data.job);
      } catch (e) {
        this.training.status = 'idle';
        this.training.job = null;
      }
    },

    // =============================================
    // USER ACTIONS (POST endpoints)
    // =============================================

    async pullModel(name) {
      if (!name) return;
      const result = await Synapse.API.Client.post(Synapse.API.ENDPOINTS.MODEL_PULL, { model: name });
      if (result.success) { await this.fetchModelList(); }
      else if (result.message) { alert(result.message); }
    },

    async deleteModel(name) {
      if (!name || !confirm('Xóa model "' + name + '"? Không thể hoàn tác.')) return;
      const result = await Synapse.API.Client.post(Synapse.API.ENDPOINTS.MODEL_DELETE, { model: name });
      if (result.success) { await this.fetchModelList(); }
      else if (result.message) { alert(result.message); }
    },

    async saveSettings(form) {
      const result = await Synapse.API.Client.post(Synapse.API.ENDPOINTS.SETTINGS, {
        default_model: form.default_model || undefined
      });
      if (result.success) {
        await this.fetchModelStatus();
        await this.fetchModelList();
        await this.fetchActivity();
      } else if (result.message) { alert(result.message); }
      return result;
    },

    async startTraining(params) {
      if (!params.model || !params.dataset) return;
      const result = await Synapse.API.Client.post(Synapse.API.ENDPOINTS.TRAINING_START, params);
      if (result.success) {
        await this.fetchTrainingStatus();
        await this.fetchActivity();
      } else if (result.message) { alert(result.message); }
      return result;
    },

    async trainChatCompletion(messages, model, temperature, signal) {
      const startTime = performance.now();
      const res = await Synapse.API.Client.postStream(Synapse.API.ENDPOINTS.CHAT_COMPLETIONS, {
        model,
        messages,
        temperature,
        stream: false
      }, signal);
      const elapsed = (performance.now() - startTime) / 1000;
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || data.error?.message || data.message || 'HTTP ' + res.status);
      }
      if (data.error) {
        throw new Error(data.error.message || data.error.detail || JSON.stringify(data.error));
      }
      if (data.choices && data.choices[0]) {
        const content = data.choices[0].message?.content || '(Không có nội dung)';
        let usage = '';
        if (data.usage) {
          const ct = data.usage.completion_tokens || 0;
          const pt = data.usage.prompt_tokens || 0;
          const tt = data.usage.total_tokens || 0;
          let tps = '';
          if (ct > 0 && elapsed > 0) tps = ' · ' + (ct / elapsed).toFixed(2) + ' token/s';
          usage = 'Token: đầu vào ' + pt + ', đầu ra ' + ct + ', tổng ' + tt + tps;
        }
        return { content, usage };
      }
      throw new Error('Phản hồi không hợp lệ.');
    },

    // =============================================
    // DATASET ACTIONS
    // =============================================

    async deleteDataset(name) {
      if (!confirm('Xóa dataset "' + name + '"?')) return;
      const result = await Synapse.API.Client.post(Synapse.API.ENDPOINTS.DATASET_DELETE, { name });
      if (result.success) { await this.fetchDatasets(); await this.fetchStats(); }
      else if (result.message) { alert(result.message); }
      return result;
    },

    async renameDataset(oldName, newName) {
      const result = await Synapse.API.Client.post(Synapse.API.ENDPOINTS.DATASET_RENAME, { name: oldName, new_name: newName });
      if (result.success) { await this.fetchDatasets(); }
      else if (result.message) { alert(result.message); }
      return result;
    },

    async validateDataset(path) {
      return Synapse.API.Client.get(Synapse.API.ENDPOINTS.DATASET_VALIDATE + '?path=' + encodeURIComponent(path));
    },

    async previewDataset(path, lines) {
      return Synapse.API.Client.get(Synapse.API.ENDPOINTS.DATASET_PREVIEW + '?path=' + encodeURIComponent(path) + '&lines=' + (lines || 50));
    },

    async createDataset(payload) {
      const result = await Synapse.API.Client.post(Synapse.API.ENDPOINTS.DATASET_CREATE, payload);
      if (result.success) { await this.fetchDatasets(); await this.fetchStats(); }
      else if (result.message) { alert(result.message); }
      return result;
    },

    async uploadDataset(formData) {
      const result = await Synapse.API.Client.postForm(Synapse.API.ENDPOINTS.DATASET_UPLOAD, formData);
      if (result.success) { await this.fetchDatasets(); await this.fetchStats(); }
      else if (result.message) { alert(result.message); }
      return result;
    }
  });
});