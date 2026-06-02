// Synapse AI — Centralized Polling Service
// Replaces the two conflicting setInterval calls with a single, managed polling system.
Synapse.Services = {
  _slowInterval: null,
  _fastInterval: null,
  _ws: null,

  boot() {
    this.startPolling();
    this.initWebSocket();
  },

  startPolling() {
    this._slowInterval = setInterval(() => this._pollSlow(), Synapse.CONFIG.POLL_INTERVAL_SLOW);
    this._fastInterval = setInterval(() => this._pollFast(), Synapse.CONFIG.POLL_INTERVAL_FAST);
  },

  stopPolling() {
    if (this._slowInterval) { clearInterval(this._slowInterval); this._slowInterval = null; }
    if (this._fastInterval) { clearInterval(this._fastInterval); this._fastInterval = null; }
  },

  _pollSlow() {
    const store = Alpine.store('app');
    const page = store.currentPage;

    store.fetchStats();
    store.fetchActivity();
    store.fetchTailscaleNodes();
    store.fetchDistributedStatus();

    if (page === 'training' || page === 'dashboard') store.fetchTrainingStatus();
    if (page === 'datasets') store.fetchDatasets();
    if (page === 'settings') store.fetchSettings();
    if (page === 'test' || page === 'models') store.fetchModelList();
    if (page === 'cluster' || page === 'dashboard') store.fetchClusterResources();
    if (page === 'monitoring') store.fetchClusterResources();
  },

  _pollFast() {
    const store = Alpine.store('app');
    if (store.currentPage === 'dashboard') {
      store.fetchTerminalLogs();
    }
  },

  // Called when page changes — immediately fetches needed data for the new page
  refetchForPage(refetchList) {
    const store = Alpine.store('app');
    const ui = Alpine.store('ui');
    if (refetchList.includes('stats')) store.fetchStats();
    if (refetchList.includes('activity')) store.fetchActivity();
    if (refetchList.includes('tailscale')) store.fetchTailscaleNodes();
    if (refetchList.includes('distributed')) store.fetchDistributedStatus();
    if (refetchList.includes('modelList')) store.fetchModelList();
    if (refetchList.includes('modelStatus')) store.fetchModelStatus();
    if (refetchList.includes('datasets')) store.fetchDatasets();
    if (refetchList.includes('training')) store.fetchTrainingStatus();
    if (refetchList.includes('cluster')) store.fetchClusterResources();
    if (refetchList.includes('settings')) store.fetchSettings();
    // Auto-set default model for test/training pages
    if (store.currentPage === 'test' && ui.chatModel === '') {
      store.fetchSettings().then(s => { if (s.default_model) ui.chatModel = s.default_model; });
    }
    if (store.currentPage === 'training' && ui.trainingModel === '') {
      store.fetchSettings().then(s => { if (s.default_model) ui.trainingModel = s.default_model; });
    }
  },

  // =============================================
  // WebSocket Layer
  // =============================================
  initWebSocket() {
    if (!Synapse.CONFIG.WS_ENABLED || !Synapse.CONFIG.WS_URL) return;

    const wsStore = Alpine.store('ws');
    try {
      this._ws = new WebSocket(Synapse.CONFIG.WS_URL);
      this._ws.addEventListener('open', () => {
        wsStore.connected = true;
        wsStore.reconnectAttempts = 0;
        this._ws.send(JSON.stringify({ action: 'subscribe', events: ['cluster:update', 'training:progress', 'log:line'] }));
      });

      this._ws.addEventListener('message', (event) => {
        wsStore.lastMessageAt = Date.now();
        const msg = JSON.parse(event.data);
        this._handleWsMessage(msg);
      });

      this._ws.addEventListener('close', () => {
        wsStore.connected = false;
        this._scheduleReconnect();
      });

      this._ws.addEventListener('error', () => {
        wsStore.connected = false;
      });
    } catch (e) { /* WS not supported */ }
  },

  _scheduleReconnect() {
    const wsStore = Alpine.store('ws');
    wsStore.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, wsStore.reconnectAttempts - 1), 30000);
    setTimeout(() => this.initWebSocket(), delay);
  },

  _handleWsMessage(msg) {
    const store = Alpine.store('app');
    switch (msg.type) {
      case 'cluster:update':
        store.cluster.nodes = Synapse.Normalizers.clusterNodes(msg.data.nodes);
        store.cluster.strategy = msg.data.strategy_name || store.cluster.strategy;
        store.cluster._lastFetch = Date.now();
        break;
      case 'training:progress':
        store.training.job = Synapse.Normalizers.trainingJob(msg.data.job);
        store.training.status = msg.data.status || store.training.status;
        break;
      case 'log:line':
        const lines = store.terminal.lines;
        lines.push(msg.data.line);
        if (lines.length > store.terminal._maxLines) {
          lines.splice(0, lines.length - store.terminal._maxLines);
        }
        break;
      case 'node:heartbeat':
        const idx = store.cluster.nodes.findIndex(n => n.node_id === msg.data.node_id);
        if (idx >= 0) Object.assign(store.cluster.nodes[idx], msg.data);
        break;
    }
  }
};