// Synapse AI — WebSocket Store: connection state
document.addEventListener('alpine:init', function () {
  Alpine.store('ws', {
    connected: false,
    reconnectAttempts: 0,
    lastMessageAt: null,

    get statusText() {
      const appStore = Alpine.store('app');
      const isConnected = this.connected || (appStore && appStore.stats._lastFetch !== null && !appStore.stats.isStale);
      if (isConnected) return 'Đã kết nối';
      if (this.reconnectAttempts > 0) return 'Đang kết nối lại (' + this.reconnectAttempts + ')';
      return 'Chưa kết nối';
    },

    get statusClass() {
      const appStore = Alpine.store('app');
      const isConnected = this.connected || (appStore && appStore.stats._lastFetch !== null && !appStore.stats.isStale);
      if (isConnected) return 'connected';
      if (this.reconnectAttempts > 0) return 'connecting';
      return 'disconnected';
    }
  });
});