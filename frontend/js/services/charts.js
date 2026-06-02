// Synapse AI — Chart.js Lifecycle Service
// Manages Chart instances to prevent memory leaks.
Synapse.Charts = {
  _instances: {},

  createOrReplace(key, canvasId, config) {
    this.destroy(key);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    this._instances[key] = new Chart(canvas, config);
    return this._instances[key];
  },

  destroy(key) {
    if (this._instances[key]) {
      this._instances[key].destroy();
      delete this._instances[key];
    }
  },

  destroyAll() {
    Object.keys(this._instances).forEach(k => this.destroy(k));
  },

  // Cluster node colors — dynamic palette supporting up to 12 nodes
  getNodeColors(nodes) {
    return nodes.map((_, i) => Synapse.CONFIG.CLUSTER_COLORS[i % Synapse.CONFIG.CLUSTER_COLORS.length]);
  },

  getNodeBgColors(nodes) {
    return this.getNodeColors(nodes).map(c => c + 'CC');
  }
};