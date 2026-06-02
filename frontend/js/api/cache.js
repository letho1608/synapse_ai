// Synapse AI — Response Cache with TTL
Synapse.Cache = {
  _store: {},

  get(key) {
    const entry = this._store[key];
    if (!entry) return null;
    if (Date.now() - entry.ts > Synapse.CONFIG.CACHE_TTL_MS) {
      delete this._store[key];
      return null;
    }
    return entry.data;
  },

  set(key, data) {
    this._store[key] = { data, ts: Date.now() };
  },

  invalidate(key) {
    if (key) {
      delete this._store[key];
    } else {
      this._store = {};
    }
  },

  has(key) {
    return this.get(key) !== null;
  }
};