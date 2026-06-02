// Synapse AI — API Client: base fetch wrapper with retry, error handling
Synapse.API.Client = {
  async _fetch(url, options) {
    try {
      const res = await fetch(url, options);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.message || data.detail || 'HTTP ' + res.status);
      }
      return await res.json();
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      console.error('[API]', url, e.message);
      throw e;
    }
  },

  async get(url, useCache) {
    if (useCache && Synapse.Cache.has(url)) {
      return Synapse.Cache.get(url);
    }
    const data = await this._fetch(url);
    if (useCache) Synapse.Cache.set(url, data);
    return data;
  },

  async post(url, body) {
    return this._fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  },

  async postForm(url, formData) {
    try {
      const res = await fetch(url, { method: 'POST', body: formData });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.message || 'HTTP ' + res.status);
      }
      return await res.json();
    } catch (e) {
      console.error('[API]', url, e.message);
      throw e;
    }
  },

  postStream(url, body, signal) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal
    });
  }
};