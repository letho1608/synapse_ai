// Synapse AI — Utility Functions
Synapse.Utils = {
  formatNumber(n, digits) {
    if (n == null) return '-';
    return Number(n).toFixed(digits || 0);
  },

  formatGB(n) {
    if (n == null || n === 0) return '0 GB';
    return Number(n).toFixed(1) + ' GB';
  },

  formatPct(p) {
    if (p == null) return '-';
    return Number(p).toFixed(0) + '%';
  },

  formatDuration(sec) {
    if (!sec) return '-';
    const d = Math.floor(sec / 86400);
    const h = Math.floor((sec % 86400) / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    const parts = [];
    if (d > 0) parts.push(d + 'd');
    if (h > 0) parts.push(h + 'h');
    if (m > 0) parts.push(m + 'm');
    if (s > 0 || parts.length === 0) parts.push(s + 's');
    return parts.join(' ');
  },

  truncate(s, maxLen) {
    if (!s) return '';
    if (s.length <= maxLen) return s;
    return s.slice(0, maxLen - 1) + '…';
  },

  ago(ms) {
    if (!ms) return '-';
    const sec = Math.floor((Date.now() - ms) / 1000);
    if (sec < 5) return 'vừa xong';
    if (sec < 60) return sec + 's ago';
    if (sec < 3600) return Math.floor(sec / 60) + 'm ago';
    return Math.floor(sec / 3600) + 'h ago';
  },

  isStale(lastFetch) {
    if (!lastFetch) return true;
    return (Date.now() - lastFetch) > Synapse.CONFIG.STALE_THRESHOLD_MS;
  },

  deepEqual(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
  },

  parseAnsi(text) {
    if (!text) return '';
    const ansiColorMap = {
      '30': 'color: black;', '31': 'color: #EF4444;', '32': 'color: #10B981;', '33': 'color: #F59E0B;',
      '34': 'color: #3B82F6;', '35': 'color: #8B5CF6;', '36': 'color: #06B6D4;', '37': 'color: white;',
      '90': 'color: gray;', '91': 'color: #F87171;', '92': 'color: #34D399;', '93': 'color: #FBBF24;',
      '94': 'color: #60A5FA;', '95': 'color: #A78BFA;', '96': 'color: #22D3EE;', '97': 'color: white;',
      '1': 'font-weight: bold;', '0': ''
    };
    let html = '';
    let currentStyle = '';
    const parts = text.split(/\x1b\[/);
    html += parts[0].replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    for (let i = 1; i < parts.length; i++) {
      const match = parts[i].match(/^([0-9;]*)m(.*)/);
      if (match) {
        const codes = match[1].split(';');
        let nextStyle = currentStyle;
        codes.forEach(code => {
          if (code === '0' || code === '') nextStyle = '';
          else if (ansiColorMap[code]) nextStyle += ansiColorMap[code];
        });
        currentStyle = nextStyle;
        const textPart = match[2].replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        if (currentStyle) {
          html += `<span style="${currentStyle}">${textPart}</span>`;
        } else {
          html += textPart;
        }
      } else {
        html += '\x1b[' + parts[i].replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      }
    }
    return html;
  }
};