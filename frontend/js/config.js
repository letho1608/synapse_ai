// Synapse AI — App Configuration & Constants
window.Synapse = window.Synapse || {};

Synapse.CONFIG = {
  APP_NAME: 'Synapse AI',
  API_BASE: '',
  POLL_INTERVAL_SLOW: 5000,
  POLL_INTERVAL_FAST: 800,
  CACHE_TTL_MS: 30000,
  STALE_THRESHOLD_MS: 30000,
  WS_URL: (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/v1/ws/metrics',
  WS_ENABLED: true,
  TERMINAL_MAX_LINES: 1000,
  CLUSTER_COLORS: [
    '#0F766E', '#0EA5E9', '#D97706', '#7C3AED', '#059669',
    '#DC2626', '#F59E0B', '#EC4899', '#6366F1', '#14B8A6',
    '#F97316', '#06B6D4'
  ]
};

Synapse.PAGES = [
  { id: 'dashboard',   label: 'Dashboard',          icon: 'fa-home',             refetch: ['stats', 'activity', 'tailscale', 'distributed', 'training', 'cluster', 'modelList', 'datasets'] },
  { id: 'models',      label: 'Quản Lý Mô Hình',    icon: 'fa-robot',            refetch: ['modelList', 'modelStatus'] },
  { id: 'test',        label: 'Kiểm thử',           icon: 'fa-comments',         refetch: ['modelList'] },
  { id: 'training',    label: 'Training',           icon: 'fa-graduation-cap',   refetch: ['modelList', 'datasets', 'training'] },
  { id: 'datasets',    label: 'Dataset',            icon: 'fa-database',         refetch: ['datasets', 'stats'] },
  { id: 'cluster',     label: 'Cluster',            icon: 'fa-network-wired',    refetch: ['cluster'] },
  { id: 'monitoring',  label: 'Giám Sát',           icon: 'fa-chart-line',       refetch: ['stats', 'tailscale', 'distributed', 'cluster'] },
  { id: 'settings',    label: 'Cài Đặt',            icon: 'fa-cog',              refetch: ['settings', 'modelList'] },
  { id: 'about',       label: 'Giới thiệu',         icon: 'fa-info-circle',      refetch: [] }
];