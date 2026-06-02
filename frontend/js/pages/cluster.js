// Synapse AI — Cluster Page Chart Rendering
// Handles Chart.js lifecycle for layer bar chart and TFLOPS donut.
Synapse.Pages = Synapse.Pages || {};

Synapse.Pages.cluster = {
  _observer: null,

  // Auto-initialize when Alpine starts
  _booted: false,
  boot() {
    if (this._booted) return;
    this._booted = true;
    this._observer = setInterval(() => this._renderIfNeeded(), 1000);
  },

  _lastNodeHash: '',
  _renderIfNeeded() {
    const store = Alpine.store ? Alpine.store('app') : null;
    if (!store) return;
    if (store.currentPage !== 'cluster') return;
    const nodes = store.cluster.nodes;
    if (!nodes || nodes.length === 0) return;
    const hash = nodes.map(n => n.node_id + ':' + n.layer_count).join(',');
    if (hash === this._lastNodeHash) return;
    this._lastNodeHash = hash;
    this.renderCharts(nodes, store.cluster.totalLayers);
  },

  renderCharts(nodes, totalLayers) {
    if (typeof Chart === 'undefined') return;
    const colors = Synapse.CONFIG.CLUSTER_COLORS;

    // Layer distribution chart
    const layerCanvas = document.getElementById('layerChart');
    if (layerCanvas) {
      Synapse.Charts.createOrReplace('layerChart', 'layerChart', {
        type: 'bar',
        data: {
          labels: nodes.map(n => n.name),
          datasets: [{
            label: 'Số lượng layer',
            data: nodes.map(n => n.layer_count),
            backgroundColor: nodes.map((_, i) => colors[i % colors.length] + 'CC'),
            borderColor: nodes.map((_, i) => colors[i % colors.length]),
            borderWidth: 2,
            borderRadius: 6
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const n = nodes[ctx.dataIndex];
                  return [' Layers: ' + n.layer_start + ' → ' + n.layer_end + ' (' + n.layer_count + ' layers)',
                    ' Tỷ lệ: ' + n.share_pct + '%',
                    n.flops_fp16 > 0 ? ' TFLOPS: ' + n.flops_fp16.toFixed(2) : ' CPU only'];
                }
              }
            }
          },
          scales: {
            x: {
              max: totalLayers,
              title: { display: true, text: 'Số lượng Layer' },
              grid: { color: 'rgba(0,0,0,0.05)' },
              ticks: { stepSize: 4 }
            },
            y: { grid: { display: false } }
          }
        }
      });
    }

    // TFLOPS donut chart
    const flopsCanvas = document.getElementById('flopsDonut');
    if (flopsCanvas) {
      const totalFlops = nodes.reduce((s, n) => s + (n.flops_fp16 || 0), 0);
      const useRam = totalFlops === 0;
      const values = useRam ? nodes.map(n => n.ram_gb || 0) : nodes.map(n => n.flops_fp16 || 0);
      const totalVal = values.reduce((s, v) => s + v, 0);

      Synapse.Charts.createOrReplace('flopsDonut', 'flopsDonut', {
        type: 'doughnut',
        data: {
          labels: nodes.map(n => n.name),
          datasets: [{
            data: values,
            backgroundColor: nodes.map((_, i) => colors[i % colors.length] + 'CC'),
            borderColor: nodes.map((_, i) => colors[i % colors.length]),
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '60%',
          plugins: {
            legend: { position: 'bottom', labels: { font: { size: 12 }, padding: 12 } },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const n = nodes[ctx.dataIndex];
                  const val = values[ctx.dataIndex];
                  const pct = totalVal > 0 ? (val / totalVal * 100).toFixed(1) : 0;
                  return useRam
                    ? ' ' + n.ram_gb + ' GB RAM (' + pct + '%)'
                    : ' ' + n.flops_fp16.toFixed(2) + ' TFLOPS (' + pct + '%)';
                }
              }
            }
          }
        }
      });
    }
  }
};