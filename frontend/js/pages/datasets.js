// Synapse AI — Datasets Page Logic
Synapse.Pages = Synapse.Pages || {};

Synapse.Pages.datasets = {
  async upload(ev) {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    const ui = Alpine.store('ui');
    ui.setLoading('upload', true);
    try {
      const form = new FormData();
      form.append('file', file);
      await Alpine.store('app').uploadDataset(form);
      await Alpine.store('app').fetchStats();
    } catch (e) {
      alert('Lỗi: ' + (e.message || e));
    } finally {
      ui.setLoading('upload', false);
      ev.target.value = '';
    }
  },

  async createDataset() {
    const ui = Alpine.store('ui');
    const name = ui.datasetCreateName.trim();
    if (!name) return;
    if (!name.endsWith('.json') && !name.endsWith('.jsonl')) {
      alert('Tên file phải có đuôi .json hoặc .jsonl');
      return;
    }
    ui.setLoading('createDataset', true);
    try {
      const payload = { name, format: ui.datasetCreateFormat, template: ui.datasetCreateTemplate };
      if (ui.datasetCreateContent && ui.datasetCreateContent.trim()) payload.content = ui.datasetCreateContent.trim();
      await Alpine.store('app').createDataset(payload);
      ui.datasetCreateOpen = false;
      ui.datasetCreateName = '';
      ui.datasetCreateContent = '';
    } finally {
      ui.setLoading('createDataset', false);
    }
  },

  async validateDataset(path, name) {
    const ui = Alpine.store('ui');
    ui.setLoading('validate', name);
    ui.datasetValidateResult = null;
    try {
      const data = await Alpine.store('app').validateDataset(path);
      if (data.success) {
        ui.datasetValidateResult = {
          valid_count: data.valid_count,
          total_lines: data.total_lines,
          format_detected: data.format_detected,
          errors: data.errors || []
        };
      } else {
        alert(data.message || 'Kiểm tra thất bại');
      }
    } catch (e) {
      alert('Lỗi: ' + (e.message || e));
    } finally {
      ui.setLoading('validate', null);
    }
  }
};