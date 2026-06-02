// Synapse AI — Normalizers: transform API responses to internal store shape
Synapse.Normalizers = {
  clusterNode(raw) {
    return {
      node_id:      raw.node_id || '',
      name:         raw.name || 'Unknown',
      ip:           raw.ip || '',
      is_self:      !!raw.is_self,
      has_gpu:      !!raw.has_gpu,
      chip:         raw.chip || 'Unknown',
      gpu_backend:  raw.gpu_backend || '',
      vram_gb:      raw.vram_gb || 0,
      ram_gb:       raw.ram_gb || 0,
      ram_used_gb:  raw.ram_used_gb || 0,
      ram_total_gb: raw.ram_total_gb || 0,
      disk_gb:      raw.disk_gb || 0,
      cpu_cores:    raw.cpu_cores || 0,
      cpu_pct:      raw.cpu_pct !== undefined ? raw.cpu_pct : null,
      flops_fp16:   raw.flops_fp16 || 0,
      layer_start:  raw.layer_start || 0,
      layer_end:    raw.layer_end || 0,
      layer_count:  raw.layer_count || 0,
      share_pct:    raw.share_pct || 0,
      layer_start_pct: raw.layer_start_pct || 0,
      layer_end_pct:   raw.layer_end_pct || 0,
      gpu_utilization: raw.gpu_utilization !== undefined ? raw.gpu_utilization : null,
      gpu_vram_used_mb: raw.gpu_vram_used_mb || null,
      gpu_power_w:  raw.gpu_power_w || null,
      gpu_temp_c:   raw.gpu_temp_c || null
    };
  },

  clusterNodes(nodes) {
    if (!Array.isArray(nodes)) return [];
    return nodes.map(n => this.clusterNode(n));
  },

  localStats(raw) {
    return {
      cpu:               raw.cpu ?? 0,
      memory:            raw.memory ?? 0,
      memory_used_gb:    raw.memory_used_gb ?? 0,
      memory_total_gb:   raw.memory_total_gb ?? 0,
      uptime:            raw.uptime ?? '-',
      gpu:               raw.gpu ?? 'N/A',
      gpu_name:          raw.gpu_name ?? null,
      gpu_vram_gb:       raw.gpu_vram_gb ?? null,
      unified_memory:    raw.unified_memory ?? false,
      gpu_utilization:   raw.gpu_utilization ?? null,
      gpu_vram_used_mb:  raw.gpu_vram_used_mb ?? null,
      gpu_power_w:       raw.gpu_power_w ?? null,
      gpu_temp_c:        raw.gpu_temp_c ?? null,
      models_count:      raw.models_count ?? 0,
      datasets_count:    raw.datasets_count ?? 0
    };
  },

  modelItem(raw, isDownloaded) {
    return {
      name:              raw.name || '',
      family:            raw.family || '',
      architecture:      raw.architecture || '',
      parameter_size:    raw.parameter_size || '',
      quantization_level: raw.quantization_level || '',
      memory:            raw.memory || '',
      size_gb:           raw.size_gb || null,
      hf_downloads:      raw.hf_downloads || 0,
      pipeline_tag:      raw.pipeline_tag || '',
      updated_at:        raw.updated_at || '',
      fit:               raw.fit || null,
      memory_needed_gb:  raw.memory_needed_gb || null,
      estimated_speed:   raw.estimated_speed || null,
      isDownloaded:      !!isDownloaded
    };
  },

  modelList(raw) {
    const downloaded = (raw.downloaded || []).map(m => this.modelItem(m, true));
    const available = (raw.available || []).map(m => this.modelItem(m, false));
    return { downloaded, available };
  },

  datasets(items) {
    if (!Array.isArray(items)) return [];
    return items.map(d => ({
      name:         d.name || '',
      path:         d.path || '',
      sample_count: d.sample_count,
      size_mb:      d.size_mb || 0,
      created:      d.created || ''
    }));
  },

  trainingJob(raw) {
    if (!raw) return null;
    return {
      job_id:         raw.job_id || '',
      model:          raw.model || '',
      dataset:        raw.dataset || '',
      epochs:         raw.epochs || 0,
      current_epoch:  raw.current_epoch || 0,
      progress:       raw.progress || 0,
      status:         raw.status || 'idle',
      started_at:     raw.started_at || '',
      completed_at:   raw.completed_at || null,
      error:          raw.error || null,
      loss:           raw.loss !== undefined ? raw.loss : null
    };
  },

  tailscaleNode(raw) {
    return {
      device_id:          raw.device_id || '',
      name:               raw.name || '',
      addresses:          raw.addresses || [],
      last_seen_sec_ago:  raw.last_seen_sec_ago !== undefined ? raw.last_seen_sec_ago : null,
      is_self:            !!raw.is_self,
      is_synapse_node:    !!raw.is_synapse_node,
      model:              raw.model || '',
      memory_gb:          raw.memory_gb || null,
      cpu_cores:          raw.cpu_cores || null,
      flops_fp16:         raw.flops_fp16 || null,
      system_ram_gb:      raw.system_ram_gb || null,
      disk_gb:            raw.disk_gb || null
    };
  },

  activity(items) {
    if (!Array.isArray(items)) return [];
    return items.map(a => ({
      message:    a.message || '',
      timestamp:  a.timestamp || '',
      type:       a.type || 'info'
    }));
  }
};