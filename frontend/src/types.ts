export interface SystemStats {
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  gpu_percent?: number;
  gpu_memory_used_gb?: number;
  gpu_memory_total_gb?: number;
  gpu_name?: string;
  disk_read_mbs?: number;
  disk_write_mbs?: number;
  network_sent_mbs?: number;
  network_recv_mbs?: number;
  tflops?: number;
}

export interface SystemInfo {
  status: string;
  version: string;
  uptime_seconds: number;
  distributed_status: string;
  node_id: string;
  is_master: boolean;
  active_model?: string;
  device?: string;
}

export interface ClusterNode {
  id: string;
  node_id?: string;
  host: string;
  port: number;
  role: 'master' | 'worker' | 'leader';
  status: 'online' | 'offline' | 'healthy' | 'degraded';
  device_name?: string;
  device?: string;
  vram_used_gb: number;
  vram_total_gb: number;
  tflops: number;
  cpu_percent: number;
  ram_percent: number;
  latency_ms: number;
  bandwidth_mbps: number;
  assigned_shards?: string;
  layers?: string;
  step_time_ms?: number;
}

export interface ClusterTopology {
  cluster_id: string;
  master_node: string;
  partition_method: 'lrm' | 'ilp';
  active_model: string;
  total_layers: number;
  pipeline_stages: number;
  nodes: ClusterNode[];
  latency_matrix?: Record<string, Record<string, number>>;
}

export interface DistributedStatus {
  enabled: boolean;
  master: string;
  nodes_count: number;
  active_pipeline: boolean;
  schedule_type: string;
  current_throughput_tps: number;
  total_vram_pool_gb: number;
  used_vram_pool_gb: number;
}

export interface ModelItem {
  name: string;
  size: string;
  format?: string;
  layers?: number;
  context_length?: number;
  status: 'loaded' | 'downloaded' | 'available';
  vram_required_gb: number;
  recommended_cluster_nodes: number;
}

export interface DatasetItem {
  name: string;
  size: string;
  samples: number;
  format: string;
  last_modified: string;
  status: 'ready' | 'processing' | 'indexed';
}

export interface TrainingJob {
  job_id: string;
  model_name: string;
  dataset_name: string;
  status: 'running' | 'paused' | 'completed' | 'failed' | 'idle';
  current_step: number;
  total_steps: number;
  loss: number;
  learning_rate: number;
  schedule: '1F1B' | 'DDP' | 'Pipeline';
  active_nodes: number;
  eta_seconds: number;
}

export interface ActivityLog {
  id: string;
  timestamp: string;
  level: 'info' | 'warn' | 'error' | 'success';
  source: string;
  message: string;
}
