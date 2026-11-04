// Shapes of the backend responses (backend/synapse/api/chatgpt_api.py).

export interface SystemInfo {
  project: string;
  version: string;
  hardware: {
    cpu: string;
    cpu_cores: number;
    ram_total_gb: number;
    ram_available_gb: number;
    gpu: string | null;
    gpu_vram_gb: number;
    unified_memory: boolean;
    gpu_count: number;
    backend: string;
    os: string;
  };
  network_discovery: string;
  attention_backends: string[];
}

export interface SystemStats {
  cpu: number;
  memory: number;
  memory_used_gb: number;
  memory_total_gb: number;
  uptime: string;
  gpu: string;
  gpu_name?: string;
  gpu_vram_gb?: number;
  gpu_utilization: number | null;
  gpu_vram_used_mb?: number;
  gpu_temp_c?: number;
  gpu_power_w?: number;
  disk_percent?: number;
  disk_used_gb?: number;
  disk_total_gb?: number;
  net_sent_mb?: number;
  net_recv_mb?: number;
  swap_percent?: number;
  process_count?: number;
}

export interface ClusterNode {
  node_id: string;
  name: string;
  is_self: boolean;
  ip: string;
  chip: string;
  gpu_backend: string;
  has_gpu: boolean;
  vram_gb: number;
  ram_gb: number;
  flops_fp16: number;
  layer_start: number | null;
  layer_end: number | null;
  layer_count: number;
  share_pct: number;
  cpu_cores: number;
  cpu_pct: number | null;
  gpu_utilization: number | null;
  ram_used_gb: number | null;
  ram_total_gb: number | null;
}

export interface ClusterResources {
  nodes: ClusterNode[];
  total_nodes: number;
  total_layers: number;
  strategy_name: string;
  latency_matrix: Record<string, Record<string, number>>;
  peer_graph: Record<string, Array<{ from_id: string; to_id: string; description?: string }>>;
  training_job: TrainingJob | null;
}

export interface AutoscaleHint {
  action: string;
  model: string;
  node?: string;
  reason: string;
}

export interface DistributedStatus {
  partitions_count: number;
  multi_node_inference: boolean;
  multi_node_training: boolean;
  node_id?: string;
  master_id?: string | null;
  is_master?: boolean;
  tls_mode?: string;
  api_key_required?: boolean;
  autoscale_mode?: string;
  autoscale_objective?: string | null;
  autoscale_hints?: AutoscaleHint[];
  partition_objective?: string;
  standby_nodes?: string[];
  transport_counts?: Record<string, number>;
  quant_active?: string | null;
  recovery_degraded?: Record<string, unknown>;
  bridge_version?: number;
  peer_versions?: Record<string, { bridge_version?: number; synapse_version?: string }>;
}

export interface ModelEntry {
  name: string;
  status: string;
  repo_id: string;
  parameter_size?: string;
  family?: string;
  memory?: string;
  architecture?: string;
  size_gb?: number;
  hf_downloads?: number;
}

export interface ModelsList {
  downloaded: ModelEntry[];
  available: ModelEntry[];
}

export interface Dataset {
  name: string;
  path: string;
  size_mb: number;
  created: string;
  sample_count: number | null;
  type: string;
}

export interface TrainingJob {
  job_id: string;
  model: string;
  dataset: string;
  status: string;
  epochs: number;
  batch_size: number;
  learning_rate?: string | null;
  progress?: number;
  current_step?: number;
  current_epoch?: number;
  loss?: number;
  error?: string;
  use_lora?: boolean;
  lora_applied?: boolean;
  completed_at?: string;
  started_at?: string;
  max_steps?: number;
  loss_history?: Array<[number, number]>;
  preprocess?: Record<string, unknown>;
}

export interface RequestMetric {
  id: string;
  model: string;
  stream: boolean;
  at: number;          // unix seconds, request end
  tokens: number;
  total_s: number;
  ttft_s: number | null;
  tok_s: number | null;
  error: string | null;
}

export interface SearchResult {
  name: string;
  repo: string;
  pipeline_tag?: string;
  downloads?: number;
  likes?: number;
  last_modified?: string;
  isDownloaded?: boolean;
}

export interface CompatModel {
  name: string;
  parameters: string;
  quantization: string;
  context_length: number;
  memory_needed_gb: number;
  fit: string;
  fit_score: number;
  estimated_speed: number;
}

export interface Compatibility {
  total: number;
  page: number;
  total_pages: number;
  budget_gb: number;
  hardware: { ram_total_gb: number; ram_available_gb: number; gpu: string | null; gpu_vram_gb: number };
  models: CompatModel[];
}

export interface Checkpoint {
  name: string;
  path: string;
  type?: 'adapter' | 'full';
  created?: string;
  [k: string]: unknown;
}

export interface DatasetValidation {
  valid_count?: number;
  total_lines?: number;
  errors?: Array<string | { line: number; message: string }>;
  format_detected?: string;
  [k: string]: unknown;
}

export interface TrainingStatus {
  status: string;
  job: TrainingJob | null;
}

export interface ActivityEntry {
  time: string;
  action: string;
  model: string;
  status: string;
}

export interface Settings {
  default_model: string;
  discovery_module: string;
  lacp_priority: string;
  load_quantization: string;
  lookahead_window: number;
  transport_quantization: string;
}

export interface Pairing {
  paired: boolean;
  code: string | null;
  tls_mode: string;
  error: string | null;
  restart_needed: boolean;
}
