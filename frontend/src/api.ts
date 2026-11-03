// Thin client for the Synapse API. No mock data: a failed call raises
// ApiError and the page shows the error.
import type {
  ActivityEntry, ClusterResources, Dataset, DistributedStatus, ModelsList, Pairing,
  Settings, SystemInfo, SystemStats, TrainingStatus,
} from './types';

const KEY_STORAGE = 'synapse.apiKey';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
  /** Admin route refused: no key, or not opened from the node's own machine. */
  get forbidden(): boolean {
    return this.status === 401 || this.status === 403;
  }
}

export function getApiKey(): string {
  try { return localStorage.getItem(KEY_STORAGE) || ''; } catch { return ''; }
}

export function setApiKey(key: string): void {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch { /* storage blocked: key lasts for this page only */ }
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const key = getApiKey();
  return key ? { ...extra, Authorization: `Bearer ${key}` } : extra;
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      headers: headers(body === undefined || body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      body: body === undefined ? undefined : body instanceof FormData ? body : JSON.stringify(body),
    });
  } catch (e) {
    throw new ApiError(0, `Không kết nối được tới node (${(e as Error).message})`);
  }
  let data: any = null;
  const text = await res.text();
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const msg = (data && (data.error || data.message || data.detail)) || `HTTP ${res.status}`;
    throw new ApiError(res.status, String(msg));
  }
  if (data && data.success === false) {
    throw new ApiError(res.status, String(data.message || data.error || 'Thao tác thất bại'));
  }
  return data as T;
}

export const api = {
  systemInfo: () => call<SystemInfo>('GET', '/v1/system/info'),
  systemStats: () => call<SystemStats>('GET', '/v1/system/stats'),
  cluster: () => call<ClusterResources>('GET', '/v1/cluster/resources'),
  distributed: () => call<DistributedStatus>('GET', '/v1/distributed/status'),

  models: () => call<ModelsList>('GET', '/v1/models/list'),
  pullModel: (model: string, full = false) =>
    call<{ message: string }>('POST', '/v1/models/pull', { model, full }),
  deleteModel: (model: string) => call<{ message: string }>('POST', '/v1/models/delete', { model }),
  downloadProgress: () => call<{ progress: Record<string, any> }>('GET', '/v1/download/progress'),

  datasets: () => call<{ data: Dataset[] }>('GET', '/v1/datasets').then(r => r.data),
  uploadDataset: (file: File) => {
    const fd = new FormData();
    fd.append('file', file);
    return call<{ name: string }>('POST', '/v1/datasets/upload', fd);
  },
  deleteDataset: (path: string) => call<{ name: string }>('POST', '/v1/datasets/delete', { path }),
  previewDataset: (path: string, lines = 20) =>
    call<{ preview: string[] }>('GET', `/v1/datasets/preview?path=${encodeURIComponent(path)}&lines=${lines}`)
      .then(r => r.preview),

  trainingStatus: () => call<TrainingStatus>('GET', '/v1/training/status'),
  startTraining: (body: Record<string, unknown>) =>
    call<{ job_id?: string; message?: string }>('POST', '/v1/training/start', body),

  activity: () => call<{ data: ActivityEntry[] }>('GET', '/v1/activity').then(r => r.data),
  terminalLogs: () => call<{ data: string[] }>('GET', '/v1/logs/terminal').then(r => r.data),

  settings: () => call<Settings>('GET', '/v1/settings'),
  saveSettings: (s: Partial<Settings>) =>
    call<{ restarted: boolean }>('POST', '/v1/settings', s),

  pairing: () => call<Pairing>('GET', '/v1/cluster/pairing'),
  pair: (action: 'create' | 'join' | 'forget', code?: string) =>
    call<{ code: string | null }>('POST', '/v1/cluster/pairing', { action, code }),
};

export interface ChatStats {
  ttftMs: number | null;
  totalMs: number;
  tokens: number;
}

/** Stream a chat completion; onDelta gets each piece of text as it arrives. */
export async function streamChat(
  model: string,
  messages: Array<{ role: string; content: string }>,
  onDelta: (text: string) => void,
  signal?: AbortSignal,
): Promise<ChatStats> {
  const t0 = performance.now();
  let res: Response;
  try {
    res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ model, messages, stream: true, temperature: 0.7 }),
      signal,
    });
  } catch (e) {
    throw new ApiError(0, `Không kết nối được tới node (${(e as Error).message})`);
  }
  if (!res.ok || !res.body) {
    let msg = `HTTP ${res.status}`;
    try { const d = await res.json(); msg = d.error?.message || d.error || d.detail || msg; } catch { /* keep */ }
    throw new ApiError(res.status, String(msg));
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let ttft: number | null = null;
  let tokens = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') continue;
      let obj: any;
      try { obj = JSON.parse(payload); } catch { continue; }
      if (obj.usage?.completion_tokens) tokens = obj.usage.completion_tokens;
      const text = obj.choices?.[0]?.delta?.content ?? obj.choices?.[0]?.text ?? '';
      if (text) {
        if (ttft === null) ttft = performance.now() - t0;
        onDelta(text);
      }
    }
  }
  return { ttftMs: ttft, totalMs: performance.now() - t0, tokens };
}
