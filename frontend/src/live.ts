// Live cluster data shared by every page: one poller, a few minutes of history,
// so charts keep their past when the user switches pages.
import { api } from './api';
import type { ClusterResources, DistributedStatus, RequestMetric, SystemStats } from './types';

export interface Sample { t: number; v: number | null }

const HISTORY_S = 300;
const POLL_MS = 3000;

export class LiveStore {
  cluster: ClusterResources | null = null;
  dist: DistributedStatus | null = null;
  stats: SystemStats | null = null;
  error: unknown = null;
  /** node id -> series, oldest first */
  cpu = new Map<string, Sample[]>();
  ram = new Map<string, Sample[]>();
  requests: RequestMetric[] = [];
  /** Stable color slot per node id, in order of first sight (color follows the entity). */
  private slots = new Map<string, number>();
  private listeners = new Set<() => void>();
  private lastMetricAt = 0;
  private timer: number | undefined;
  private inFlight = false;

  start() {
    if (this.timer !== undefined) return;
    this.tick();
    this.timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') this.tick();
    }, POLL_MS);
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  slot(nodeId: string): number {
    if (!this.slots.has(nodeId)) this.slots.set(nodeId, this.slots.size);
    return this.slots.get(nodeId)!;
  }

  /** Nodes currently in the cluster, in color-slot order. */
  nodeIds(): string[] {
    const ids = (this.cluster?.nodes || []).map(n => n.node_id);
    ids.forEach(id => this.slot(id));
    return ids.sort((a, b) => this.slot(a) - this.slot(b));
  }

  /** True while a request finished in the last few seconds (drives the 3D flow animation). */
  busy(): boolean {
    const last = this.requests[this.requests.length - 1];
    return !!last && Date.now() / 1000 - last.at < 6;
  }

  private push(map: Map<string, Sample[]>, id: string, t: number, v: number | null) {
    const s = map.get(id) || [];
    s.push({ t, v });
    while (s.length && s[0].t < t - HISTORY_S) s.shift();
    map.set(id, s);
  }

  private emit() {
    this.listeners.forEach(fn => { try { fn(); } catch (e) { console.error(e); } });
  }

  /** Each response updates the store and the page as soon as it lands, so a
   *  slow route (cluster resources while the node is busy) never holds back
   *  the fast ones. */
  async tick() {
    if (this.inFlight) return;
    this.inFlight = true;
    const now = Date.now() / 1000;
    const jobs = [
      api.cluster().then(c => {
        this.cluster = c;
        this.error = null;
        // New ids take slots in sorted order, so the same cluster gets the
        // same colors on every page load (response order is not stable).
        c.nodes.map(n => n.node_id).sort().forEach(id => this.slot(id));
        for (const n of c.nodes) {
          this.push(this.cpu, n.node_id, now, n.cpu_pct);
          const total = n.ram_total_gb ?? n.ram_gb;
          this.push(this.ram, n.node_id, now,
            n.ram_used_gb !== null && total ? (100 * n.ram_used_gb) / total : null);
        }
      }, e => { this.error = e; }),
      api.distributed().then(d => { this.dist = d; }, () => {}),
      api.systemStats().then(s => { this.stats = s; }, () => {}),
      api.requestMetrics(this.lastMetricAt).then(m => {
        if (!m.length) return;
        this.requests.push(...m);
        this.lastMetricAt = m[m.length - 1].at;
        if (this.requests.length > 200) this.requests.splice(0, this.requests.length - 200);
      }, () => {}),
    ].map(p => p.then(() => this.emit()));
    try { await Promise.all(jobs); } finally { this.inFlight = false; }
  }
}

export const live = new LiveStore();
