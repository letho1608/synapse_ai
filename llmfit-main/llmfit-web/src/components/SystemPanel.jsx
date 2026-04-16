import { useModelContext } from '../contexts/ModelContext';
import { round } from '../utils';

function SystemCard({ label, value, detail }) {
  return (
    <article className="system-card">
      <p className="system-label">{label}</p>
      <p className="system-value">{value}</p>
      {detail ? <p className="system-detail">{detail}</p> : null}
    </article>
  );
}

export default function SystemPanel() {
  const { systemInfo, systemLoading, systemError } = useModelContext();

  const gpus = systemInfo?.system?.gpus ?? [];
  const gpuSummary =
    gpus.length === 0
      ? 'No GPU detected'
      : gpus
          .map(
            (gpu) =>
              `${gpu.name}${gpu.vram_gb ? ` (${round(gpu.vram_gb, 1)} GB)` : ''}`
          )
          .join(', ');

  return (
    <section className="panel system-panel">
      <div className="panel-heading">
        <h2>System Summary</h2>
        {systemInfo?.node ? (
          <span className="chip">
            {systemInfo.node.name} &middot; {systemInfo.node.os}
          </span>
        ) : null}
      </div>

      {systemError ? (
        <div role="alert" className="alert error">
          Could not load system information: {systemError}. Make sure `llmfit
          serve` is running.
        </div>
      ) : null}

      <div className="system-grid" aria-busy={systemLoading}>
        <SystemCard
          label="CPU"
          value={systemInfo?.system?.cpu_name ?? 'Loading\u2026'}
          detail={
            systemInfo?.system?.cpu_cores
              ? `${systemInfo.system.cpu_cores} cores`
              : undefined
          }
        />
        <SystemCard
          label="Total RAM"
          value={
            systemInfo?.system?.total_ram_gb
              ? `${round(systemInfo.system.total_ram_gb, 1)} GB`
              : '\u2014'
          }
        />
        <SystemCard
          label="Available RAM"
          value={
            systemInfo?.system?.available_ram_gb
              ? `${round(systemInfo.system.available_ram_gb, 1)} GB`
              : '\u2014'
          }
        />
        <SystemCard
          label="GPU"
          value={gpuSummary}
          detail={
            systemInfo?.system?.unified_memory
              ? 'Unified memory (CPU + GPU shared)'
              : undefined
          }
        />
      </div>
    </section>
  );
}
