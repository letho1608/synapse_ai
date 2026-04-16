import { useModelContext, MAX_COMPARE } from '../contexts/ModelContext';
import { round, fitClass, modeClass, copyModelName } from '../utils';

export default function ModelTable() {
  const {
    models,
    loading,
    error,
    selectedModelName,
    setSelectedModelName,
    compareList,
    toggleCompare,
    installedModels
  } = useModelContext();

  const installedSet = new Set(
    Array.isArray(installedModels) ? installedModels : []
  );
  const compareFull = compareList.length >= MAX_COMPARE;

  return (
    <div className="table-wrap">
      {error ? (
        <div role="alert" className="alert error" style={{ margin: '0.75rem' }}>
          Could not load models: {error}. Confirm this page is opened from
          `llmfit serve`.
        </div>
      ) : null}

      <table>
        <thead>
          <tr>
            <th className="col-compare">Cmp</th>
            <th>Model</th>
            <th>Provider</th>
            <th>Params</th>
            <th>Fit</th>
            <th>Mode</th>
            <th>Runtime</th>
            <th>Score</th>
            <th>TPS</th>
            <th>Mem%</th>
            <th>Context</th>
            <th>Release</th>
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr>
              <td colSpan="12" className="table-status">
                Loading model fit data&hellip;
              </td>
            </tr>
          ) : null}

          {!loading && models.length === 0 && !error ? (
            <tr>
              <td colSpan="12" className="table-status">
                No models match the current filters.
              </td>
            </tr>
          ) : null}

          {!loading
            ? models.map((model) => {
                const isSelected = model.name === selectedModelName;
                const isCompared = compareList.includes(model.name);
                const isInstalled = installedSet.has(model.name);
                const disableCompare = !isCompared && compareFull;

                return (
                  <tr
                    key={model.name}
                    className={isSelected ? 'selected' : ''}
                    onClick={() => setSelectedModelName(model.name)}
                  >
                    <td
                      className="col-compare"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <input
                        type="checkbox"
                        className="compare-checkbox"
                        checked={isCompared}
                        disabled={disableCompare}
                        onChange={() => toggleCompare(model.name)}
                        title={
                          disableCompare
                            ? `Max ${MAX_COMPARE} models for comparison`
                            : 'Add to comparison'
                        }
                      />
                    </td>
                    <td className="model-name">
                      <span>{model.name}</span>
                      {isInstalled && (
                        <span className="chip chip-installed">Installed</span>
                      )}
                      <button
                        type="button"
                        className="btn-copy"
                        title="Copy model name"
                        onClick={(e) => {
                          e.stopPropagation();
                          copyModelName(model.name);
                        }}
                      >
                        &#x2398;
                      </button>
                    </td>
                    <td>{model.provider}</td>
                    <td>{round(model.params_b, 1)}B</td>
                    <td>
                      <span className={fitClass(model.fit_level)}>
                        {model.fit_label}
                      </span>
                    </td>
                    <td>
                      <span className={modeClass(model.run_mode)}>
                        {model.run_mode_label}
                      </span>
                    </td>
                    <td>{model.runtime_label}</td>
                    <td>{round(model.score, 1)}</td>
                    <td>{round(model.estimated_tps, 1)}</td>
                    <td>{round(model.utilization_pct, 1)}</td>
                    <td>
                      {model.context_length?.toLocaleString?.() ??
                        model.context_length ??
                        '\u2014'}
                    </td>
                    <td>{model.release_date ?? '\u2014'}</td>
                  </tr>
                );
              })
            : null}
        </tbody>
      </table>
    </div>
  );
}
