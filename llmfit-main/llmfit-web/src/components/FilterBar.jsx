import { useMemo, useRef, useState, useEffect } from 'react';
import { useFilters, useFilterDispatch } from '../contexts/FilterContext';
import { useModelContext } from '../contexts/ModelContext';
import { collectUniqueValues } from '../utils';

const FIT_OPTIONS = [
  { value: 'marginal', label: 'Runnable (Marginal+)' },
  { value: 'good', label: 'Good or better' },
  { value: 'perfect', label: 'Perfect only' },
  { value: 'too_tight', label: 'Too-tight only' },
  { value: 'all', label: 'All levels' }
];

const RUNTIME_OPTIONS = [
  { value: 'any', label: 'Any runtime' },
  { value: 'mlx', label: 'MLX' },
  { value: 'llamacpp', label: 'llama.cpp' },
  { value: 'vllm', label: 'vLLM' }
];

const USE_CASE_OPTIONS = [
  { value: 'all', label: 'All use cases' },
  { value: 'general', label: 'General' },
  { value: 'coding', label: 'Coding' },
  { value: 'reasoning', label: 'Reasoning' },
  { value: 'chat', label: 'Chat' },
  { value: 'multimodal', label: 'Multimodal' },
  { value: 'embedding', label: 'Embedding' }
];

const LIMIT_OPTIONS = [
  { value: '10', label: '10' },
  { value: '20', label: '20' },
  { value: '50', label: '50' },
  { value: '100', label: '100' },
  { value: '200', label: '200' },
  { value: '', label: 'All' }
];

const SORT_OPTIONS = [
  { value: 'score', label: 'Sort: Score' },
  { value: 'tps', label: 'Sort: TPS' },
  { value: 'params', label: 'Sort: Params' },
  { value: 'mem', label: 'Sort: Memory' },
  { value: 'ctx', label: 'Sort: Context' },
  { value: 'date', label: 'Sort: Release date' },
  { value: 'use_case', label: 'Sort: Use case' }
];

const PARAMS_BUCKET_OPTIONS = [
  { value: 'all', label: 'All sizes' },
  { value: 'tiny', label: 'Tiny (<3B)' },
  { value: 'small', label: 'Small (3-8B)' },
  { value: 'medium', label: 'Medium (8-30B)' },
  { value: 'large', label: 'Large (30-70B)' },
  { value: 'xl', label: 'XL (70B+)' }
];

const TP_OPTIONS = [
  { value: 'all', label: 'Any TP' },
  { value: '1', label: 'TP=1' },
  { value: '2', label: 'TP=2' },
  { value: '4', label: 'TP=4' },
  { value: '8', label: 'TP=8' }
];

function MultiSelectDropdown({ label, field, options }) {
  const filters = useFilters();
  const dispatch = useFilterDispatch();
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  const selected = filters[field] || [];
  const count = selected.length;

  useEffect(() => {
    function handleClickOutside(e) {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open]);

  function toggle(value) {
    const next = selected.includes(value)
      ? selected.filter((v) => v !== value)
      : [...selected, value];
    dispatch({ type: 'SET_FILTER', field, value: next });
  }

  return (
    <div className="multi-select-wrap" ref={ref}>
      <span>{label}</span>
      <button
        type="button"
        className="multi-select-btn"
        onClick={() => setOpen((o) => !o)}
      >
        {count > 0 ? `${count} selected` : 'Any'}
        <span className="multi-select-caret">{open ? '\u25B2' : '\u25BC'}</span>
      </button>
      {open && (
        <div className="multi-select-popover">
          {options.length === 0 ? (
            <p className="multi-select-empty">No options available</p>
          ) : (
            options.map((opt) => (
              <label key={opt} className="multi-select-option">
                <input
                  type="checkbox"
                  checked={selected.includes(opt)}
                  onChange={() => toggle(opt)}
                />
                <span>{opt}</span>
              </label>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default function FilterBar() {
  const filters = useFilters();
  const dispatch = useFilterDispatch();
  const { allModels } = useModelContext();

  const handleChange = (field) => (e) => {
    const value =
      e.target.type === 'checkbox' ? e.target.checked : e.target.value;
    dispatch({ type: 'SET_FILTER', field, value });
  };

  const availableCapabilities = useMemo(
    () => collectUniqueValues(allModels, 'capabilities'),
    [allModels]
  );

  const availableQuants = useMemo(
    () => collectUniqueValues(allModels, 'best_quant'),
    [allModels]
  );

  const availableRunModes = useMemo(
    () => collectUniqueValues(allModels, 'run_mode'),
    [allModels]
  );

  const advancedCount =
    (filters.capability.length > 0 ? 1 : 0) +
    (filters.license ? 1 : 0) +
    (filters.quant.length > 0 ? 1 : 0) +
    (filters.runMode.length > 0 ? 1 : 0) +
    (filters.paramsBucket !== 'all' ? 1 : 0) +
    (filters.tp !== 'all' ? 1 : 0) +
    (filters.maxContext ? 1 : 0);

  return (
    <div className="filters-outer">
      <div className="filters-shell">
        <label>
          <span>Search</span>
          <input
            type="text"
            value={filters.search}
            onChange={handleChange('search')}
            placeholder="model, provider, use case"
          />
        </label>

        <label>
          <span>Fit filter</span>
          <select value={filters.minFit} onChange={handleChange('minFit')}>
            {FIT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Runtime</span>
          <select value={filters.runtime} onChange={handleChange('runtime')}>
            {RUNTIME_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Use case</span>
          <select value={filters.useCase} onChange={handleChange('useCase')}>
            {USE_CASE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Provider</span>
          <input
            type="text"
            value={filters.provider}
            onChange={handleChange('provider')}
            placeholder="Meta, Qwen, Mistral"
          />
        </label>

        <label>
          <span>Sort</span>
          <select value={filters.sort} onChange={handleChange('sort')}>
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Limit</span>
          <select
            value={String(filters.limit)}
            onChange={handleChange('limit')}
          >
            {LIMIT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="filters-toggle-row">
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={() =>
            dispatch({
              type: 'SET_FILTER',
              field: 'showAdvanced',
              value: !filters.showAdvanced
            })
          }
        >
          {filters.showAdvanced ? 'Fewer filters' : 'More filters'}
          {advancedCount > 0 ? ` (${advancedCount} active)` : ''}
        </button>
      </div>

      {filters.showAdvanced && (
        <div className="filters-shell filters-advanced">
          <MultiSelectDropdown
            label="Capability"
            field="capability"
            options={availableCapabilities}
          />

          <label>
            <span>License</span>
            <input
              type="text"
              value={filters.license}
              onChange={handleChange('license')}
              placeholder="apache-2.0, mit, ..."
            />
          </label>

          <MultiSelectDropdown
            label="Quantization"
            field="quant"
            options={availableQuants}
          />

          <MultiSelectDropdown
            label="Run mode"
            field="runMode"
            options={availableRunModes}
          />

          <label>
            <span>Params bucket</span>
            <select
              value={filters.paramsBucket}
              onChange={handleChange('paramsBucket')}
            >
              {PARAMS_BUCKET_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Tensor Parallel</span>
            <select value={filters.tp} onChange={handleChange('tp')}>
              {TP_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Max context</span>
            <input
              type="number"
              value={filters.maxContext}
              onChange={handleChange('maxContext')}
              placeholder="e.g. 32768"
              min="0"
            />
          </label>
        </div>
      )}
    </div>
  );
}
