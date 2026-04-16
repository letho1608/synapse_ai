import { useState, useEffect } from 'react';
import { useFilterDispatch } from '../contexts/FilterContext';
import { useModelContext } from '../contexts/ModelContext';

const THEME_KEY = 'llmfit-theme';

const THEMES = [
  { value: 'default', label: 'Default' },
  { value: 'dracula', label: 'Dracula' },
  { value: 'solarized', label: 'Solarized' },
  { value: 'nord', label: 'Nord' },
  { value: 'monokai', label: 'Monokai' },
  { value: 'gruvbox', label: 'Gruvbox' },
  { value: 'catppuccin-latte', label: 'Catppuccin Latte' },
  { value: 'catppuccin-frappe', label: 'Catppuccin Frappé' },
  { value: 'catppuccin-macchiato', label: 'Catppuccin Macchiato' },
  { value: 'catppuccin-mocha', label: 'Catppuccin Mocha' },
];

function initialTheme() {
  if (typeof window === 'undefined') {
    return 'default';
  }

  const stored = window.localStorage.getItem(THEME_KEY);
  if (stored && THEMES.some((t) => t.value === stored)) {
    return stored;
  }

  // Legacy light/dark mapping
  if (stored === 'light') return 'catppuccin-latte';
  if (stored === 'dark') return 'default';

  return window.matchMedia?.('(prefers-color-scheme: light)').matches
    ? 'catppuccin-latte'
    : 'default';
}

export default function Header() {
  const [theme, setTheme] = useState(initialTheme);
  const dispatch = useFilterDispatch();
  const { triggerRefresh } = useModelContext();

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  return (
    <header className="hero-shell">
      <div>
        <p className="hero-eyebrow">Local LLM Planning</p>
        <h1>llmfit Dashboard</h1>
        <p className="hero-copy">
          Hundreds of models &amp; providers. One command to find what runs on
          your hardware.
        </p>
      </div>

      <div className="hero-actions">
        <button
          type="button"
          onClick={() => dispatch({ type: 'RESET_FILTERS' })}
          className="btn btn-ghost"
        >
          Reset filters
        </button>
        <button
          type="button"
          onClick={triggerRefresh}
          className="btn btn-accent"
        >
          Refresh
        </button>
        <select
          value={theme}
          onChange={(e) => setTheme(e.target.value)}
          className="btn btn-theme theme-select"
          aria-label="Theme"
        >
          {THEMES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>
      </div>
    </header>
  );
}
