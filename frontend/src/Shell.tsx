import { NavLink, Outlet } from 'react-router-dom';
import { type Health, api, useGet } from './lib/api';

/** The tabs, in the order the story runs: the answer → how it was made → the data → the runs → try it. */
const NAV: [string, string][] = [
  ['/', 'Overview'],
  ['/method', 'Method'],
  ['/data', 'Data'],
  ['/runs', 'Runs'],
  ['/search', 'Search'],
];

export function Shell() {
  const { data: health, error } = useGet<Health>(api.health());
  return (
    <>
      <header className="appbar">
        <div className="in">
          <NavLink to="/" className="brand">
            finance-rag-sft
          </NavLink>
          <nav aria-label="Pages">
            {NAV.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => (isActive ? 'on' : '')}>
                {label}
              </NavLink>
            ))}
          </nav>
          <HealthPill health={health} error={error} />
        </div>
      </header>
      <main>
        <Outlet />
      </main>
      <footer className="appfoot">
        <div className="in">
          finance-rag-sft · fine-tuning <code>BAAI/bge-base-en-v1.5</code> on finance (question, passage) pairs ·
          scored on split v1 · build <code>{health?.code_sha ?? '…'}</code>
        </div>
      </footer>
    </>
  );
}

function HealthPill({ health, error }: { health: Health | null; error: string | null }) {
  if (error)
    return (
      <span className="pill" title={error}>
        <span className="badge bad">✕ api down</span>
      </span>
    );
  if (!health)
    return (
      <span className="pill">
        <span className="badge">api …</span>
      </span>
    );
  return (
    <span className="pill">
      <span className="badge ok" title="FastAPI on :8084">
        api ok · {health.code_sha}
      </span>
      <span
        className={`badge ${health.mlflow === 'up' ? 'ok' : 'warn'}`}
        title="The central MLflow: the viewer links to it, it never reads from it"
      >
        mlflow {health.mlflow === 'up' ? '● up' : '○ down'}
      </span>
    </span>
  );
}
