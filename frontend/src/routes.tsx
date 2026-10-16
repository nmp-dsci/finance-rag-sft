import { Link, type RouteObject, useRouteError } from 'react-router-dom';
import { Shell } from './Shell';
import { Data, DataPassage, DataQuestion } from './pages/Data';
import { Method } from './pages/Method';
import { Overview } from './pages/Overview';
import { RunQuestion } from './pages/Question';
import { Run, Runs } from './pages/Runs';
import { Search } from './pages/Search';

/**
 * Every viewer address, as data. tau2-loop's grammar: one id per thing, the path names the
 * subject, the query string holds the lens (dim, filter, q, vs, offset), and a detail opens
 * inside its list (`/data/q/:qid` renders into the Data page's list).
 */
export const routes: RouteObject[] = [
  {
    id: 'shell',
    element: <Shell />,
    errorElement: <Crash />,
    children: [
      { id: 'overview', path: '/', element: <Overview /> },
      { id: 'method', path: '/method', element: <Method /> },
      {
        id: 'data',
        path: '/data',
        element: <Data />,
        children: [
          { id: 'data-question', path: 'q/:qid', element: <DataQuestion /> },
          { id: 'data-passage', path: 'p/:pid', element: <DataPassage /> },
        ],
      },
      { id: 'runs', path: '/runs', element: <Runs /> },
      { id: 'run', path: '/runs/:runId', element: <Run /> },
      { id: 'run-question', path: '/runs/:runId/:qid', element: <RunQuestion /> },
      { id: 'search', path: '/search', element: <Search /> },
      { id: 'missing', path: '*', element: <NotFound /> },
    ],
  },
];

function NotFound() {
  return (
    <div className="empty">
      <h1>No page here</h1>
      <p>
        Try the <Link to="/">Overview</Link> or the <Link to="/runs">Runs</Link>.
      </p>
    </div>
  );
}

function Crash() {
  const err = useRouteError() as Error | undefined;
  return (
    <main>
      <div className="callout red" role="alert">
        <span className="badge bad">✕ error</span> The viewer hit an error: <span className="mono">{err?.message ?? String(err)}</span>
      </div>
      <p>
        <a href="/">Back to the overview</a>
      </p>
    </main>
  );
}
