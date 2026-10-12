import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider, createBrowserRouter } from 'react-router-dom';
// The house UI, vendored by `make install-house-ui` into ../.lavish: tokens and components
// first, then the project's one number (--hue 15). house.css @imports fonts/plex.css with
// relative URLs, so Vite rebases them and bundles the IBM Plex woff2 files. Nothing in src/
// declares a colour of its own (scripts/design_lint.mjs enforces it).
import '../../.lavish/house.css';
import '../../.lavish/project.css';
import './styles.css';
import { routes } from './routes';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={createBrowserRouter(routes)} />
  </React.StrictMode>,
);
