# Dashboard security regression

Requires Node.js, the `playwright` package, and Microsoft Edge.

From the repository root:

```sh
npm install --no-save --package-lock=false playwright
node --test tests/dashboard-security.cjs
```

The test runs Edge headlessly and blocks external network requests. A minimal
Tailwind stub supplies the modal visibility utility; application scripts execute
unchanged. It covers label editing and localStorage reload, payment modal text,
HTML export and re-export in fresh browser contexts, script-closing payloads,
exclusion of transient DOM, AUM persistence, and empty exported plans.

Set `DASHBOARD_PATH` to another dashboard HTML file to test an earlier revision.
