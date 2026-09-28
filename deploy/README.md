# deploy/

`github-pages-workflow.yml` is **parked and unused**. It is the GitHub Actions workflow
that would publish `site/` to Pages. The token I build with has no `workflow` scope, so
GitHub rejects any push touching `.github/workflows/**`, and the Pages REST enablement
endpoint is unreachable from my environment.

The site is published by `scripts/deploy_pages.sh` instead. It force-pushes `site/` to
the root of a `gh-pages` branch, which auto-enables Pages, and mirrors it into `docs/`
on `main`. If your token can install workflows, move this file to
`.github/workflows/pages.yml` and it produces the same output.
