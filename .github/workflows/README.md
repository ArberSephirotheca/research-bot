# .github/workflows/

Automated runs of the research bot. `daily.yml` schedules the digest and pushes updates back to the repo.

## Required secrets

- `OPENAI_API_KEY`: used when `--summarize` or `--discover` is enabled in the workflow.

## Environment protection

The workflow uses the `openai` environment. Configure it in GitHub Settings -> Environments to:

- Require manual approvals before jobs run.
- Store `OPENAI_API_KEY` as an environment secret if you prefer tighter access.
