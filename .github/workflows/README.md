# .github/workflows/

Automated runs of the research bot. `daily.yml` schedules the digest and pushes updates back to the repo.
`rag_artifacts.yml` builds paper PDFs + RAG embeddings and uploads them as artifacts (so you can sync them to a server without committing large files).

## Required secrets

- `OPENAI_API_KEY`: used when `--summarize` or `--discover` is enabled in the workflow.
- `OPENAI_API_KEY`: also required by `rag_artifacts.yml` to build embeddings.

## Optional variables

- `OPENAI_BASE_URL`: override API base URL for embeddings.
- `OPENAI_EMBED_MODEL`: override embedding model (default: `text-embedding-3-small`).

## Environment protection

The workflow uses the `openai` environment. Configure it in GitHub Settings -> Environments to:

- Require manual approvals before jobs run.
- Store `OPENAI_API_KEY` as an environment secret if you prefer tighter access.

## Artifacts workflow

`rag_artifacts.yml` runs on a daily schedule and on manual dispatch. It uploads an artifact named `rag-artifacts` that contains:

- `papers/` (downloaded PDFs)
- `rag/chunks.jsonl`
- `rag/embeddings.jsonl`
