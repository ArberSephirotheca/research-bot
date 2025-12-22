# GPU Research Bot

A small Rust CLI that pulls GPU-related feeds, writes a daily Markdown digest, and tracks what has already been seen so runs stay deduplicated. The repo is designed to be driven entirely by agents ("code is lava"), so every subdirectory includes its own README.

## Quick start

1) Review and edit `data/sources.yml`.
2) Run the bot:

```bash
cargo run -- --config data/sources.yml --db data/visited.jsonl --max-age-days 90
```

3) Read the report in `reports/YYYY-MM-DD.md`.
4) To clear the dedupe database for a fresh run, add `--reset-db`.
5) To enable LLM summaries, set `OPENAI_API_KEY` and add `--summarize`.

## LLM summaries

```bash
export OPENAI_API_KEY=your-key-here
cargo run -- --config data/sources.yml --db data/visited.jsonl --summarize
```

You can also put keys in a local `.env` file (auto-loaded by `dotenvy`):

```bash
OPENAI_API_KEY=your-key-here
```

## Hybrid web search

The hybrid mode uses OpenAI to generate search queries and OpenAI web_search to fetch web results.

```bash
export OPENAI_API_KEY=your-key-here
cargo run -- --config data/sources.yml --db data/visited.jsonl --discover --summarize
```

Tune discovery topics and limits in `data/discovery.yml`.

## Secrets setup

GitHub Actions reads `OPENAI_API_KEY` from secrets. Prefer the environment secret so you can gate runs:

1) Settings -> Environments -> `openai` -> Add secret `OPENAI_API_KEY`
2) (Optional) Require approvals in the `openai` environment

Or use a repo secret:

1) Settings -> Secrets and variables -> Actions -> New repository secret
2) Name: `OPENAI_API_KEY`

## GitHub Actions

A scheduled workflow runs daily and commits new reports plus the visited DB back to the repo. Make sure the repo allows workflows to write:

- Settings -> Actions -> General -> Workflow permissions -> Read and write

The workflow is at `.github/workflows/daily.yml`.
If `--summarize` or `--discover` is enabled in the workflow, add `OPENAI_API_KEY` in repo or environment settings.
The workflow uses the `openai` environment so you can require approvals and keep the key scoped.
Use a dedicated OpenAI key with strict spend limits to reduce blast radius.

## Layout

- `src/` Rust CLI implementation
- `data/` source configuration + visited database + discovery config
- `reports/` daily digests (generated)
- `.github/workflows/` scheduled GitHub Actions run

## Config format

`data/sources.yml` lists feed URLs (RSS/Atom). Example:

```yaml
sources:
  - name: "arXiv: GPU OR CUDA"
    url: "http://export.arxiv.org/api/query?search_query=all:GPU+OR+all:CUDA&start=0&max_results=50"
```

`data/discovery.yml` controls hybrid web search. Example:

```yaml
topics:
  - GPU architecture
  - matrix multiplication
  - CUDA
max_queries: 4
max_results_per_query: 5
```

## Notes

- The report shows up to `--max-per-source` items per source, but the visited DB tracks all new items so they do not repeat.
- `--max-age-days` filters out items older than the window; entries without dates are skipped.
- `--summarize` calls OpenAI to produce 2-3 bullet summaries per item. Requires `OPENAI_API_KEY`.
- `--discover` enables hybrid web search using OpenAI web_search. Requires `OPENAI_API_KEY`.
- Use `--openai-model` to override the model (default: `gpt-5.2`).
- Use `--summary-max-chars` to cap abstract length sent to the model.
- Use `--reset-db` to clear `data/visited.jsonl` before a run if you need to re-ingest everything.
- When you add features or directories, update the corresponding README so agents can reorient quickly.
