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

## Discord RAG bot

This repo also includes a Discord bot that answers `/ask` using full-text papers.

### Setup

1) Put PDFs under `papers/`.
2) Build embeddings (PDFs + reports):

```bash
python scripts/download_papers.py --reports-dir reports --out-dir papers
python scripts/rag_ingest.py --input-dir papers --reports-dir reports
```

Omit `--reports-dir` if you only want PDFs.

3) Run the bot:

```bash
cargo run --bin discord_bot
```

The bot logs each `/ask` request to stdout and replies with section headings for Paper, Question, Answer, and Context. `/ask_paper` includes the paper title when available.
`/ask_paper` title matching is built from the report markdown files in `REPORTS_DIR`.
Cached `/ask_paper` outputs are stored under `RAG_PAPER_SUMMARY_CACHE_DIR` to speed up repeat queries.
Use `RAG_PAPER_SUMMARY_CACHE_TTL_SECS` to expire cached outputs (0 disables expiry).

### Commands

- `/ask question:...` answers using top-K retrieved chunks.
- `/ask_paper paper:... question:...` answers from the full paper via map-reduce. `paper` can be a filename, substring of the source path (for example `2512.04226v1.pdf`), or a report title (for example `tritonBLAS`).
- The `paper` option supports Discord autocomplete with title and source suggestions.

### Required env

```
DISCORD_TOKEN=your-bot-token
GUILD_ID=your-guild-id
APPLICATION_ID=your-application-id
OPENAI_API_KEY=your-openai-key
```

Optional:

```
OPENAI_CHAT_MODEL=gpt-5.2
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_BASE_URL=https://api.openai.com/v1
RAG_EMBEDDINGS_PATH=rag/embeddings.jsonl
REPORTS_DIR=reports
RAG_TOP_K=12
RAG_MAX_CONTEXT_CHARS=8000
RAG_PAPER_MAP_MAX_CHARS=6000
RAG_PAPER_REDUCE_MAX_CHARS=12000
RAG_PAPER_MAP_CONCURRENCY=3
RAG_PAPER_SUMMARY_CACHE_DIR=rag/paper_summaries
RAG_PAPER_SUMMARY_CACHE_TTL_SECS=604800
ASK_MAX_RESPONSE_CHARS=1800
```

You can place these in `.env` for local runs.

## Slack RAG bot (Socket Mode)

This repo also includes a Slack bot that answers `/ask` and `/ask_paper` via Socket Mode (no public HTTPS endpoint required).

### Setup

1) Create a Slack app, enable **Socket Mode**, and generate an app-level token with the `connections:write` scope.
2) Add a bot user and install the app to your workspace.
3) Enable **Interactivity** and set a placeholder Request URL (Socket Mode delivers the events).
4) Add four slash commands: `/ask`, `/ask_paper`, `/papers`, and `/sync`. The request URL is unused in Socket Mode.
5) Ensure the app has `commands`, `chat:write`, and `views:write` scopes.

### Run

```bash
cargo run --bin slack_bot
```

### Commands

- `/ask <question>` answers using top-K retrieved chunks with question-aware reduction.
- `/ask_paper` opens a modal with paper autocomplete and a question field, then answers via question-aware map-reduce over the full paper.
- `/papers [filter]` lists available papers (up to 30). Use `/papers all` to show everything, `/papers all <filter>` for full filtered lists, `/papers categories` for report categories, `/papers category:<name>` to list by category, `/papers tags` for keyword tags, or `/papers tag:<name>` to list by tag.
- `/sync` triggers the user-level RAG sync service to pull the latest artifacts and restart the bot.
Slack replies use plain text labels (`Paper`, `Question`, `Answer`, `Context`) for reliable formatting.

### Required env

```
SLACK_APP_TOKEN=your-xapp-token
SLACK_BOT_TOKEN=your-xoxb-token
OPENAI_API_KEY=your-openai-key
```

Optional:

```
SLACK_RESPONSE_TYPE=ephemeral
OPENAI_CHAT_MODEL=gpt-5.2
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_BASE_URL=https://api.openai.com/v1
RAG_EMBEDDINGS_PATH=rag/embeddings.jsonl
REPORTS_DIR=reports
RAG_TOP_K=12
RAG_MAX_CONTEXT_CHARS=8000
RAG_PAPER_MAP_MAX_CHARS=6000
RAG_PAPER_REDUCE_MAX_CHARS=12000
RAG_PAPER_MAP_CONCURRENCY=3
RAG_PAPER_SUMMARY_CACHE_DIR=rag/paper_summaries
RAG_PAPER_SUMMARY_CACHE_TTL_SECS=604800
ASK_MAX_RESPONSE_CHARS=1800
RAG_SYNC_SERVICE=research-bot-rag-sync.service
RAG_SYNC_USE_SUDO=false
```

You can place these in `.env` for local runs.

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

## RAG artifacts workflow (server sync)

The bot reads local files from `papers/` and `rag/`, so it cannot use GitHub storage directly. To keep the repo clean, use the artifacts workflow instead of committing PDFs/embeddings.

`rag_artifacts.yml` runs daily (and on manual dispatch) and uploads an artifact named `rag-artifacts` containing:

- `papers/` (downloaded PDFs)
- `rag/chunks.jsonl`
- `rag/embeddings.jsonl`

Server sync example (requires `gh` CLI + a token with `actions:read` and `repo` scope for private repos):

```bash
gh run list -w "Build RAG Artifacts" -L 1 --json databaseId -q '.[0].databaseId'
gh run download <run-id> -n rag-artifacts -D /tmp/rag-artifacts
rsync -a /tmp/rag-artifacts/ /path/to/research-bot/
```

Run the sync before starting the Slack/Discord bot, and keep the repo updated with `git pull` for the latest reports.

For automatic daily sync on your server, see the systemd units in `scripts/systemd/` (timer + sync service).

## Layout

- `src/` Rust CLI implementation
- `data/` source configuration + visited database + discovery config
- `papers/` full-text PDFs for RAG ingestion
- `rag/` chunk + embedding outputs
- `scripts/` ingestion helpers
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
- Report items include authors and affiliations when available from the feed.
- `--max-age-days` filters out items older than the window; entries without dates are skipped.
- `--summarize` calls OpenAI to produce 2-3 bullet summaries per item. Requires `OPENAI_API_KEY`.
- `--discover` enables hybrid web search using OpenAI web_search. Requires `OPENAI_API_KEY`.
- Use `--openai-model` to override the model (default: `gpt-5.2`).
- Use `--summary-max-chars` to cap abstract length sent to the model.
- Use `--reset-db` to clear `data/visited.jsonl` before a run if you need to re-ingest everything.
- When you add features or directories, update the corresponding README so agents can reorient quickly.
