# GPU Research Bot

A small Rust CLI that pulls GPU-related feeds, writes a daily Markdown digest, and tracks what has already been seen so runs stay deduplicated. The repo is designed to be driven entirely by agents ("code is lava"), so every subdirectory includes its own README.

## Quick start

1) Review and edit `data/sources.yml`.
2) Run the bot:

```bash
cargo run -- --config data/sources.yml --db data/visited.jsonl
```

3) Read the report in `reports/YYYY-MM-DD.md`.

## GitHub Actions

A scheduled workflow runs daily and commits new reports plus the visited DB back to the repo. Make sure the repo allows workflows to write:

- Settings -> Actions -> General -> Workflow permissions -> Read and write

The workflow is at `.github/workflows/daily.yml`.

## Layout

- `src/` Rust CLI implementation
- `data/` source configuration + visited database
- `reports/` daily digests (generated)
- `.github/workflows/` scheduled GitHub Actions run

## Config format

`data/sources.yml` lists feed URLs (RSS/Atom). Example:

```yaml
sources:
  - name: "arXiv: GPU OR CUDA"
    url: "http://export.arxiv.org/api/query?search_query=all:GPU+OR+all:CUDA&start=0&max_results=50"
```

## Notes

- The report shows up to `--max-per-source` items per source, but the visited DB tracks all new items so they do not repeat.
- When you add features or directories, update the corresponding README so agents can reorient quickly.
