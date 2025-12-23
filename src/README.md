# src/

Rust CLI that fetches GPU-related feeds, generates a daily digest, and updates the visited database. The entry point is `src/main.rs`. A Discord RAG bot lives at `src/bin/discord_bot.rs`.

## Key pieces

- CLI args: config path, visited DB path, output path, per-source report limit, `--max-age-days` window, `--reset-db` to clear the visited file, `--summarize` for OpenAI summaries (requires `OPENAI_API_KEY`), `--discover` for OpenAI web_search, and `--discovery-config` to point at a different discovery file. Local `.env` is auto-loaded.
- Feed parsing: uses `feed-rs` to parse RSS/Atom.
- Deduping: URLs are stored in `data/visited.jsonl` to avoid repeats.
- Discord RAG bot logs each `/ask` and annotates replies with a `Context:` note (dataset sources or none).
- Discord RAG bot also supports `/ask_paper` to answer using full-paper map-reduce summaries.
- `/ask_paper` title matching is derived from report markdown under `REPORTS_DIR`.
- `/ask_paper` caches per-paper summaries under `RAG_PAPER_SUMMARY_CACHE_DIR` for faster repeat queries.
- `RAG_PAPER_SUMMARY_CACHE_TTL_SECS` controls cache expiry (0 disables expiry).
