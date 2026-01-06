# src/

Rust CLI that fetches GPU-related feeds, generates a daily digest, and updates the visited database. The entry point is `src/main.rs`. A Discord RAG bot lives at `src/bin/discord_bot.rs`. A Slack Socket Mode bot lives at `src/bin/slack_bot.rs`.

## Key pieces

- CLI args: config path, visited DB path, output path, per-source report limit, `--max-age-days` window, `--reset-db` to clear the visited file, `--summarize` for OpenAI summaries (requires `OPENAI_API_KEY`), `--discover` for OpenAI web_search, and `--discovery-config` to point at a different discovery file. Local `.env` is auto-loaded.
- Feed parsing: uses `feed-rs` to parse RSS/Atom.
- Deduping: URLs are stored in `data/visited.jsonl` to avoid repeats.
- Reports include authors and affiliations when available from the feed.
- Discord RAG bot logs each `/ask` and replies with section headings for Paper, Question, Answer, and Context; `/ask_paper` includes the paper title when available.
- Discord RAG bot also supports `/ask_paper` to answer using full-paper map-reduce summaries.
- `/ask_paper` title matching is derived from report markdown under `REPORTS_DIR`.
- `/ask_paper` caches per-paper summaries under `RAG_PAPER_SUMMARY_CACHE_DIR` for faster repeat queries.
- `RAG_PAPER_SUMMARY_CACHE_TTL_SECS` controls cache expiry (0 disables expiry).
- `/ask_paper` paper option supports Discord autocomplete suggestions from titles and sources.
- Slack bot runs in Socket Mode and responds to `/ask`, `/ask_paper`, `/papers`, and `/sync` slash commands with plain text labels for Paper/Question/Answer/Context.
