# src/

Rust CLI that fetches GPU-related feeds, generates a daily digest, and updates the visited database. The entry point is `src/main.rs`.

## Key pieces

- CLI args: config path, visited DB path, output path, and per-source report limit.
- Feed parsing: uses `feed-rs` to parse RSS/Atom.
- Deduping: URLs are stored in `data/visited.jsonl` to avoid repeats.
