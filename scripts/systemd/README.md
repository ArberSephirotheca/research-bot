# scripts/systemd/

User-level systemd unit files for running the Slack bot and syncing RAG artifacts daily.

## Files

- `research-bot-slack.service`: user service for `cargo run --bin slack_bot`.
- `research-bot-rag-sync.service`: user oneshot service that runs `scripts/sync_rag_artifacts.sh`.
- `research-bot-rag-sync.timer`: user timer that triggers the sync service.

## Environment file

Create `~/research-bot/rag-sync.env` with:

```
REPO_DIR=/home/your-user/research-bot
RAG_SYNC_TMP_DIR=/home/your-user/.cache/research-bot/rag-artifacts
RAG_SYNC_SERVICE=research-bot-slack.service
RAG_SYNC_SYSTEMCTL_USER=true
GH_TOKEN=your-gh-token
```

`GH_TOKEN` (or `GITHUB_TOKEN`) must have `actions:read` and `repo` access for private repos.
Set `RAG_SYNC_USE_SUDO=false` in `~/.env` so the Slack `/sync` command uses `systemctl --user`.

## Install

```
mkdir -p ~/.config/systemd/user
install -m 0644 scripts/systemd/research-bot-slack.service ~/.config/systemd/user/
install -m 0644 scripts/systemd/research-bot-rag-sync.service ~/.config/systemd/user/
install -m 0644 scripts/systemd/research-bot-rag-sync.timer ~/.config/systemd/user/
install -m 0755 scripts/sync_rag_artifacts.sh ~/research-bot/scripts/sync_rag_artifacts.sh
systemctl --user daemon-reload
systemctl --user enable --now research-bot-slack.service
systemctl --user enable --now research-bot-rag-sync.timer
```

If you want the timer to run while logged out, enable linger:

```
loginctl enable-linger your-user
```

Adjust `WorkingDirectory`, `EnvironmentFile`, and `ExecStart` paths to match your server layout.
Update the timer `OnCalendar` to run after the `Build RAG Artifacts` workflow finishes.
