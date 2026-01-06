# scripts/systemd/

Systemd unit files for running the Slack bot and syncing RAG artifacts daily.

## Files

- `research-bot-slack.service`: example service for `cargo run --bin slack_bot`.
- `research-bot-rag-sync.service`: oneshot service that runs `scripts/sync_rag_artifacts.sh`.
- `research-bot-rag-sync.timer`: daily timer that triggers the sync service.

## Environment file

Create `/etc/research-bot/rag-sync.env` with:

```
REPO_DIR=/opt/research-bot
RAG_SYNC_TMP_DIR=/var/lib/research-bot/rag-artifacts
RAG_SYNC_SERVICE=research-bot-slack.service
GH_TOKEN=your-gh-token
```

`GH_TOKEN` (or `GITHUB_TOKEN`) must have `actions:read` and `repo` access for private repos.

## Install

```
sudo install -d /etc/research-bot
sudo install -m 0644 scripts/systemd/research-bot-slack.service /etc/systemd/system/
sudo install -m 0644 scripts/systemd/research-bot-rag-sync.service /etc/systemd/system/
sudo install -m 0644 scripts/systemd/research-bot-rag-sync.timer /etc/systemd/system/
sudo install -m 0755 scripts/sync_rag_artifacts.sh /opt/research-bot/scripts/sync_rag_artifacts.sh
sudo systemctl daemon-reload
sudo systemctl enable --now research-bot-slack.service
sudo systemctl enable --now research-bot-rag-sync.timer
```

Adjust `WorkingDirectory`, `EnvironmentFile`, and `ExecStart` paths to match your server layout.
Update the timer `OnCalendar` to run after the `Build RAG Artifacts` workflow finishes.
