#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is required" >&2
    exit 1
  fi
}

require_env REPO_DIR
require_env RAG_SYNC_TMP_DIR
require_env RAG_SYNC_SERVICE

if [[ -z "${GH_TOKEN:-}" && -z "${GITHUB_TOKEN:-}" ]]; then
  echo "GH_TOKEN or GITHUB_TOKEN is required for gh auth" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required" >&2
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl is required" >&2
  exit 1
fi

WORKFLOW_NAME="Build RAG Artifacts"
ARTIFACT_NAME="rag-artifacts"
SYSTEMCTL_ARGS=()

if [[ "${RAG_SYNC_SYSTEMCTL_USER:-}" =~ ^(1|true|yes|on)$ ]]; then
  SYSTEMCTL_ARGS+=(--user)
fi

cd "$REPO_DIR"
git pull --ff-only

run_id=$(gh run list -w "$WORKFLOW_NAME" --status success -L 1 --json databaseId -q '.[0].databaseId')
if [[ -z "$run_id" || "$run_id" == "null" ]]; then
  echo "No successful runs found for workflow: ${WORKFLOW_NAME}" >&2
  exit 1
fi

rm -rf "$RAG_SYNC_TMP_DIR"
mkdir -p "$RAG_SYNC_TMP_DIR"

gh run download "$run_id" -n "$ARTIFACT_NAME" -D "$RAG_SYNC_TMP_DIR"

if [[ ! -d "$RAG_SYNC_TMP_DIR/papers" ]]; then
  echo "Artifact missing papers/" >&2
  exit 1
fi
if [[ ! -f "$RAG_SYNC_TMP_DIR/rag/chunks.jsonl" ]]; then
  echo "Artifact missing rag/chunks.jsonl" >&2
  exit 1
fi
if [[ ! -f "$RAG_SYNC_TMP_DIR/rag/embeddings.jsonl" ]]; then
  echo "Artifact missing rag/embeddings.jsonl" >&2
  exit 1
fi

mkdir -p "$REPO_DIR/papers" "$REPO_DIR/rag"
rsync -a --delete "$RAG_SYNC_TMP_DIR/papers/" "$REPO_DIR/papers/"
cp "$RAG_SYNC_TMP_DIR/rag/chunks.jsonl" "$REPO_DIR/rag/chunks.jsonl"
cp "$RAG_SYNC_TMP_DIR/rag/embeddings.jsonl" "$REPO_DIR/rag/embeddings.jsonl"

systemctl "${SYSTEMCTL_ARGS[@]}" restart "$RAG_SYNC_SERVICE"
