use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    env,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, Result};
use anyhow::Context as _;
use dotenvy::dotenv;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::process::Command;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::connect_async;

#[derive(Clone)]
struct Config {
    slack_app_token: String,
    slack_bot_token: String,
    slack_response_type: String,
    openai_api_key: String,
    openai_base_url: String,
    openai_chat_model: String,
    openai_embed_model: String,
    rag_embeddings_path: String,
    reports_dir: String,
    rag_top_k: usize,
    rag_max_context_chars: usize,
    rag_paper_map_max_chars: usize,
    rag_paper_reduce_max_chars: usize,
    rag_paper_map_concurrency: usize,
    rag_paper_summary_cache_dir: String,
    rag_paper_summary_cache_ttl_secs: u64,
    ask_max_response_chars: usize,
    rag_sync_service: String,
    rag_sync_use_sudo: bool,
}

impl Config {
    fn from_env() -> Result<Self> {
        Ok(Self {
            slack_app_token: parse_env_required("SLACK_APP_TOKEN")?,
            slack_bot_token: parse_env_required("SLACK_BOT_TOKEN")?,
            slack_response_type: parse_env_string("SLACK_RESPONSE_TYPE", "ephemeral"),
            openai_api_key: parse_env_required("OPENAI_API_KEY")?,
            openai_base_url: parse_env_string("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_chat_model: parse_env_string("OPENAI_CHAT_MODEL", "gpt-5.2"),
            openai_embed_model: parse_env_string("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            rag_embeddings_path: parse_env_string("RAG_EMBEDDINGS_PATH", "rag/embeddings.jsonl"),
            reports_dir: parse_env_string("REPORTS_DIR", "reports"),
            rag_top_k: parse_env_usize("RAG_TOP_K", 12),
            rag_max_context_chars: parse_env_usize("RAG_MAX_CONTEXT_CHARS", 8000),
            rag_paper_map_max_chars: parse_env_usize("RAG_PAPER_MAP_MAX_CHARS", 6000),
            rag_paper_reduce_max_chars: parse_env_usize("RAG_PAPER_REDUCE_MAX_CHARS", 12000),
            rag_paper_map_concurrency: parse_env_usize("RAG_PAPER_MAP_CONCURRENCY", 3),
            rag_paper_summary_cache_dir: parse_env_string(
                "RAG_PAPER_SUMMARY_CACHE_DIR",
                "rag/paper_summaries",
            ),
            rag_paper_summary_cache_ttl_secs: parse_env_usize(
                "RAG_PAPER_SUMMARY_CACHE_TTL_SECS",
                7 * 24 * 60 * 60,
            ) as u64,
            ask_max_response_chars: parse_env_usize("ASK_MAX_RESPONSE_CHARS", 1800),
            rag_sync_service: parse_env_string(
                "RAG_SYNC_SERVICE",
                "research-bot-rag-sync.service",
            ),
            rag_sync_use_sudo: parse_env_bool("RAG_SYNC_USE_SUDO", false),
        })
    }
}

fn parse_env_required(name: &str) -> Result<String> {
    env::var(name).with_context(|| format!("{name} is not set"))
}

fn parse_env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

#[derive(Debug, Deserialize)]
struct RagChunkRow {
    text: String,
    embedding: Vec<f32>,
    source: Option<String>,
    chunk_index: Option<usize>,
}

struct RagChunk {
    text: String,
    embedding: Vec<f32>,
    norm: f32,
    source: Option<String>,
    chunk_index: Option<usize>,
}

struct EmbeddingsIndex {
    chunks: Vec<RagChunk>,
}

impl EmbeddingsIndex {
    fn load(path: &str) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {path}"))?;
        let reader = BufReader::new(file);
        let mut chunks = Vec::new();
        for line in reader.lines() {
            let line = line.context("read embeddings line")?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: RagChunkRow = serde_json::from_str(line).context("parse embeddings row")?;
            if row.embedding.is_empty() {
                bail!("embedding row missing vector");
            }
            let norm = vector_norm(&row.embedding);
            if norm == 0.0 {
                bail!("embedding row has zero norm");
            }
            chunks.push(RagChunk {
                text: row.text,
                embedding: row.embedding,
                norm,
                source: row.source,
                chunk_index: row.chunk_index,
            });
        }
        if chunks.is_empty() {
            bail!("no embeddings loaded from {path}");
        }
        Ok(Self { chunks })
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<&RagChunk> {
        let query_norm = vector_norm(query);
        if query_norm == 0.0 {
            return Vec::new();
        }
        let mut scored: Vec<(f32, &RagChunk)> = self
            .chunks
            .iter()
            .filter(|chunk| chunk.embedding.len() == query.len())
            .map(|chunk| (cosine_similarity(query, query_norm, chunk), chunk))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(_, chunk)| chunk).collect()
    }
}

fn vector_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn cosine_similarity(query: &[f32], query_norm: f32, chunk: &RagChunk) -> f32 {
    let dot = query
        .iter()
        .zip(chunk.embedding.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();
    dot / (query_norm * chunk.norm)
}

struct TitleEntry {
    title: String,
    normalized: String,
    source: Option<String>,
    arxiv_id: Option<String>,
}

struct TitleIndex {
    entries: Vec<TitleEntry>,
}

impl TitleIndex {
    fn empty() -> Self {
        Self { entries: Vec::new() }
    }

    fn search(&self, query: &str) -> Vec<&TitleEntry> {
        let query_norm = normalize_title(query);
        if query_norm.is_empty() {
            return Vec::new();
        }
        self.entries
            .iter()
            .filter(|entry| {
                entry.normalized.contains(&query_norm)
                    || query_norm.contains(&entry.normalized)
            })
            .collect()
    }
}

struct PaperEntry {
    source: String,
    title: Option<String>,
    category: Option<String>,
    tags: Vec<String>,
    label: String,
    value: String,
}

struct PaperIndex {
    entries: Vec<PaperEntry>,
    value_to_source: HashMap<String, String>,
    category_counts: BTreeMap<String, usize>,
    tag_counts: BTreeMap<String, usize>,
}

const UNCATEGORIZED_LABEL: &str = "Uncategorized";

struct TagRule {
    name: &'static str,
    keywords: &'static [&'static str],
}

const TAG_RULES: &[TagRule] = &[
    TagRule {
        name: "compiler",
        keywords: &["compiler", "codegen", "code generation", "autotuning", "llvm"],
    },
    TagRule {
        name: "mlir",
        keywords: &["mlir"],
    },
    TagRule {
        name: "triton",
        keywords: &["triton"],
    },
];

#[derive(Debug, Serialize, Deserialize)]
struct PaperSummaryCache {
    source: String,
    question_hash: String,
    content_hash: String,
    model: String,
    map_max_chars: usize,
    reduce_max_chars: usize,
    created_at: u64,
    answer: String,
}

struct SlackState {
    config: Config,
    http_client: HttpClient,
    rag_index: EmbeddingsIndex,
    title_index: TitleIndex,
    paper_index: PaperIndex,
}

#[derive(Debug, Deserialize)]
struct SlackSocketResponse {
    ok: bool,
    url: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SlackEnvelope {
    envelope_id: String,
    #[serde(rename = "type")]
    envelope_type: String,
    payload: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct SlackSlashPayload {
    command: String,
    text: String,
    response_url: String,
    user_id: Option<String>,
    channel_id: Option<String>,
    trigger_id: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let config = Config::from_env()?;
    let rag_index = EmbeddingsIndex::load(&config.rag_embeddings_path)?;
    let title_index = match load_title_index(&config, &rag_index) {
        Ok(index) => index,
        Err(err) => {
            eprintln!("Failed to build title index: {err}");
            TitleIndex::empty()
        }
    };
    let report_categories = match load_report_categories(&config) {
        Ok(categories) => categories,
        Err(err) => {
            eprintln!("Failed to load report categories: {err}");
            HashMap::new()
        }
    };
    let paper_index = build_paper_index(&rag_index, &title_index, &report_categories);
    let http_client = HttpClient::builder()
        .user_agent("research-bot-slack/0.1")
        .build()
        .context("build http client")?;
    let state = Arc::new(SlackState {
        config,
        http_client,
        rag_index,
        title_index,
        paper_index,
    });

    loop {
        if let Err(err) = run_socket_loop(state.clone()).await {
            eprintln!("Socket loop error: {err}");
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }
}

async fn run_socket_loop(state: Arc<SlackState>) -> Result<()> {
    let socket_url = open_socket_url(&state.http_client, &state.config).await?;
    let (ws_stream, _) = connect_async(socket_url)
        .await
        .context("connect slack socket")?;
    let (mut ws_write, mut ws_read) = ws_stream.split();
    while let Some(message) = ws_read.next().await {
        let message = match message {
            Ok(value) => value,
            Err(err) => return Err(err).context("read slack socket message"),
        };
        let text = match message {
            Message::Text(text) => text,
            Message::Close(_) => break,
            _ => continue,
        };
        let envelope: SlackEnvelope = match serde_json::from_str(&text) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("Slack envelope parse error: {err}");
                continue;
            }
        };
        let mut ack_payload = None;
        if envelope.envelope_type == "interactive" {
            match handle_interactive_payload(state.clone(), &envelope.payload).await {
                Ok(payload) => ack_payload = payload,
                Err(err) => eprintln!("Slack interactive error: {err}"),
            }
        }
        let ack = if let Some(payload) = ack_payload {
            json!({ "envelope_id": envelope.envelope_id, "payload": payload }).to_string()
        } else {
            json!({ "envelope_id": envelope.envelope_id }).to_string()
        };
        ws_write
            .send(Message::Text(ack))
            .await
            .context("send slack ack")?;
        if envelope.envelope_type != "slash_commands" {
            continue;
        }
        let payload: SlackSlashPayload = match serde_json::from_value(envelope.payload) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("Slack payload parse error: {err}");
                continue;
            }
        };
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(err) = handle_slash_command(state, payload).await {
                eprintln!("Slack command error: {err}");
            }
        });
    }
    Ok(())
}

async fn open_socket_url(client: &HttpClient, config: &Config) -> Result<String> {
    let response = client
        .post("https://slack.com/api/apps.connections.open")
        .bearer_auth(&config.slack_app_token)
        .send()
        .await
        .context("request slack socket url")?;
    let payload = response.text().await.context("read slack socket response")?;
    let data: SlackSocketResponse =
        serde_json::from_str(&payload).context("parse slack socket response")?;
    if !data.ok {
        bail!(
            "slack apps.connections.open failed: {}",
            data.error.unwrap_or_else(|| "unknown error".to_string())
        );
    }
    data.url.ok_or_else(|| anyhow::anyhow!("slack socket url missing"))
}

async fn handle_slash_command(state: Arc<SlackState>, payload: SlackSlashPayload) -> Result<()> {
    let text = payload.text.trim();
    println!(
        "slack: command={} user={:?} channel={:?} text={}",
        payload.command, payload.user_id, payload.channel_id, text
    );
    match payload.command.as_str() {
        "/ask" => {
            if text.is_empty() {
                return post_slack_response(
                    &state.http_client,
                    &payload.response_url,
                    "Usage: /ask <question>",
                    &state.config.slack_response_type,
                )
                .await;
            }
            let response = answer_ask(&state, text).await?;
            post_slack_response(
                &state.http_client,
                &payload.response_url,
                &response,
                &state.config.slack_response_type,
            )
            .await?;
        }
        "/sync" => {
            let response = trigger_rag_sync(&state).await;
            post_slack_response(
                &state.http_client,
                &payload.response_url,
                &response,
                &state.config.slack_response_type,
            )
            .await?;
        }
        "/papers" => {
            let responses = list_available_papers(&state, text);
            let mut iterator = responses.into_iter();
            if let Some(first) = iterator.next() {
                post_slack_response(
                    &state.http_client,
                    &payload.response_url,
                    &first,
                    &state.config.slack_response_type,
                )
                .await?;
                for message in iterator {
                    post_slack_response_followup(
                        &state.http_client,
                        &payload.response_url,
                        &message,
                        &state.config.slack_response_type,
                    )
                    .await?;
                }
            }
        }
        "/ask_paper" => {
            let trigger_id = match payload.trigger_id.as_deref() {
                Some(value) if !value.trim().is_empty() => value,
                _ => {
                    return post_slack_response(
                        &state.http_client,
                        &payload.response_url,
                        "Missing trigger_id for modal. Please try again.",
                        &state.config.slack_response_type,
                    )
                    .await;
                }
            };
            if let Err(err) = open_ask_paper_modal(
                &state,
                trigger_id,
                payload.channel_id.as_deref(),
                payload.user_id.as_deref(),
                if text.is_empty() { None } else { Some(text) },
            )
            .await
            {
                let message = format!("Failed to open modal: {err}");
                post_slack_response(
                    &state.http_client,
                    &payload.response_url,
                    &message,
                    &state.config.slack_response_type,
                )
                .await?;
            }
        }
        _ => {
            let message = format!("Unknown command: {}", payload.command);
            post_slack_response(
                &state.http_client,
                &payload.response_url,
                &message,
                &state.config.slack_response_type,
            )
            .await?;
        }
    }
    Ok(())
}

async fn trigger_rag_sync(state: &SlackState) -> String {
    let service = state.config.rag_sync_service.trim();
    if service.is_empty() {
        return "RAG_SYNC_SERVICE is empty; cannot run sync.".to_string();
    }
    let mut command = if state.config.rag_sync_use_sudo {
        let mut command = Command::new("sudo");
        command.arg("systemctl");
        command
    } else {
        let mut command = Command::new("systemctl");
        command.arg("--user");
        command
    };
    command.arg("start").arg(service);

    match command.output().await {
        Ok(output) if output.status.success() => {
            format!("Sync triggered: {}", service)
        }
        Ok(output) => {
            let stdout = trim_output(&output.stdout);
            let stderr = trim_output(&output.stderr);
            let mut message = format!("Sync failed ({}).", output.status);
            if !stderr.is_empty() {
                message.push_str(&format!(" stderr: {}", stderr));
            }
            if !stdout.is_empty() {
                message.push_str(&format!(" stdout: {}", stdout));
            }
            message
        }
        Err(err) => format!("Sync failed: {}", err),
    }
}

fn trim_output(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim().to_string()
}

enum PapersQuery {
    All(Option<String>),
    Categories,
    CategoryFilter(String),
    Tags,
    TagFilter(String),
    TextFilter(String),
}

fn parse_papers_query(query: &str) -> PapersQuery {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return PapersQuery::TextFilter(String::new());
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower == "all" {
        return PapersQuery::All(None);
    }
    if lower.starts_with("all ") {
        let filter = trimmed.get(4..).unwrap_or("").trim();
        return PapersQuery::All(Some(filter.to_string()));
    }
    if lower == "categories" || lower == "category" {
        return PapersQuery::Categories;
    }
    if lower == "tags" || lower == "tag" {
        return PapersQuery::Tags;
    }
    if lower.starts_with("category:") {
        let filter = trimmed.get(9..).unwrap_or("").trim();
        if filter.is_empty() {
            return PapersQuery::Categories;
        }
        return PapersQuery::CategoryFilter(filter.to_string());
    }
    if lower.starts_with("tag:") {
        let filter = trimmed.get(4..).unwrap_or("").trim();
        if filter.is_empty() {
            return PapersQuery::Tags;
        }
        return PapersQuery::TagFilter(filter.to_string());
    }
    if lower.starts_with("cat:") {
        let filter = trimmed.get(4..).unwrap_or("").trim();
        if filter.is_empty() {
            return PapersQuery::Categories;
        }
        return PapersQuery::CategoryFilter(filter.to_string());
    }
    if lower.starts_with("category ") {
        let filter = trimmed.get(9..).unwrap_or("").trim();
        if filter.is_empty() {
            return PapersQuery::Categories;
        }
        return PapersQuery::CategoryFilter(filter.to_string());
    }
    if lower.starts_with("tag ") {
        let filter = trimmed.get(4..).unwrap_or("").trim();
        if filter.is_empty() {
            return PapersQuery::Tags;
        }
        return PapersQuery::TagFilter(filter.to_string());
    }
    if lower.starts_with("cat ") {
        let filter = trimmed.get(4..).unwrap_or("").trim();
        if filter.is_empty() {
            return PapersQuery::Categories;
        }
        return PapersQuery::CategoryFilter(filter.to_string());
    }
    PapersQuery::TextFilter(trimmed.to_string())
}

fn list_available_papers(state: &SlackState, query: &str) -> Vec<String> {
    match parse_papers_query(query) {
        PapersQuery::Categories => list_paper_categories(state),
        PapersQuery::CategoryFilter(category) => list_papers_by_category(state, &category),
        PapersQuery::Tags => list_paper_tags(state),
        PapersQuery::TagFilter(tag) => list_papers_by_tag(state, &tag),
        PapersQuery::All(filter) => list_papers_filtered(state, filter.as_deref(), None),
        PapersQuery::TextFilter(filter) => list_papers_filtered(state, Some(&filter), Some(30)),
    }
}

fn list_paper_categories(state: &SlackState) -> Vec<String> {
    if state.paper_index.category_counts.is_empty() {
        return vec!["No categories found in the reports yet.".to_string()];
    }
    let mut lines = Vec::new();
    for (category, count) in &state.paper_index.category_counts {
        lines.push(format!("- {} ({})", category, count));
    }
    let header = format!(
        "Available categories ({}). Use /papers category:<name> to list papers.",
        lines.len()
    );
    paginate_lines_with_header(&lines, &header, state.config.ask_max_response_chars)
}

fn list_paper_tags(state: &SlackState) -> Vec<String> {
    if state.paper_index.tag_counts.is_empty() {
        return vec!["No tags matched any papers.".to_string()];
    }
    let mut lines = Vec::new();
    for (tag, count) in &state.paper_index.tag_counts {
        lines.push(format!("- {} ({})", tag, count));
    }
    let header = format!(
        "Available tags ({}). Use /papers tag:<name> to list papers.",
        lines.len()
    );
    paginate_lines_with_header(&lines, &header, state.config.ask_max_response_chars)
}

fn list_papers_by_category(state: &SlackState, category_query: &str) -> Vec<String> {
    let query = category_query.trim();
    if query.is_empty() {
        return list_paper_categories(state);
    }
    let mut entries: Vec<&PaperEntry> = state
        .paper_index
        .entries
        .iter()
        .filter(|entry| category_matches(entry, query))
        .collect();
    entries.sort_by(|a, b| a.label.cmp(&b.label).then(a.source.cmp(&b.source)));

    let total = entries.len();
    if total == 0 {
        return vec![format!(
            "No papers matched category \"{}\". Use /papers categories to list categories.",
            query
        )];
    }
    let mut lines = Vec::with_capacity(total);
    for entry in entries {
        lines.push(paper_entry_line(entry));
    }
    let header = format!(
        "Papers in categories matching \"{}\" (total: {}).",
        query, total
    );
    paginate_lines_with_header(&lines, &header, state.config.ask_max_response_chars)
}

fn list_papers_by_tag(state: &SlackState, tag_query: &str) -> Vec<String> {
    let query = tag_query.trim();
    if query.is_empty() {
        return list_paper_tags(state);
    }
    let mut entries: Vec<&PaperEntry> = state
        .paper_index
        .entries
        .iter()
        .filter(|entry| tag_matches(entry, query))
        .collect();
    entries.sort_by(|a, b| a.label.cmp(&b.label).then(a.source.cmp(&b.source)));

    let total = entries.len();
    if total == 0 {
        return vec![format!(
            "No papers matched tag \"{}\". Use /papers tags to list tags.",
            query
        )];
    }
    let mut lines = Vec::with_capacity(total);
    for entry in entries {
        lines.push(paper_entry_line(entry));
    }
    let header = format!("Papers tagged \"{}\" (total: {}).", query, total);
    paginate_lines_with_header(&lines, &header, state.config.ask_max_response_chars)
}

fn list_papers_filtered(
    state: &SlackState,
    query: Option<&str>,
    limit: Option<usize>,
) -> Vec<String> {
    let query = query.unwrap_or("").trim();
    let mut entries: Vec<&PaperEntry> = state.paper_index.entries.iter().collect();
    if !query.is_empty() {
        entries.retain(|entry| paper_matches_query(entry, query));
    }
    entries.sort_by(|a, b| a.label.cmp(&b.label).then(a.source.cmp(&b.source)));

    let total = entries.len();
    if total == 0 {
        if query.is_empty() {
            return vec!["No papers found in the RAG index.".to_string()];
        }
        return vec![format!("No papers matched \"{}\".", query)];
    }

    let shown = limit.map(|limit| std::cmp::min(total, limit)).unwrap_or(total);
    let mut lines = Vec::with_capacity(shown);
    for entry in entries.into_iter().take(shown) {
        lines.push(paper_entry_line(entry));
    }

    let mut header = if query.is_empty() {
        format!("Available papers (total: {total}).")
    } else {
        format!("Matched papers for \"{}\" (total: {total}).", query)
    };
    if let Some(_) = limit {
        header.push_str(&format!(" Showing {shown}."));
        if shown < total {
            if query.is_empty() {
                header.push_str(" Use /papers all to show everything.");
            } else {
                header.push_str(&format!(" Use /papers all {} to show everything.", query));
            }
        }
    }
    header.push_str(" Use /papers categories or /papers tags to list categories.");

    paginate_lines_with_header(&lines, &header, state.config.ask_max_response_chars)
}

fn paper_entry_line(entry: &PaperEntry) -> String {
    if let Some(title) = &entry.title {
        format!("- {} ({})", title, entry.source)
    } else {
        format!("- {}", entry.source)
    }
}

fn paginate_lines_with_header(
    lines: &[String],
    header: &str,
    max_chars: usize,
) -> Vec<String> {
    if lines.is_empty() {
        return vec![truncate_text(header, max_chars)];
    }
    let page_overhead = 16usize;
    let max_body_chars = max_chars.saturating_sub(header.len() + page_overhead + 1);
    let mut pages: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_len = 0usize;
    for line in lines {
        let line_len = line.len();
        let next_len = if current.is_empty() {
            line_len
        } else {
            current_len + 1 + line_len
        };
        if !current.is_empty() && next_len > max_body_chars {
            pages.push(current);
            current = Vec::new();
            current_len = 0;
        }
        current_len = if current.is_empty() {
            line_len
        } else {
            current_len + 1 + line_len
        };
        current.push(line.clone());
    }
    if !current.is_empty() {
        pages.push(current);
    }

    let total_pages = pages.len();
    let mut messages = Vec::with_capacity(total_pages);
    for (idx, page) in pages.into_iter().enumerate() {
        let mut header_line = header.to_string();
        if total_pages > 1 {
            header_line.push_str(&format!(" Page {}/{}.", idx + 1, total_pages));
        }
        let message = format!("{}\n{}", header_line, page.join("\n"));
        messages.push(truncate_text(&message, max_chars));
    }
    messages
}

fn is_paper_source(source: &str) -> bool {
    source.starts_with("papers/") && source.to_ascii_lowercase().ends_with(".pdf")
}

async fn handle_interactive_payload(
    state: Arc<SlackState>,
    payload: &serde_json::Value,
) -> Result<Option<serde_json::Value>> {
    let payload_type = payload
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    match payload_type {
        "block_suggestion" => Ok(Some(handle_block_suggestion(state, payload))),
        "view_submission" => handle_view_submission(state, payload).await,
        _ => Ok(None),
    }
}

fn handle_block_suggestion(state: Arc<SlackState>, payload: &serde_json::Value) -> serde_json::Value {
    let action_id = payload
        .get("action_id")
        .and_then(|value| value.as_str())
        .or_else(|| {
            payload
                .get("actions")
                .and_then(|value| value.as_array())
                .and_then(|actions| actions.first())
                .and_then(|action| action.get("action_id"))
                .and_then(|value| value.as_str())
        })
        .unwrap_or("");
    if action_id != "paper_select" {
        return json!({ "options": [] });
    }
    let query = payload
        .get("value")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let options = paper_options_for_query(&state.paper_index, query);
    json!({ "options": options })
}

async fn handle_view_submission(
    state: Arc<SlackState>,
    payload: &serde_json::Value,
) -> Result<Option<serde_json::Value>> {
    let view = match payload.get("view") {
        Some(view) => view,
        None => return Ok(None),
    };
    let values = view
        .get("state")
        .and_then(|state| state.get("values"))
        .and_then(|values| values.as_object())
        .ok_or_else(|| anyhow::anyhow!("missing modal state values"))?;
    let paper_value = values
        .get("paper_block")
        .and_then(|block| block.get("paper_select"))
        .and_then(|value| value.get("selected_option"))
        .and_then(|value| value.get("value"))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());
    let question = values
        .get("question_block")
        .and_then(|block| block.get("question_input"))
        .and_then(|value| value.get("value"))
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_string());

    if paper_value.is_none() {
        return Ok(Some(json!({
            "response_action": "errors",
            "errors": { "paper_block": "Select a paper." }
        })));
    }
    if question.as_deref().unwrap_or("").is_empty() {
        return Ok(Some(json!({
            "response_action": "errors",
            "errors": { "question_block": "Enter a question." }
        })));
    }

    let metadata = view
        .get("private_metadata")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let metadata: ModalMetadata = serde_json::from_str(metadata)
        .context("parse modal private_metadata")?;

    let paper_value = paper_value.expect("paper value checked");
    let question = question.expect("question checked");
    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(err) =
            handle_modal_submission(state_clone, metadata, paper_value, question).await
        {
            eprintln!("Modal submission error: {err}");
        }
    });

    Ok(None)
}

#[derive(Deserialize)]
struct ModalMetadata {
    channel_id: Option<String>,
    user_id: Option<String>,
    response_type: Option<String>,
}

async fn handle_modal_submission(
    state: Arc<SlackState>,
    metadata: ModalMetadata,
    paper_value: String,
    question: String,
) -> Result<()> {
    let channel_id = metadata
        .channel_id
        .ok_or_else(|| anyhow::anyhow!("modal missing channel_id"))?;
    let response_type = metadata
        .response_type
        .unwrap_or_else(|| state.config.slack_response_type.clone());
    let user_id = metadata.user_id;
    let source = state
        .paper_index
        .value_to_source
        .get(&paper_value)
        .ok_or_else(|| anyhow::anyhow!("unknown paper selection"))?
        .to_string();
    let answer = answer_ask_paper_by_source(&state, &source, &question).await?;
    post_slack_chat(
        &state.http_client,
        &state.config.slack_bot_token,
        &channel_id,
        user_id.as_deref(),
        &response_type,
        &answer,
    )
    .await
}

async fn post_slack_response(
    client: &HttpClient,
    response_url: &str,
    text: &str,
    response_type: &str,
) -> Result<()> {
    let payload = json!({
        "response_type": response_type,
        "text": text,
    });
    let response = client
        .post(response_url)
        .json(&payload)
        .send()
        .await
        .context("send slack response")?;
    let status = response.status();
    let body = response.text().await.context("read slack response body")?;
    if !status.is_success() {
        bail!("slack response error: {} {}", status, body);
    }
    Ok(())
}

async fn post_slack_response_followup(
    client: &HttpClient,
    response_url: &str,
    text: &str,
    response_type: &str,
) -> Result<()> {
    let payload = json!({
        "response_type": response_type,
        "replace_original": false,
        "text": text,
    });
    let response = client
        .post(response_url)
        .json(&payload)
        .send()
        .await
        .context("send slack followup response")?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("read slack followup response body")?;
    if !status.is_success() {
        bail!("slack response error: {} {}", status, body);
    }
    Ok(())
}

#[derive(Deserialize)]
struct SlackApiResponse {
    ok: bool,
    error: Option<String>,
}

async fn slack_api_post(
    client: &HttpClient,
    token: &str,
    endpoint: &str,
    body: serde_json::Value,
) -> Result<serde_json::Value> {
    let url = format!("https://slack.com/api/{endpoint}");
    let response = client
        .post(url)
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
        .context("send slack api request")?;
    let status = response.status();
    let payload = response.text().await.context("read slack api response")?;
    if !status.is_success() {
        bail!("slack api error: {} {}", status, payload);
    }
    let parsed: serde_json::Value =
        serde_json::from_str(&payload).context("parse slack api response")?;
    let ok = serde_json::from_value::<SlackApiResponse>(parsed.clone())
        .context("parse slack api ok response")?;
    if !ok.ok {
        bail!(
            "slack api error: {}",
            ok.error.unwrap_or_else(|| "unknown error".to_string())
        );
    }
    Ok(parsed)
}

async fn post_slack_chat(
    client: &HttpClient,
    token: &str,
    channel_id: &str,
    user_id: Option<&str>,
    response_type: &str,
    text: &str,
) -> Result<()> {
    let response_type = response_type.trim();
    let is_direct_message = channel_id.starts_with('D');
    if response_type == "in_channel" || is_direct_message {
        let body = json!({ "channel": channel_id, "text": text });
        slack_api_post(client, token, "chat.postMessage", body).await?;
        return Ok(());
    }
    let user_id = user_id.ok_or_else(|| anyhow::anyhow!("missing user_id for ephemeral reply"))?;
    let body = json!({ "channel": channel_id, "user": user_id, "text": text });
    slack_api_post(client, token, "chat.postEphemeral", body).await?;
    Ok(())
}

async fn open_ask_paper_modal(
    state: &SlackState,
    trigger_id: &str,
    channel_id: Option<&str>,
    user_id: Option<&str>,
    initial_question: Option<&str>,
) -> Result<()> {
    let metadata = json!({
        "channel_id": channel_id,
        "user_id": user_id,
        "response_type": state.config.slack_response_type,
    })
    .to_string();
    let question_input = json!({
        "type": "plain_text_input",
        "action_id": "question_input",
        "multiline": true,
    });
    let question_input = if let Some(initial) = initial_question {
        let initial = initial.trim();
        if initial.is_empty() {
            question_input
        } else {
            let mut value = question_input;
            if let Some(obj) = value.as_object_mut() {
                obj.insert("initial_value".to_string(), json!(truncate_text(initial, 1500)));
            }
            value
        }
    } else {
        question_input
    };
    let view = json!({
        "type": "modal",
        "callback_id": "ask_paper_modal",
        "title": { "type": "plain_text", "text": "Ask paper" },
        "submit": { "type": "plain_text", "text": "Ask" },
        "close": { "type": "plain_text", "text": "Cancel" },
        "private_metadata": metadata,
        "blocks": [
            {
                "type": "input",
                "block_id": "paper_block",
                "label": { "type": "plain_text", "text": "Paper" },
                "element": {
                    "type": "external_select",
                    "action_id": "paper_select",
                    "placeholder": { "type": "plain_text", "text": "Search papers..." },
                    "min_query_length": 1
                }
            },
            {
                "type": "input",
                "block_id": "question_block",
                "label": { "type": "plain_text", "text": "Question" },
                "element": question_input
            }
        ]
    });
    let body = json!({
        "trigger_id": trigger_id,
        "view": view
    });
    slack_api_post(
        &state.http_client,
        &state.config.slack_bot_token,
        "views.open",
        body,
    )
    .await?;
    Ok(())
}

async fn answer_ask(state: &SlackState, question: &str) -> Result<String> {
    let query_embedding = openai_embed(&state.http_client, &state.config, question).await?;
    let matches = state
        .rag_index
        .search(&query_embedding, state.config.rag_top_k);
    if matches.is_empty() {
        return Ok(format_reply(
            question,
            None,
            "No relevant context found in the RAG index.",
            "Context: none.",
            state.config.ask_max_response_chars,
        ));
    }
    let sections = build_sections_from_chunks(&matches, state.config.rag_paper_map_max_chars)?;
    let notes = extract_question_notes(
        &state.http_client,
        &state.config,
        "retrieved chunks",
        question,
        &sections,
    )
    .await?;
    let notes: Vec<String> = notes
        .into_iter()
        .filter(|note| !note_is_empty(note))
        .filter(|note| !note_is_not_mentioned(note))
        .collect();
    let answer = if notes.is_empty() {
        "I don't know based on the retrieved context.".to_string()
    } else {
        answer_from_section_notes(
            &state.http_client,
            &state.config,
            "retrieved chunks",
            question,
            &notes,
        )
        .await?
    };
    let context_note = build_context_note(&matches);
    Ok(format_reply(
        question,
        None,
        &answer,
        &context_note,
        state.config.ask_max_response_chars,
    ))
}

async fn answer_ask_paper_by_source(
    state: &SlackState,
    source: &str,
    question: &str,
) -> Result<String> {
    let chunks: Vec<&RagChunk> = state
        .rag_index
        .chunks
        .iter()
        .filter(|chunk| chunk.source.as_deref() == Some(source))
        .collect();
    if chunks.is_empty() {
        bail!("no chunks found for selected paper");
    }
    build_paper_answer(state, source, chunks, question).await
}

async fn build_paper_answer(
    state: &SlackState,
    source: &str,
    mut chunks: Vec<&RagChunk>,
    question: &str,
) -> Result<String> {
    chunks.sort_by_key(|chunk| chunk.chunk_index.unwrap_or(usize::MAX));
    let sections = build_paper_sections(&chunks, state.config.rag_paper_map_max_chars)?;
    let content_hash = paper_content_hash(&chunks);
    let question_hash = hash_string(question.as_bytes());
    let cached_answer =
        load_paper_summary_cache(&state.config, source, &question_hash, &content_hash)?;
    let answer = if let Some(answer) = cached_answer {
        answer
    } else {
        let section_notes = extract_question_notes(
            &state.http_client,
            &state.config,
            source,
            question,
            &sections,
        )
        .await?;
        let mut notes: Vec<String> = section_notes
            .into_iter()
            .filter(|note| !note_is_empty(note))
            .collect();
        notes.retain(|note| !note_is_not_mentioned(note));
        let answer = if notes.is_empty() {
            "I don't know based on the paper.".to_string()
        } else {
            answer_from_section_notes(
                &state.http_client,
                &state.config,
                source,
                question,
                &notes,
            )
            .await?
        };
        if let Err(err) = write_paper_summary_cache(
            &state.config,
            source,
            &question_hash,
            &content_hash,
            &answer,
        ) {
            eprintln!("Failed to write paper summary cache: {err}");
        }
        answer
    };
    let title = title_for_source(&state.title_index, source);
    let context_note =
        build_paper_context_note(source, title.as_deref(), chunks.len(), sections.len());
    let question_label = paper_label_for_question(title.as_deref(), source);
    Ok(format_reply(
        question,
        question_label.as_deref(),
        &answer,
        &context_note,
        state.config.ask_max_response_chars,
    ))
}

async fn openai_embed(
    client: &HttpClient,
    config: &Config,
    text: &str,
) -> Result<Vec<f32>> {
    let url = format!("{}/embeddings", config.openai_base_url.trim_end_matches('/'));
    let body = json!({
        "model": config.openai_embed_model,
        "input": text,
    });
    let response = client
        .post(url)
        .bearer_auth(&config.openai_api_key)
        .json(&body)
        .send()
        .await
        .context("send embeddings request")?;
    let status = response.status();
    let payload = response.text().await.context("read embeddings response")?;
    if !status.is_success() {
        bail!("embeddings API error: {} {}", status, payload);
    }
    let data: EmbeddingsResponse =
        serde_json::from_str(&payload).context("parse embeddings response")?;
    let embedding = data
        .data
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("embeddings response missing data"))?
        .embedding;
    if embedding.is_empty() {
        bail!("embeddings response missing vector");
    }
    Ok(embedding)
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingRow>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingRow {
    embedding: Vec<f32>,
}

async fn openai_response_text(
    client: &HttpClient,
    config: &Config,
    prompt: &str,
) -> Result<String> {
    let url = format!("{}/responses", config.openai_base_url.trim_end_matches('/'));
    let body = json!({
        "model": config.openai_chat_model,
        "input": prompt,
        "temperature": 0.2,
        "max_output_tokens": 600
    });
    let response = client
        .post(url)
        .bearer_auth(&config.openai_api_key)
        .json(&body)
        .send()
        .await
        .context("send responses request")?;
    let status = response.status();
    let payload = response.text().await.context("read responses body")?;
    if !status.is_success() {
        bail!("responses API error: {} {}", status, payload);
    }
    extract_output_text(&payload)
}

fn extract_output_text(payload: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(payload).context("parse responses JSON")?;
    if let Some(text) = value.get("output_text").and_then(|t| t.as_str()) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            bail!("responses output_text is empty");
        }
        return Ok(trimmed.to_string());
    }
    let mut chunks = Vec::new();
    if let Some(output) = value.get("output").and_then(|o| o.as_array()) {
        for item in output {
            if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                for part in content {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            chunks.push(trimmed.to_string());
                        }
                    }
                }
            }
        }
    }
    if chunks.is_empty() {
        bail!("responses output missing text");
    }
    Ok(chunks.join("\n"))
}

fn build_sections_from_chunks(chunks: &[&RagChunk], max_chars: usize) -> Result<Vec<String>> {
    if max_chars == 0 {
        bail!("RAG_PAPER_MAP_MAX_CHARS must be greater than zero");
    }
    let mut sections = Vec::new();
    let mut current = String::new();
    for chunk in chunks {
        let text = chunk.text.trim();
        if text.is_empty() {
            continue;
        }
        if current.is_empty() && text.len() > max_chars {
            sections.extend(split_text(text, max_chars));
            continue;
        }
        if !current.is_empty() && current.len() + text.len() + 2 > max_chars {
            sections.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(text);
    }
    if !current.trim().is_empty() {
        sections.push(current.trim().to_string());
    }
    if sections.is_empty() {
        bail!("context sections empty");
    }
    Ok(sections)
}

fn build_context_note(matches: &[&RagChunk]) -> String {
    let mut sources = Vec::new();
    for chunk in matches.iter().take(3) {
        let source = chunk
            .source
            .as_deref()
            .unwrap_or("unknown-source");
        if !sources.contains(&source) {
            sources.push(source);
        }
    }
    let mut note = format!("Context: dataset (chunks: {})", matches.len());
    if !sources.is_empty() {
        note.push_str("; sources: ");
        note.push_str(&sources.join(", "));
    }
    note
}

fn normalize_title(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

fn title_for_source(title_index: &TitleIndex, source: &str) -> Option<String> {
    title_index
        .entries
        .iter()
        .find_map(|entry| {
            if entry.source.as_deref() == Some(source) {
                Some(entry.title.clone())
            } else {
                None
            }
        })
}

fn paper_label_for_question(title: Option<&str>, source: &str) -> Option<String> {
    if let Some(title) = title {
        let trimmed = title.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    Path::new(source)
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_string())
}

fn build_paper_context_note(
    source: &str,
    title: Option<&str>,
    chunk_count: usize,
    section_count: usize,
) -> String {
    match title {
        Some(title) => format!(
            "Context: paper {} (source: {}; chunks: {}; sections: {})",
            title, source, chunk_count, section_count
        ),
        None => format!(
            "Context: paper {} (chunks: {}, sections: {})",
            source, chunk_count, section_count
        ),
    }
}

fn build_paper_sections(chunks: &[&RagChunk], max_chars: usize) -> Result<Vec<String>> {
    if max_chars == 0 {
        bail!("RAG_PAPER_MAP_MAX_CHARS must be greater than zero");
    }
    let mut sections = Vec::new();
    let mut current = String::new();
    for chunk in chunks {
        let text = chunk.text.trim();
        if text.is_empty() {
            continue;
        }
        if current.is_empty() && text.len() > max_chars {
            sections.extend(split_text(text, max_chars));
            continue;
        }
        if !current.is_empty() && current.len() + text.len() + 2 > max_chars {
            sections.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(text);
    }
    if !current.trim().is_empty() {
        sections.push(current.trim().to_string());
    }
    if sections.is_empty() {
        bail!("paper sections empty");
    }
    Ok(sections)
}

fn split_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = std::cmp::min(start + max_chars, text.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            break;
        }
        parts.push(text[start..end].to_string());
        start = end;
    }
    parts
}

async fn extract_question_notes(
    client: &HttpClient,
    config: &Config,
    source: &str,
    question: &str,
    sections: &[String],
) -> Result<Vec<String>> {
    let total = sections.len();
    let concurrency = config.rag_paper_map_concurrency.max(1);
    if concurrency == 1 {
        let mut notes = Vec::new();
        for (idx, section) in sections.iter().enumerate() {
            let prompt = format!(
                "You are extracting evidence to answer a question about paper {source}. \
Question: {question}\n\
From section {part} of {total}, extract only information relevant to the question. \
If the section does not mention the answer, reply \"Not mentioned.\" \
Use short bullet points and avoid speculation.\n\n\
Section text:\n{section}\n\nNotes:",
                part = idx + 1,
                total = total,
                source = source,
                question = question,
                section = section
            );
            let note = openai_response_text(client, config, &prompt).await?;
            notes.push(note);
        }
        return Ok(notes);
    }

    let sem = Arc::new(Semaphore::new(concurrency));
    let mut set = JoinSet::new();
    for (idx, section) in sections.iter().enumerate() {
        let permit = sem
            .clone()
            .acquire_owned()
            .await
            .context("acquire summary permit")?;
        let client = client.clone();
        let config = config.clone();
        let source = source.to_string();
        let question = question.to_string();
        let section = section.clone();
        set.spawn(async move {
            let _permit = permit;
            let prompt = format!(
                "You are extracting evidence to answer a question about paper {source}. \
Question: {question}\n\
From section {part} of {total}, extract only information relevant to the question. \
If the section does not mention the answer, reply \"Not mentioned.\" \
Use short bullet points and avoid speculation.\n\n\
Section text:\n{section}\n\nNotes:",
                part = idx + 1,
                total = total,
                source = source,
                question = question,
                section = section
            );
            let note = openai_response_text(&client, &config, &prompt).await?;
            Ok::<_, anyhow::Error>((idx, note))
        });
    }

    let mut notes = Vec::with_capacity(total);
    while let Some(result) = set.join_next().await {
        let item = result.context("join summary task")??;
        notes.push(item);
    }
    notes.sort_by_key(|(idx, _)| *idx);
    Ok(notes.into_iter().map(|(_, note)| note).collect())
}

async fn answer_from_section_notes(
    client: &HttpClient,
    config: &Config,
    source: &str,
    question: &str,
    notes: &[String],
) -> Result<String> {
    if notes.is_empty() {
        bail!("paper notes missing");
    }
    let combined = truncate_text(&notes.join("\n\n"), config.rag_paper_reduce_max_chars);
    let prompt = format!(
        "You are answering a question about paper {source} using only the notes below. \
If the notes do not contain the answer, say you don't know.\n\n\
Question: {question}\n\nNotes:\n{combined}\n\nAnswer:",
        source = source,
        question = question,
        combined = combined
    );
    let answer = openai_response_text(client, config, &prompt).await?;
    Ok(answer)
}

fn note_is_empty(note: &str) -> bool {
    note.trim().is_empty()
}

fn note_is_not_mentioned(note: &str) -> bool {
    let trimmed = note
        .trim()
        .trim_end_matches('.')
        .to_ascii_lowercase();
    matches!(
        trimmed.as_str(),
        "not mentioned" | "no relevant info" | "no relevant information"
    )
}

fn paper_content_hash(chunks: &[&RagChunk]) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for chunk in chunks {
        hash = fnv1a_update(hash, chunk.text.as_bytes(), FNV_PRIME);
    }
    format!("{hash:016x}")
}

fn hash_string(bytes: &[u8]) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let hash = fnv1a_update(FNV_OFFSET, bytes, FNV_PRIME);
    format!("{hash:016x}")
}

fn fnv1a_update(mut hash: u64, bytes: &[u8], prime: u64) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(prime);
    }
    hash
}

fn sanitize_cache_key(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn cache_path_for_question(config: &Config, source: &str, question_hash: &str) -> Result<String> {
    let dir = config.rag_paper_summary_cache_dir.trim();
    if dir.is_empty() {
        bail!("RAG_PAPER_SUMMARY_CACHE_DIR is empty");
    }
    let filename = format!("{}_{}.json", sanitize_cache_key(source), question_hash);
    Ok(Path::new(dir)
        .join(filename)
        .to_string_lossy()
        .to_string())
}

fn now_unix_seconds() -> Result<u64> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time before unix epoch")?;
    Ok(now.as_secs())
}

fn load_paper_summary_cache(
    config: &Config,
    source: &str,
    question_hash: &str,
    content_hash: &str,
) -> Result<Option<String>> {
    if config.rag_paper_summary_cache_dir.trim().is_empty() {
        return Ok(None);
    }
    let path = cache_path_for_question(config, source, question_hash)?;
    let payload = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("read cache {path}")),
    };
    let cache: PaperSummaryCache =
        serde_json::from_str(&payload).context("parse paper summary cache")?;
    if cache.source != source
        || cache.question_hash != question_hash
        || cache.content_hash != content_hash
        || cache.model != config.openai_chat_model
        || cache.map_max_chars != config.rag_paper_map_max_chars
        || cache.reduce_max_chars != config.rag_paper_reduce_max_chars
    {
        return Ok(None);
    }
    let ttl = config.rag_paper_summary_cache_ttl_secs;
    if ttl > 0 {
        let now = now_unix_seconds()?;
        if cache.created_at == 0 || now.saturating_sub(cache.created_at) > ttl {
            return Ok(None);
        }
    }
    if cache.answer.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(cache.answer))
}

fn write_paper_summary_cache(
    config: &Config,
    source: &str,
    question_hash: &str,
    content_hash: &str,
    answer: &str,
) -> Result<()> {
    let dir = config.rag_paper_summary_cache_dir.trim();
    if dir.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir).with_context(|| format!("create {dir}"))?;
    let path = cache_path_for_question(config, source, question_hash)?;
    let cache = PaperSummaryCache {
        source: source.to_string(),
        question_hash: question_hash.to_string(),
        content_hash: content_hash.to_string(),
        model: config.openai_chat_model.clone(),
        map_max_chars: config.rag_paper_map_max_chars,
        reduce_max_chars: config.rag_paper_reduce_max_chars,
        created_at: now_unix_seconds()?,
        answer: answer.to_string(),
    };
    let payload = serde_json::to_string(&cache).context("serialize paper summary cache")?;
    fs::write(&path, payload).with_context(|| format!("write cache {path}"))?;
    Ok(())
}

fn list_report_files(dir: &str) -> Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read {dir}"))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.ends_with(".md") {
            continue;
        }
        if name.eq_ignore_ascii_case("readme.md") {
            continue;
        }
        files.push(path.to_string_lossy().to_string());
    }
    files.sort();
    Ok(files)
}

fn extract_markdown_link(line: &str) -> Option<(String, String)> {
    let start = line.find('[')?;
    let mid = line[start + 1..].find("](")? + start + 1;
    let end = line[mid + 2..].find(')')? + mid + 2;
    let title = line[start + 1..mid].trim();
    let url = line[mid + 2..end].trim();
    if title.is_empty() || url.is_empty() {
        return None;
    }
    Some((title.to_string(), url.to_string()))
}

fn extract_arxiv_id(url: &str) -> Option<String> {
    let url = url.trim();
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let mut parts = rest.splitn(2, '/');
    let host = parts.next()?.to_lowercase();
    let path = parts.next().unwrap_or("");
    if host != "arxiv.org" && host != "www.arxiv.org" && host != "export.arxiv.org" {
        return None;
    }
    let mut arxiv_id = if let Some(rest) = path.strip_prefix("abs/") {
        rest
    } else if let Some(rest) = path.strip_prefix("pdf/") {
        rest.trim_end_matches(".pdf")
    } else {
        return None;
    };
    if let Some(stripped) = arxiv_id.split('?').next() {
        arxiv_id = stripped;
    }
    if let Some(stripped) = arxiv_id.split('#').next() {
        arxiv_id = stripped;
    }
    let arxiv_id = arxiv_id.trim_matches('/');
    if arxiv_id.is_empty() {
        return None;
    }
    Some(arxiv_id.to_string())
}

fn safe_filename(arxiv_id: &str) -> String {
    format!("{}.pdf", arxiv_id.replace('/', "_"))
}

fn tags_for_entry(source: &str, title: Option<&str>) -> Vec<String> {
    let mut haystack = String::new();
    if let Some(title) = title {
        haystack.push_str(title);
        haystack.push(' ');
    }
    haystack.push_str(source);
    let haystack = haystack.to_lowercase();

    let mut tags = BTreeSet::new();
    for rule in TAG_RULES {
        if rule
            .keywords
            .iter()
            .any(|keyword| haystack.contains(&keyword.to_lowercase()))
        {
            tags.insert(rule.name.to_string());
        }
    }
    tags.into_iter().collect()
}

fn parse_report_heading(line: &str) -> Option<String> {
    let line = line.trim_start();
    let heading = line
        .strip_prefix("### ")
        .or_else(|| line.strip_prefix("## "))?;
    let heading = heading.trim();
    if heading.is_empty() {
        return None;
    }
    let heading = heading.split(" (").next().unwrap_or(heading).trim();
    if heading.is_empty() {
        return None;
    }
    Some(heading.to_string())
}

fn load_report_categories(config: &Config) -> Result<HashMap<String, String>> {
    if !Path::new(&config.reports_dir).is_dir() {
        return Ok(HashMap::new());
    }
    let report_files = list_report_files(&config.reports_dir)?;
    if report_files.is_empty() {
        return Ok(HashMap::new());
    }

    let mut categories = HashMap::new();
    for path in report_files.iter().rev() {
        let content = fs::read_to_string(path).with_context(|| format!("read report {path}"))?;
        let mut current_category: Option<String> = None;
        for line in content.lines() {
            if let Some(category) = parse_report_heading(line) {
                current_category = Some(category);
                continue;
            }
            let Some((_title, url)) = extract_markdown_link(line) else {
                continue;
            };
            let Some(arxiv_id) = extract_arxiv_id(&url) else {
                continue;
            };
            let Some(category) = current_category.as_ref() else {
                continue;
            };
            let source = format!("papers/{}", safe_filename(&arxiv_id));
            categories.entry(source).or_insert_with(|| category.clone());
        }
    }

    Ok(categories)
}

fn load_title_index(config: &Config, rag_index: &EmbeddingsIndex) -> Result<TitleIndex> {
    if !Path::new(&config.reports_dir).is_dir() {
        println!("reports dir not found: {}", config.reports_dir);
        return Ok(TitleIndex::empty());
    }
    let report_files = list_report_files(&config.reports_dir)?;
    if report_files.is_empty() {
        return Ok(TitleIndex::empty());
    }
    let mut available_sources = BTreeSet::new();
    for chunk in &rag_index.chunks {
        if let Some(source) = chunk.source.as_ref() {
            available_sources.insert(source.clone());
        }
    }

    let mut entries = Vec::new();
    let mut seen = BTreeSet::new();
    for path in report_files {
        let content = fs::read_to_string(&path).with_context(|| format!("read report {path}"))?;
        for line in content.lines() {
            let Some((title, url)) = extract_markdown_link(line) else {
                continue;
            };
            let Some(arxiv_id) = extract_arxiv_id(&url) else {
                continue;
            };
            let source = format!("papers/{}", safe_filename(&arxiv_id));
            let source_opt = if available_sources.contains(&source) {
                Some(source)
            } else {
                None
            };
            let normalized = normalize_title(&title);
            if normalized.is_empty() {
                continue;
            }
            let key = format!(
                "{}|{}",
                normalized,
                source_opt.as_deref().unwrap_or("missing")
            );
            if !seen.insert(key) {
                continue;
            }
            entries.push(TitleEntry {
                title,
                normalized,
                source: source_opt,
                arxiv_id: Some(arxiv_id),
            });
        }
    }
    Ok(TitleIndex { entries })
}

fn build_paper_index(
    rag_index: &EmbeddingsIndex,
    title_index: &TitleIndex,
    report_categories: &HashMap<String, String>,
) -> PaperIndex {
    let mut sources = BTreeSet::new();
    for chunk in &rag_index.chunks {
        if let Some(source) = chunk.source.as_deref() {
            if is_paper_source(source) {
                sources.insert(source.to_string());
            }
        }
    }

    let mut entries = Vec::new();
    let mut value_to_source = HashMap::new();
    let mut category_counts = BTreeMap::new();
    let mut tag_counts = BTreeMap::new();
    let mut uncategorized = 0usize;
    for source in sources {
        let title = title_for_source(title_index, &source);
        let category = report_categories.get(&source).cloned();
        let tags = tags_for_entry(&source, title.as_deref());
        let label = paper_label(&source, title.as_deref());
        let value = paper_value_for_source(&source);
        value_to_source.insert(value.clone(), source.clone());
        if let Some(category) = &category {
            *category_counts.entry(category.clone()).or_insert(0) += 1;
        } else {
            uncategorized += 1;
        }
        for tag in &tags {
            *tag_counts.entry(tag.clone()).or_insert(0) += 1;
        }
        entries.push(PaperEntry {
            source,
            title,
            category,
            tags,
            label,
            value,
        });
    }
    if uncategorized > 0 {
        category_counts.insert(UNCATEGORIZED_LABEL.to_string(), uncategorized);
    }
    PaperIndex {
        entries,
        value_to_source,
        category_counts,
        tag_counts,
    }
}

fn paper_label(source: &str, title: Option<&str>) -> String {
    if let Some(title) = title {
        let name = Path::new(source)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(source);
        let label = format!("{title} ({name})");
        return truncate_option_text(&label, 75);
    }
    let name = Path::new(source)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(source);
    truncate_option_text(name, 75)
}

fn paper_value_for_source(source: &str) -> String {
    let hash = hash_string(source.as_bytes());
    format!("paper:{hash}")
}

fn paper_options_for_query(index: &PaperIndex, query: &str) -> Vec<serde_json::Value> {
    let query = query.trim();
    let mut entries: Vec<&PaperEntry> = index.entries.iter().collect();
    if !query.is_empty() {
        entries.retain(|entry| paper_matches_query(entry, query));
    }
    entries.sort_by(|a, b| a.label.cmp(&b.label).then(a.source.cmp(&b.source)));
    entries
        .into_iter()
        .take(50)
        .map(|entry| {
            json!({
                "text": { "type": "plain_text", "text": entry.label },
                "value": entry.value.clone(),
            })
        })
        .collect()
}

fn category_matches(entry: &PaperEntry, query: &str) -> bool {
    let query_lower = query.to_lowercase();
    if query_lower.is_empty() {
        return false;
    }
    if let Some(category) = &entry.category {
        return category.to_lowercase().contains(&query_lower);
    }
    query_lower == UNCATEGORIZED_LABEL.to_lowercase()
}

fn tag_matches(entry: &PaperEntry, query: &str) -> bool {
    let query_lower = query.to_lowercase();
    if query_lower.is_empty() {
        return false;
    }
    entry
        .tags
        .iter()
        .any(|tag| tag.to_lowercase().contains(&query_lower))
}

fn paper_matches_query(entry: &PaperEntry, query: &str) -> bool {
    let query_lower = query.to_lowercase();
    if entry.source.to_lowercase().contains(&query_lower) {
        return true;
    }
    if let Some(category) = &entry.category {
        if category.to_lowercase().contains(&query_lower) {
            return true;
        }
    } else if query_lower == UNCATEGORIZED_LABEL.to_lowercase() {
        return true;
    }
    if entry
        .tags
        .iter()
        .any(|tag| tag.to_lowercase().contains(&query_lower))
    {
        return true;
    }
    let query_norm = normalize_title(query);
    if query_norm.is_empty() {
        return false;
    }
    if let Some(title) = &entry.title {
        let title_norm = normalize_title(title);
        return title_norm.contains(&query_norm) || query_norm.contains(&title_norm);
    }
    false
}

fn format_reply(
    question: &str,
    paper_label: Option<&str>,
    answer: &str,
    context_note: &str,
    max: usize,
) -> String {
    let question = truncate_text(&normalize_slack_text(question.trim()), 400);
    let paper = paper_label
        .and_then(|label| {
            let trimmed = label.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(truncated_label(&normalize_slack_text(trimmed)))
            }
        })
        .unwrap_or_else(|| "n/a".to_string());
    let mut context = context_note.trim();
    if let Some(stripped) = context.strip_prefix("Context:") {
        context = stripped.trim();
    }
    if context.is_empty() {
        context = "none";
    }
    let context = truncate_text(&normalize_slack_text(context), 600);

    let question = if question.is_empty() {
        "n/a".to_string()
    } else {
        question
    };
    let answer = normalize_slack_text(answer.trim());
    let header = format!("Paper: {paper}\nQuestion: {question}\nAnswer:\n");
    let footer = format!("\nContext: {context}");
    if header.len() + footer.len() >= max {
        let combined = format!(
            "Paper: {paper}\nQuestion: {question}\nAnswer:\n\nContext: {context}"
        );
        return truncate_text(&combined, max);
    }
    let max_answer = max.saturating_sub(header.len() + footer.len());
    let answer = truncate_text(&answer, max_answer);
    format!("{header}{answer}{footer}")
}

fn truncated_label(label: &str) -> String {
    truncate_text(label, 200)
}

fn truncate_option_text(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    if max <= 3 {
        return "...".to_string();
    }
    truncate_text(text, max - 3)
}

fn normalize_slack_text(text: &str) -> String {
    text.replace("**", "*")
}

fn truncate_text(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = text[..end].to_string();
    out.push_str("...");
    out
}
