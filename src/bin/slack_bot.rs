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

#[derive(Debug, Serialize, Deserialize)]
struct PaperSummaryCache {
    source: String,
    content_hash: String,
    model: String,
    map_max_chars: usize,
    reduce_max_chars: usize,
    created_at: u64,
    summary: String,
}

struct SlackState {
    config: Config,
    http_client: HttpClient,
    rag_index: EmbeddingsIndex,
    title_index: TitleIndex,
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
    let http_client = HttpClient::builder()
        .user_agent("research-bot-slack/0.1")
        .build()
        .context("build http client")?;
    let state = Arc::new(SlackState {
        config,
        http_client,
        rag_index,
        title_index,
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
        let ack = json!({ "envelope_id": envelope.envelope_id }).to_string();
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
        "/ask_paper" => {
            let (paper, question) = parse_paper_question(text)?;
            let response = answer_ask_paper(&state, &paper, &question).await?;
            post_slack_response(
                &state.http_client,
                &payload.response_url,
                &response,
                &state.config.slack_response_type,
            )
            .await?;
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

fn parse_paper_question(text: &str) -> Result<(String, String)> {
    let parts: Vec<&str> = text.splitn(2, '|').collect();
    if parts.len() != 2 {
        bail!("Usage: /ask_paper <paper> | <question>");
    }
    let paper = parts[0].trim();
    let question = parts[1].trim();
    if paper.is_empty() || question.is_empty() {
        bail!("Usage: /ask_paper <paper> | <question>");
    }
    Ok((paper.to_string(), question.to_string()))
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
    let context = build_context(&matches, state.config.rag_max_context_chars)?;
    let prompt = format!(
        "You are a GPU research assistant. Use only the provided context. \
If the answer is not in the context, say you don't know.\n\n\
Context:\n{context}\n\nQuestion: {question}\nAnswer:",
        context = context,
        question = question
    );
    let answer = openai_response_text(&state.http_client, &state.config, &prompt).await?;
    let context_note = build_context_note(&matches);
    Ok(format_reply(
        question,
        None,
        &answer,
        &context_note,
        state.config.ask_max_response_chars,
    ))
}

async fn answer_ask_paper(
    state: &SlackState,
    paper_query: &str,
    question: &str,
) -> Result<String> {
    let mut by_source: BTreeMap<String, Vec<&RagChunk>> = BTreeMap::new();
    for chunk in &state.rag_index.chunks {
        if let Some(source) = chunk.source.as_deref() {
            by_source.entry(source.to_string()).or_default().push(chunk);
        }
    }
    let all_sources: Vec<String> = by_source.keys().cloned().collect();
    let query_lower = paper_query.to_lowercase();
    let mut matches: Vec<(String, Vec<&RagChunk>)> = by_source
        .into_iter()
        .filter(|(source, _)| source_matches_query(source, &query_lower))
        .collect();

    if matches.is_empty() {
        let title_matches = state.title_index.search(paper_query);
        if title_matches.is_empty() {
            let mut sample = all_sources;
            sample.truncate(8);
            let mut message = String::from(
                "No paper matched that query. Try a filename like 2512.04226v1.pdf or a report title.",
            );
            if !sample.is_empty() {
                message.push_str("\nSample sources:\n- ");
                message.push_str(&sample.join("\n- "));
            }
            return Ok(message);
        }

        let mut with_source: Vec<&TitleEntry> = title_matches
            .iter()
            .copied()
            .filter(|entry| entry.source.is_some())
            .collect();
        if with_source.is_empty() {
            let mut unique_titles = BTreeSet::new();
            let mut message = String::from(
                "Matched a report title, but the PDF is not in the RAG index. \
Run download/ingest to add papers.\nTitles:\n- ",
            );
            for entry in title_matches {
                unique_titles.insert(format_title_candidate(entry));
            }
            message.push_str(&unique_titles.into_iter().take(8).collect::<Vec<_>>().join("\n- "));
            return Ok(message);
        }

        with_source.sort_by(|a, b| a.source.cmp(&b.source));
        with_source.dedup_by(|a, b| a.source == b.source);
        if with_source.len() > 1 {
            let mut message =
                String::from("Multiple papers matched that title. Please be more specific.\n- ");
            let options: Vec<String> = with_source
                .iter()
                .take(8)
                .map(|entry| format_title_candidate(entry))
                .collect();
            message.push_str(&options.join("\n- "));
            return Ok(message);
        }

        let entry = with_source
            .pop()
            .ok_or_else(|| anyhow::anyhow!("title match disappeared"))?;
        if let Some(source) = &entry.source {
            matches.push((source.clone(), Vec::new()));
        }
    }

    if matches.len() > 1 {
        matches.sort_by(|a, b| a.0.cmp(&b.0));
        let mut options: Vec<String> = matches.iter().map(|(source, _)| source.clone()).collect();
        options.truncate(8);
        let mut message = String::from("Multiple papers matched. Please be more specific.\n- ");
        message.push_str(&options.join("\n- "));
        return Ok(message);
    }

    let (source, mut chunks) = matches
        .pop()
        .ok_or_else(|| anyhow::anyhow!("paper match disappeared"))?;
    if chunks.is_empty() {
        chunks = state
            .rag_index
            .chunks
            .iter()
            .filter(|chunk| chunk.source.as_deref() == Some(source.as_str()))
            .collect();
    }
    chunks.sort_by_key(|chunk| chunk.chunk_index.unwrap_or(usize::MAX));
    let sections = build_paper_sections(&chunks, state.config.rag_paper_map_max_chars)?;
    let content_hash = paper_content_hash(&chunks);
    let cached_summary = load_paper_summary_cache(&state.config, &source, &content_hash)?;
    let summary = if let Some(summary) = cached_summary {
        summary
    } else {
        let section_summaries =
            summarize_paper_sections(&state.http_client, &state.config, &source, &sections).await?;
        let summary = reduce_paper_summary(
            &state.http_client,
            &state.config,
            &source,
            &section_summaries,
        )
        .await?;
        if let Err(err) = write_paper_summary_cache(&state.config, &source, &content_hash, &summary)
        {
            eprintln!("Failed to write paper summary cache: {err}");
        }
        summary
    };

    let prompt = format!(
        "You are a GPU research assistant. Use only the paper summary below. \
If the answer is not in the summary, say you don't know.\n\n\
Paper summary:\n{summary}\n\nQuestion: {question}\nAnswer:",
        summary = summary,
        question = question
    );
    let answer = openai_response_text(&state.http_client, &state.config, &prompt).await?;
    let title = title_for_source(&state.title_index, &source);
    let context_note =
        build_paper_context_note(&source, title.as_deref(), chunks.len(), sections.len());
    let question_label = paper_label_for_question(title.as_deref(), &source);
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

fn build_context(matches: &[&RagChunk], max_chars: usize) -> Result<String> {
    let mut out = String::new();
    for (idx, chunk) in matches.iter().enumerate() {
        let source = chunk
            .source
            .as_deref()
            .unwrap_or("unknown-source");
        let entry = format!(
            "[{}] {} (chunk {:?})\n{}\n\n",
            idx + 1,
            source,
            chunk.chunk_index,
            chunk.text
        );
        if out.len() + entry.len() > max_chars {
            break;
        }
        out.push_str(&entry);
    }
    if out.trim().is_empty() {
        bail!("context empty after truncation");
    }
    Ok(out)
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

fn format_title_candidate(entry: &TitleEntry) -> String {
    match (&entry.arxiv_id, &entry.source) {
        (Some(_arxiv_id), Some(source)) => format!("{} ({})", entry.title, source),
        (Some(arxiv_id), None) => format!("{} (arXiv {})", entry.title, arxiv_id),
        _ => entry.title.clone(),
    }
}

fn source_matches_query(source: &str, query_lower: &str) -> bool {
    let source_lower = source.to_lowercase();
    if source_lower.contains(query_lower) {
        return true;
    }
    if let Some(name) = Path::new(source).file_name().and_then(|name| name.to_str()) {
        return name.to_lowercase().contains(query_lower);
    }
    false
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

async fn summarize_paper_sections(
    client: &HttpClient,
    config: &Config,
    source: &str,
    sections: &[String],
) -> Result<Vec<String>> {
    let total = sections.len();
    let concurrency = config.rag_paper_map_concurrency.max(1);
    if concurrency == 1 {
        let mut summaries = Vec::new();
        for (idx, section) in sections.iter().enumerate() {
            let prompt = format!(
                "You are summarizing section {part} of {total} from paper {source}. \
Extract the motivation, problem statement, approach, key results, and limitations. \
Keep it concise and factual. Use short bullet points.\n\n\
Section text:\n{section}\n\nSummary:",
                part = idx + 1,
                total = total,
                source = source,
                section = section
            );
            let summary = openai_response_text(client, config, &prompt).await?;
            summaries.push(summary);
        }
        return Ok(summaries);
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
        let section = section.clone();
        set.spawn(async move {
            let _permit = permit;
            let prompt = format!(
                "You are summarizing section {part} of {total} from paper {source}. \
Extract the motivation, problem statement, approach, key results, and limitations. \
Keep it concise and factual. Use short bullet points.\n\n\
Section text:\n{section}\n\nSummary:",
                part = idx + 1,
                total = total,
                source = source,
                section = section
            );
            let summary = openai_response_text(&client, &config, &prompt).await?;
            Ok::<_, anyhow::Error>((idx, summary))
        });
    }

    let mut summaries = Vec::with_capacity(total);
    while let Some(result) = set.join_next().await {
        let item = result.context("join summary task")??;
        summaries.push(item);
    }
    summaries.sort_by_key(|(idx, _)| *idx);
    Ok(summaries.into_iter().map(|(_, summary)| summary).collect())
}

async fn reduce_paper_summary(
    client: &HttpClient,
    config: &Config,
    source: &str,
    summaries: &[String],
) -> Result<String> {
    if summaries.is_empty() {
        bail!("paper summary sections missing");
    }
    let combined = summaries.join("\n\n");
    if combined.len() <= config.rag_paper_reduce_max_chars {
        return Ok(combined);
    }
    let prompt = format!(
        "You are given section summaries for paper {source}. Merge them into a compact \
summary under {limit} characters. Include motivation, approach, results, and limitations. \
Use short bullet points.\n\nSummaries:\n{combined}\n\nCondensed summary:",
        source = source,
        limit = config.rag_paper_reduce_max_chars,
        combined = combined
    );
    let condensed = openai_response_text(client, config, &prompt).await?;
    Ok(truncate_text(
        &condensed,
        config.rag_paper_reduce_max_chars,
    ))
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

fn cache_path_for_source(config: &Config, source: &str) -> Result<String> {
    let dir = config.rag_paper_summary_cache_dir.trim();
    if dir.is_empty() {
        bail!("RAG_PAPER_SUMMARY_CACHE_DIR is empty");
    }
    let filename = format!("{}.json", sanitize_cache_key(source));
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
    content_hash: &str,
) -> Result<Option<String>> {
    if config.rag_paper_summary_cache_dir.trim().is_empty() {
        return Ok(None);
    }
    let path = cache_path_for_source(config, source)?;
    let payload = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("read cache {path}")),
    };
    let cache: PaperSummaryCache =
        serde_json::from_str(&payload).context("parse paper summary cache")?;
    if cache.source != source
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
    if cache.summary.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(cache.summary))
}

fn write_paper_summary_cache(
    config: &Config,
    source: &str,
    content_hash: &str,
    summary: &str,
) -> Result<()> {
    let dir = config.rag_paper_summary_cache_dir.trim();
    if dir.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir).with_context(|| format!("create {dir}"))?;
    let path = cache_path_for_source(config, source)?;
    let cache = PaperSummaryCache {
        source: source.to_string(),
        content_hash: content_hash.to_string(),
        model: config.openai_chat_model.clone(),
        map_max_chars: config.rag_paper_map_max_chars,
        reduce_max_chars: config.rag_paper_reduce_max_chars,
        created_at: now_unix_seconds()?,
        summary: summary.to_string(),
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

fn format_reply(
    question: &str,
    paper_label: Option<&str>,
    answer: &str,
    context_note: &str,
    max: usize,
) -> String {
    let question = truncate_text(question.trim(), 400);
    let paper = paper_label
        .and_then(|label| {
            let trimmed = label.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(truncated_label(trimmed))
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
    let context = truncate_text(context, 600);

    let question = if question.is_empty() {
        "n/a".to_string()
    } else {
        question
    };
    let answer = answer.trim();
    let header = format!("### Paper\n{paper}\n\n### Question\n{question}\n\n### Answer\n");
    let footer = format!("\n\n### Context\n{context}");
    if header.len() + footer.len() >= max {
        let combined = format!(
            "### Paper\n{paper}\n\n### Question\n{question}\n\n### Answer\n\n### Context\n{context}"
        );
        return truncate_text(&combined, max);
    }
    let max_answer = max.saturating_sub(header.len() + footer.len());
    let answer = truncate_text(answer, max_answer);
    format!("{header}{answer}{footer}")
}

fn truncated_label(label: &str) -> String {
    truncate_text(label, 200)
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
