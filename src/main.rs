use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Duration, NaiveDate, Utc};
use clap::Parser;
use dotenvy::dotenv;
use feed_rs::parser;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Parser)]
#[command(name = "research-bot", version, about = "Daily GPU research digest generator")]
struct Cli {
    #[arg(long, default_value = "data/sources.yml")]
    config: PathBuf,
    #[arg(long, default_value = "data/visited.jsonl")]
    db: PathBuf,
    #[arg(long)]
    reset_db: bool,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = 90)]
    max_age_days: i64,
    #[arg(long, default_value_t = 20)]
    max_per_source: usize,
    #[arg(long)]
    summarize: bool,
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,
    #[arg(long, default_value = "gpt-5.2")]
    openai_model: String,
    #[arg(long, default_value_t = 2000)]
    summary_max_chars: usize,
    #[arg(long)]
    discover: bool,
    #[arg(long, default_value = "data/discovery.yml")]
    discovery_config: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Config {
    sources: Vec<Source>,
}

#[derive(Debug, Deserialize)]
struct Source {
    name: String,
    url: String,
}

#[derive(Debug, Deserialize)]
struct DiscoveryConfig {
    topics: Vec<String>,
    max_queries: usize,
    max_results_per_query: usize,
}

#[derive(Clone, Debug)]
struct Item {
    title: String,
    url: String,
    published: Option<String>,
    abstract_text: Option<String>,
    summary: Option<String>,
}

#[derive(Debug)]
struct Section {
    name: String,
    items: Vec<Item>,
    new_count: usize,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VisitedRecord {
    url: String,
    title: String,
    source: String,
    published: Option<String>,
    fetched_at: String,
}

#[derive(Debug)]
struct OpenAiConfig {
    api_key: String,
    model: String,
    summary_max_chars: usize,
}

#[derive(Debug, Deserialize)]
struct WebSearchResult {
    title: String,
    url: String,
    snippet: Option<String>,
    published: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WebSearchResponse {
    results: Vec<WebSearchResult>,
}

#[derive(Debug, Deserialize)]
struct DiscoveryQueryResponse {
    queries: Vec<String>,
}

fn main() -> Result<()> {
    dotenv().ok();
    let cli = Cli::parse();
    let config = load_config(&cli.config)?;
    if cli.max_age_days < 1 {
        bail!("--max-age-days must be >= 1");
    }
    if cli.summary_max_chars < 1 {
        bail!("--summary-max-chars must be >= 1");
    }
    let discovery = if cli.discover {
        Some(load_discovery_config(&cli.discovery_config)?)
    } else {
        None
    };
    if let Some(config) = &discovery {
        if config.topics.is_empty() {
            bail!("discovery topics must be non-empty");
        }
        if config.max_queries < 1 {
            bail!("discovery max_queries must be >= 1");
        }
        if config.max_results_per_query < 1 {
            bail!("discovery max_results_per_query must be >= 1");
        }
    }
    let out_path = cli.out.unwrap_or_else(default_report_path);

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).context("create report directory")?;
    }
    if let Some(parent) = cli.db.parent() {
        fs::create_dir_all(parent).context("create data directory")?;
    }

    if cli.reset_db {
        reset_db(&cli.db)?;
    }

    let needs_openai = cli.summarize || cli.discover;
    let openai = if needs_openai {
        let api_key = cli.openai_api_key.clone().ok_or_else(|| {
            anyhow!("OPENAI_API_KEY is required when --summarize or --discover is set")
        })?;
        Some(OpenAiConfig {
            api_key,
            model: cli.openai_model.clone(),
            summary_max_chars: cli.summary_max_chars,
        })
    } else {
        None
    };

    let mut visited = load_visited(&cli.db)?;
    let mut new_records = Vec::new();
    let mut sections = Vec::new();
    let mut already_seen = 0usize;
    let mut failed_sources = 0usize;

    let client = Client::builder()
        .user_agent("research-bot/0.1 (+github actions)")
        .build()
        .context("build http client")?;

    let cutoff = Utc::now() - Duration::days(cli.max_age_days);
    for source in config.sources {
        match fetch_source(&client, &source, cutoff) {
            Ok(items) => {
                let mut report_items = Vec::new();
                let mut new_count = 0usize;
                for mut item in items {
                    if visited.contains(&item.url) {
                        already_seen += 1;
                        continue;
                    }
                    visited.insert(item.url.clone());
                    new_count += 1;
                    if report_items.len() < cli.max_per_source {
                        if cli.summarize {
                            let openai = openai.as_ref().expect("openai config missing");
                            let abstract_text = item
                                .abstract_text
                                .as_ref()
                                .ok_or_else(|| anyhow!("missing abstract for {}", item.url))?;
                            let summary = summarize_openai(
                                &client,
                                openai,
                                &item.title,
                                abstract_text,
                            )
                            .with_context(|| format!("summarize {}", item.url))?;
                            item.summary = Some(summary);
                        }
                        report_items.push(item.clone());
                    }
                    new_records.push(VisitedRecord {
                        url: item.url,
                        title: item.title,
                        source: source.name.clone(),
                        published: item.published,
                        fetched_at: Utc::now().to_rfc3339(),
                    });
                }
                sections.push(Section {
                    name: source.name,
                    items: report_items,
                    new_count,
                    error: None,
                });
            }
            Err(err) => {
                failed_sources += 1;
                eprintln!("warning: source '{}' failed: {}", source.name, err);
                sections.push(Section {
                    name: source.name,
                    items: Vec::new(),
                    new_count: 0,
                    error: Some(err.to_string()),
                });
            }
        }
    }

    if let (Some(discovery), Some(openai)) = (&discovery, &openai) {
        let queries = generate_discovery_queries(&client, openai, discovery, cli.max_age_days)
            .context("generate discovery queries")?;
        for query in queries {
            match search_openai_web(
                &client,
                openai,
                &query,
                discovery.max_results_per_query,
                cli.max_age_days,
                cli.summarize,
            ) {
                Ok(items) => {
                    let mut report_items = Vec::new();
                    let mut new_count = 0usize;
                    for mut item in items {
                        if visited.contains(&item.url) {
                            already_seen += 1;
                            continue;
                        }
                        visited.insert(item.url.clone());
                        new_count += 1;
                        if report_items.len() < cli.max_per_source {
                            if cli.summarize {
                                let abstract_text = item
                                    .abstract_text
                                    .as_ref()
                                    .ok_or_else(|| {
                                        anyhow!("missing snippet for {}", item.url)
                                    })?;
                                let summary = summarize_openai(
                                    &client,
                                    openai,
                                    &item.title,
                                    abstract_text,
                                )
                                .with_context(|| format!("summarize {}", item.url))?;
                                item.summary = Some(summary);
                            }
                            report_items.push(item.clone());
                        }
                        new_records.push(VisitedRecord {
                            url: item.url,
                            title: item.title,
                            source: format!("Search: {}", query),
                            published: item.published,
                            fetched_at: Utc::now().to_rfc3339(),
                        });
                    }
                    sections.push(Section {
                        name: format!("Search: {}", query),
                        items: report_items,
                        new_count,
                        error: None,
                    });
                }
                Err(err) => {
                    failed_sources += 1;
                    eprintln!("warning: search '{}' failed: {}", query, err);
                    sections.push(Section {
                        name: format!("Search: {}", query),
                        items: Vec::new(),
                        new_count: 0,
                        error: Some(err.to_string()),
                    });
                }
            }
        }
    }

    write_report(
        &out_path,
        &sections,
        already_seen,
        failed_sources,
        cli.max_age_days,
    )?;
    append_visited(&cli.db, &new_records)?;

    Ok(())
}

fn load_config(path: &Path) -> Result<Config> {
    let raw = fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
    let config = serde_yaml::from_str(&raw).context("parse config yaml")?;
    Ok(config)
}

fn load_discovery_config(path: &Path) -> Result<DiscoveryConfig> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
    let config = serde_yaml::from_str(&raw).context("parse discovery yaml")?;
    Ok(config)
}

fn default_report_path() -> PathBuf {
    let today = Utc::now().date_naive();
    PathBuf::from(format!("reports/{}.md", today.format("%Y-%m-%d")))
}

fn load_visited(path: &Path) -> Result<HashSet<String>> {
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let file = File::open(path).with_context(|| format!("open visited db {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut visited = HashSet::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<VisitedRecord>(trimmed) {
            Ok(record) => {
                visited.insert(record.url);
            }
            Err(err) => {
                eprintln!("warning: invalid visited entry: {}", err);
            }
        }
    }
    Ok(visited)
}

fn reset_db(path: &Path) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("reset visited db {}", path.display()))?;
    file.flush().context("flush reset db")?;
    Ok(())
}

fn fetch_source(client: &Client, source: &Source, cutoff: DateTime<Utc>) -> Result<Vec<Item>> {
    let resp = client
        .get(&source.url)
        .send()
        .with_context(|| format!("fetch {}", source.url))?
        .error_for_status()
        .with_context(|| format!("bad status for {}", source.url))?;
    let bytes = resp.bytes().context("read response body")?;
    let feed = parser::parse(&bytes[..]).context("parse feed")?;

    let mut items = Vec::new();
    for entry in feed.entries {
        let url = entry
            .links
            .iter()
            .map(|link| link.href.clone())
            .find(|href| !href.trim().is_empty())
            .or_else(|| {
                if entry.id.trim().is_empty() {
                    None
                } else {
                    Some(entry.id.clone())
                }
            });
        let Some(url) = url else { continue };
        let title = entry
            .title
            .as_ref()
            .map(|text| text.content.clone())
            .unwrap_or_else(|| "Untitled".to_string());
        let published_at = entry
            .published
            .or(entry.updated)
            .map(|dt| dt.with_timezone(&Utc));
        let Some(published_at) = published_at else {
            continue;
        };
        if published_at < cutoff {
            continue;
        }
        let published = Some(format_date(published_at));
        let abstract_text = extract_abstract(&entry)
            .as_deref()
            .map(normalize_text);
        items.push(Item {
            title,
            url,
            published,
            abstract_text,
            summary: None,
        });
    }
    Ok(items)
}

fn format_date(dt: DateTime<Utc>) -> String {
    dt.date_naive().to_string()
}

fn extract_abstract(entry: &feed_rs::model::Entry) -> Option<String> {
    entry
        .summary
        .as_ref()
        .map(|text| text.content.clone())
        .or_else(|| entry.content.as_ref().and_then(|content| content.body.clone()))
}

fn normalize_text(input: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn openai_responses_text(
    client: &Client,
    api_key: &str,
    body: serde_json::Value,
    context: &str,
) -> Result<String> {
    let response = client
        .post("https://api.openai.com/v1/responses")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .with_context(|| format!("send OpenAI response for {}", context))?;

    let status = response.status();
    let text = response
        .text()
        .with_context(|| format!("read OpenAI response body for {}", context))?;
    if !status.is_success() {
        bail!(
            "OpenAI response error for {}: {} {}",
            context,
            status,
            text
        );
    }

    let value: serde_json::Value =
        serde_json::from_str(&text).context("parse OpenAI response")?;
    if let Some(output_text) = value.get("output_text").and_then(|content| content.as_str()) {
        let trimmed = output_text.trim();
        if trimmed.is_empty() {
            bail!("OpenAI response returned empty output_text for {}", context);
        }
        return Ok(trimmed.to_string());
    }

    let mut chunks = Vec::new();
    if let Some(output) = value.get("output").and_then(|content| content.as_array()) {
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

    if !chunks.is_empty() {
        return Ok(chunks.join("\n"));
    }

    bail!(
        "OpenAI response missing output text for {}: {}",
        context,
        truncate_preview(&text, 1000)
    );
}

fn truncate_preview(input: &str, max: usize) -> String {
    if input.len() <= max {
        return input.to_string();
    }
    let mut end = max;
    while end > 0 && !input.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = input[..end].to_string();
    out.push_str("...");
    out
}

fn truncate_chars(input: &str, max: usize) -> String {
    if input.len() <= max {
        return input.to_string();
    }
    let mut end = max;
    while end > 0 && !input.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = input[..end].to_string();
    out.push_str("...");
    out
}

fn generate_discovery_queries(
    client: &Client,
    config: &OpenAiConfig,
    discovery: &DiscoveryConfig,
    max_age_days: i64,
) -> Result<Vec<String>> {
    let topics = discovery.topics.join(", ");
    let prompt = format!(
        "You generate precise web search queries for technical research.\n\n\
Topics: {topics}\n\n\
Generate exactly {count} web search queries about recent GPU research. \
Focus on matrix multiplication, CUDA, GPU architecture, and AI/ML acceleration. \
Limit to the last {days} days. Return JSON only: {{\"queries\":[...]}}.",
        topics = topics,
        count = discovery.max_queries,
        days = max_age_days
    );

    let body = json!({
        "model": config.model,
        "input": prompt,
        "text": { "format": { "type": "json_object" } },
        "temperature": 0.2,
        "max_output_tokens": 220
    });

    let content = openai_responses_text(client, &config.api_key, body, "generate queries")?;
    let parsed: DiscoveryQueryResponse =
        serde_json::from_str(&content).context("parse discovery query JSON")?;
    let queries = parsed.queries;
    if queries.len() != discovery.max_queries {
        bail!(
            "expected {} queries, got {}",
            discovery.max_queries,
            queries.len()
        );
    }
    if queries.iter().any(|query| query.trim().is_empty()) {
        bail!("discovery queries must be non-empty");
    }
    Ok(queries)
}

fn summarize_openai(
    client: &Client,
    config: &OpenAiConfig,
    title: &str,
    abstract_text: &str,
) -> Result<String> {
    let mut trimmed = abstract_text.to_string();
    if trimmed.len() > config.summary_max_chars {
        let mut end = config.summary_max_chars;
        while end > 0 && !trimmed.is_char_boundary(end) {
            end -= 1;
        }
        trimmed.truncate(end);
    }

    let prompt = format!(
        "Title: {title}\nAbstract: {abstract}\n\n\
Return 2-3 concise bullet points. Focus on GPU relevance and matrix multiplication if present. \
Return only the bullets.",
        title = title,
        abstract = trimmed
    );

    let body = json!({
        "model": config.model,
        "input": format!(
            "You summarize research papers for GPU practitioners.\n\n{prompt}",
            prompt = prompt
        ),
        "temperature": 0.2,
        "max_output_tokens": 180
    });

    openai_responses_text(client, &config.api_key, body, "summarize")
}

fn search_openai_web(
    client: &Client,
    config: &OpenAiConfig,
    query: &str,
    max_results: usize,
    max_age_days: i64,
    require_snippet: bool,
) -> Result<Vec<Item>> {
    let cutoff = Utc::now() - Duration::days(max_age_days);
    let prompt = format!(
        "You search the web and return structured JSON results.\n\n\
Query: {query}\n\n\
Use web search to find up to {max_results} results from the last {days} days. \
If a result does not have a full published date, omit it. \
Return JSON only: {{\"results\":[{{\"title\":\"...\",\"url\":\"...\",\"snippet\":\"...\",\"published\":\"YYYY-MM-DD\"}}]}}.",
        query = query,
        max_results = max_results,
        days = max_age_days
    );

    let body = json!({
        "model": config.model,
        "input": prompt,
        "tools": [
            { "type": "web_search" }
        ],
        "temperature": 0.2,
        "max_output_tokens": 500
    });

    let raw = openai_responses_text(client, &config.api_key, body, "web_search")?;
    let parsed = extract_web_search_results(client, config, &raw, max_results, max_age_days)
        .context("extract web_search JSON")?;

    if parsed.results.len() > max_results {
        bail!(
            "web_search returned {} results (max {})",
            parsed.results.len(),
            max_results
        );
    }

    let mut items = Vec::new();
    for result in parsed.results {
        if result.title.trim().is_empty() {
            bail!("web_search returned empty title");
        }
        if result.url.trim().is_empty() {
            bail!("web_search returned empty url");
        }
        let Some(published_raw) = result.published.as_ref() else {
            continue;
        };
        let published_date = match NaiveDate::parse_from_str(published_raw, "%Y-%m-%d") {
            Ok(date) => date,
            Err(_) => continue,
        };
        let published_at = published_date.and_hms_opt(0, 0, 0).ok_or_else(|| {
            anyhow!("web_search invalid published date '{}'", published_raw)
        })?;
        let published_at = DateTime::<Utc>::from_naive_utc_and_offset(published_at, Utc);
        if published_at < cutoff {
            continue;
        }
        let abstract_text = result.snippet.map(|snippet| normalize_text(&snippet));
        if require_snippet && abstract_text.as_ref().map(|s| s.is_empty()).unwrap_or(true) {
            continue;
        }
        items.push(Item {
            title: result.title,
            url: result.url,
            published: Some(published_date.to_string()),
            abstract_text,
            summary: None,
        });
    }
    if items.is_empty() {
        bail!(
            "web_search returned no in-window results (cutoff {})",
            cutoff.date_naive()
        );
    }
    Ok(items)
}

fn extract_web_search_results(
    client: &Client,
    config: &OpenAiConfig,
    raw: &str,
    max_results: usize,
    max_age_days: i64,
) -> Result<WebSearchResponse> {
    let clipped = truncate_chars(raw, 6000);
    let prompt = format!(
        "Extract up to {max_results} results from the raw web search output below. \
Only include results with a full published date in YYYY-MM-DD within the last {days} days. \
Return JSON only: {{\"results\":[{{\"title\":\"...\",\"url\":\"...\",\"snippet\":\"...\",\"published\":\"YYYY-MM-DD\"}}]}}.\n\n\
Raw output:\n{raw}",
        max_results = max_results,
        days = max_age_days,
        raw = clipped
    );

    let body = json!({
        "model": config.model,
        "input": prompt,
        "text": { "format": { "type": "json_object" } },
        "temperature": 0.0,
        "max_output_tokens": 500
    });

    let content =
        openai_responses_text(client, &config.api_key, body, "extract web_search")?;
    let parsed: WebSearchResponse =
        serde_json::from_str(&content).context("parse extracted web_search JSON")?;
    Ok(parsed)
}

fn write_report(
    path: &Path,
    sections: &[Section],
    already_seen: usize,
    failed_sources: usize,
    max_age_days: i64,
) -> Result<()> {
    let generated_at = Utc::now();
    let total_new: usize = sections.iter().map(|section| section.new_count).sum();
    let total_sources = sections.len();

    let mut file = File::create(path).with_context(|| format!("create report {}", path.display()))?;
    writeln!(
        file,
        "# GPU Research Digest — {}",
        generated_at.date_naive().format("%Y-%m-%d")
    )?;
    writeln!(file, "Generated: {}", generated_at.to_rfc3339())?;
    writeln!(file, "Sources: {}", total_sources)?;
    writeln!(file, "New items: {}", total_new)?;
    writeln!(file, "Already seen: {}", already_seen)?;
    writeln!(file, "Window: last {} days", max_age_days)?;
    if failed_sources > 0 {
        writeln!(file, "Sources failed: {}", failed_sources)?;
    }

    for section in sections {
        writeln!(file)?;
        if section.error.is_some() {
            writeln!(file, "## {} (error)", section.name)?;
            if let Some(error) = &section.error {
                writeln!(file, "- Fetch error: {}", error)?;
            }
            continue;
        }

        if section.new_count == 0 {
            writeln!(file, "## {}", section.name)?;
            writeln!(file, "- No new items found in the last {} days.", max_age_days)?;
            continue;
        }

        writeln!(
            file,
            "## {} (new: {}, showing: {})",
            section.name,
            section.new_count,
            section.items.len()
        )?;
        for item in &section.items {
            match &item.published {
                Some(date) => {
                    writeln!(file, "- [{}]({}) — {}", item.title, item.url, date)?;
                }
                None => {
                    writeln!(file, "- [{}]({})", item.title, item.url)?;
                }
            }
            if let Some(summary) = &item.summary {
                for line in summary.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    writeln!(file, "  {}", trimmed)?;
                }
            }
        }
    }

    Ok(())
}

fn append_visited(path: &Path, records: &[VisitedRecord]) -> Result<()> {
    if records.is_empty() {
        return Ok(());
    }
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open visited db {}", path.display()))?;
    let mut writer = io::BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record).context("write visited record")?;
        writeln!(writer)?;
    }
    writer.flush().context("flush visited db")?;
    Ok(())
}
