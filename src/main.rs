use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use feed_rs::parser;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

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
    #[arg(long, default_value_t = 20)]
    max_per_source: usize,
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

#[derive(Clone, Debug)]
struct Item {
    title: String,
    url: String,
    published: Option<String>,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(&cli.config)?;
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

    let mut visited = load_visited(&cli.db)?;
    let mut new_records = Vec::new();
    let mut sections = Vec::new();
    let mut already_seen = 0usize;
    let mut failed_sources = 0usize;

    let client = Client::builder()
        .user_agent("research-bot/0.1 (+github actions)")
        .build()
        .context("build http client")?;

    for source in config.sources {
        match fetch_source(&client, &source) {
            Ok(items) => {
                let mut report_items = Vec::new();
                let mut new_count = 0usize;
                for item in items {
                    if visited.contains(&item.url) {
                        already_seen += 1;
                        continue;
                    }
                    visited.insert(item.url.clone());
                    new_count += 1;
                    if report_items.len() < cli.max_per_source {
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

    write_report(&out_path, &sections, already_seen, failed_sources)?;
    append_visited(&cli.db, &new_records)?;

    Ok(())
}

fn load_config(path: &Path) -> Result<Config> {
    let raw = fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
    let config = serde_yaml::from_str(&raw).context("parse config yaml")?;
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

fn fetch_source(client: &Client, source: &Source) -> Result<Vec<Item>> {
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
        let published = entry
            .published
            .or(entry.updated)
            .map(format_date);
        items.push(Item {
            title,
            url,
            published,
        });
    }
    Ok(items)
}

fn format_date(dt: DateTime<Utc>) -> String {
    dt.date_naive().to_string()
}

fn write_report(
    path: &Path,
    sections: &[Section],
    already_seen: usize,
    failed_sources: usize,
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
            writeln!(file, "- No new items found.")?;
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
