# data/

Configuration and state files for the research bot.

## sources.yml

A YAML list of feed sources. Each entry has:

- `name`: label used in the report
- `url`: RSS/Atom feed URL

Example:

```yaml
sources:
  - name: "arXiv: GPU OR CUDA"
    url: "http://export.arxiv.org/api/query?search_query=all:GPU+OR+all:CUDA&start=0&max_results=50"
```

## visited.jsonl

A JSONL database of seen items used to dedupe future runs. Each line is a JSON object like:

```json
{"url":"...","title":"...","source":"...","published":"YYYY-MM-DD","fetched_at":"YYYY-MM-DDTHH:MM:SSZ"}
```
