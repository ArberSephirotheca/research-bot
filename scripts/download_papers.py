#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Iterable, List, Optional, Set
from urllib.parse import urlparse

import requests


ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}


def list_reports(reports_dir: str) -> List[str]:
    items = []
    for name in os.listdir(reports_dir):
        if not name.endswith(".md"):
            continue
        path = os.path.join(reports_dir, name)
        if os.path.isfile(path):
            items.append(path)
    items.sort()
    return items


def extract_urls(text: str) -> Iterable[str]:
    for match in re.finditer(r"\[[^\]]+\]\((https?://[^)]+)\)", text):
        yield match.group(1)


def extract_arxiv_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.netloc.lower()
    if host not in ARXIV_HOSTS:
        return None
    path = parsed.path
    if path.startswith("/abs/"):
        arxiv_id = path[len("/abs/") :]
    elif path.startswith("/pdf/"):
        arxiv_id = path[len("/pdf/") :]
        if arxiv_id.endswith(".pdf"):
            arxiv_id = arxiv_id[:-4]
    else:
        return None
    arxiv_id = arxiv_id.split("?")[0].split("#")[0].strip("/")
    if not arxiv_id:
        return None
    return arxiv_id


def pdf_url_for(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def safe_filename(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_") + ".pdf"


def download_pdf(url: str, out_path: str) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed: {url} (status {resp.status_code})")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as handle:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                handle.write(chunk)
    os.replace(tmp_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory containing report markdown files.",
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Specific report file to scan (repeatable).",
    )
    parser.add_argument(
        "--out-dir",
        default="papers",
        help="Output directory for downloaded PDFs.",
    )
    args = parser.parse_args()

    reports = list(args.report)
    if not reports:
        if not os.path.isdir(args.reports_dir):
            raise RuntimeError(f"Reports directory not found: {args.reports_dir}")
        reports = list_reports(args.reports_dir)
    if not reports:
        raise RuntimeError("No report files found.")

    arxiv_ids: Set[str] = set()
    for path in reports:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        for url in extract_urls(content):
            arxiv_id = extract_arxiv_id(url)
            if arxiv_id:
                arxiv_ids.add(arxiv_id)

    if not arxiv_ids:
        raise RuntimeError("No arXiv links found in reports.")

    os.makedirs(args.out_dir, exist_ok=True)
    downloaded = 0
    for arxiv_id in sorted(arxiv_ids):
        filename = safe_filename(arxiv_id)
        out_path = os.path.join(args.out_dir, filename)
        if os.path.exists(out_path):
            continue
        url = pdf_url_for(arxiv_id)
        download_pdf(url, out_path)
        downloaded += 1

    if downloaded == 0:
        print("No new PDFs downloaded.")
    else:
        print(f"Downloaded {downloaded} PDFs into {args.out_dir}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
