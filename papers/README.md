# papers/

Store full-text paper PDFs here for RAG ingestion.

The ingestion script scans this directory recursively for `.pdf` files.

## Download helper

To pull arXiv PDFs from report links:

```bash
python scripts/download_papers.py --reports-dir reports --out-dir papers
```
