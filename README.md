---
title: TikTokify
emoji: 📱
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# TikTokify

A CLI tool that generates a TikTok-style swipeable blog viewer with hybrid recommendations, semantic embeddings, and optional LLM enrichment.

## Installation

```bash
uv sync
```

## Usage

Basic usage:
```bash
uv run tiktokify -u https://nish1001.github.io -o ./output/index.html
```

With deeper crawling:
```bash
uv run tiktokify -u https://example.com -o output.html --max-depth 2 -v
```

With semantic embeddings (local, no API key needed):
```bash
uv run tiktokify -u https://example.com -o output.html \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-weight 0.3
```

With LLM enrichment (key points + Wikipedia):
```bash
export OPENAI_API_KEY=your-key
uv run tiktokify -u https://example.com -o output.html -m gpt-4o-mini
```

With external link crawling:
```bash
uv run tiktokify -u https://example.com -o output.html \
  --max-depth 3 \
  --follow-external \
  --external-depth 1
```

Debug mode (fast testing):
```bash
uv run tiktokify -u https://example.com -o output.html --debug
```

## Options

### Core Options

| Option | Description |
|--------|-------------|
| `-u, --base-url` | Base URL to crawl (required) |
| `-o, --output-html` | Output path for generated HTML (required) |
| `-m, --model` | LLM model for enrichment (e.g., `gpt-4o-mini`, `claude-3-haiku-20240307`) |
| `-v, --verbose` | Enable verbose output |
| `--debug` | Debug mode: limit to 5 posts, skip external sources, enable verbose |
| `--limit` | Limit number of posts to process |

### Crawling Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-depth` | 1 | Spider crawl depth (0=seed only, 1=seed+links, 2=two levels) |
| `--follow-external/--no-follow-external` | off | Follow external links during crawling |
| `--external-depth` | 1 | Max depth for external links (only when `--follow-external`) |
| `--max-concurrent` | 5 | Maximum concurrent requests |
| `--stealth/--no-stealth` | on | Stealth mode for anti-detection |
| `--headless/--no-headless` | on | Run browser in headless mode |

### Recommendation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--content-weight` | 0.6 | Weight for TF-IDF content similarity |
| `--metadata-weight` | 0.4 | Weight for tag/category similarity |
| `--embedding-model` | - | Embedding model for semantic similarity (e.g., `sentence-transformers/all-MiniLM-L6-v2` or `text-embedding-3-small`) |
| `--embedding-weight` | 0.3 | Weight for semantic similarity (when embedding model set) |
| `--top-k` | 5 | Number of recommendations per post |
| `--min-similarity` | 0.0 | Minimum similarity score threshold |

### Filtering Options

| Option | Default | Description |
|--------|---------|-------------|
| `--filter-meta-pages/--no-filter-meta-pages` | on | Filter out meta pages (tags, categories, about) |
| `--min-word-count` | 100 | Minimum word count for content filter |
| `--use-llm-filter/--no-llm-filter` | off | Use LLM for content quality assessment |
| `--skip-url-filter` | off | Skip URL pattern filtering |
| `--skip-content-filter` | off | Skip content quality filtering |

### Enrichment Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n-key-points` | 5 | Number of key points to generate per post |
| `--n-wiki` | 3 | Number of Wikipedia articles to suggest per post |
| `--sources` | - | Comma-separated external sources: `hackernews`, `hn`, `links`, `hn-frontpage` |
| `--n-external` | 3 | Number of items to fetch per external source |
| `-t, --temperature` | 0.5 | LLM temperature |

### Cache Options

| Option | Description |
|--------|-------------|
| `--save-cache` | Save intermediate data to JSON file for reuse |
| `--load-cache` | Load from cached JSON file (skip crawling/enrichment) |

## Features

- **Single-pass crawling** - Efficient crawling that fetches each page only once
- **Vertical swipe navigation** - TikTok-style full-screen post cards
- **Hybrid recommendations** - Combines TF-IDF, metadata, and semantic embeddings
- **Semantic embeddings** - Local (sentence-transformers) or API (OpenAI, etc.)
- **Cross-domain crawling** - Follow external links with separate depth control
- **LLM enrichment** - Key points extraction + Wikipedia suggestions
- **External sources** - HackerNews discussions, linked content
- **Content filtering** - Automatic filtering of low-quality/meta pages
- **Keyboard navigation** - Arrow keys, j/k, Enter to open
- **Responsive** - Works on mobile and desktop

## Output

The generated HTML is a single self-contained file with:
- Embedded post metadata and recommendation graph
- CSS-based scroll snapping for smooth swiping
- On-demand content fetching
- No external dependencies
