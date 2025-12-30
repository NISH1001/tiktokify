---
title: TikTokify
emoji: 📱
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.12.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# TikTokify

A CLI tool that generates a TikTok-style swipeable blog viewer with recommendations and optional Wikipedia suggestions.

## Installation

```bash
cd tools/tiktokify
uv sync
```

## Usage

Basic usage (without Wikipedia enrichment):
```bash
uv run tiktokify -u https://nish1001.github.io -o ./tiktokify/index.html
```

With Wikipedia suggestions (requires LLM API key):
```bash
# Using OpenAI
export OPENAI_API_KEY=your-key
uv run tiktokify -u https://nish1001.github.io -o ./tiktokify/index.html -m gpt-4o-mini

# Using Anthropic
export ANTHROPIC_API_KEY=your-key
uv run tiktokify -u https://nish1001.github.io -o ./tiktokify/index.html -m claude-3-haiku-20240307

# Using local Ollama
uv run tiktokify -u https://nish1001.github.io -o ./tiktokify/index.html -m ollama/llama3
```

## Options

| Option | Description |
|--------|-------------|
| `-u, --base-url` | Base URL of the Jekyll blog (required) |
| `-o, --output-html` | Output path for generated HTML (required) |
| `-m, --model` | LLM model for Wikipedia suggestions (optional) |
| `--content-weight` | Weight for content similarity (default: 0.6) |
| `--metadata-weight` | Weight for tag/category similarity (default: 0.4) |
| `--top-k` | Number of recommendations per post (default: 5) |
| `--max-concurrent` | Max concurrent requests (default: 5) |
| `-v, --verbose` | Enable verbose output |

## Features

- **Vertical swipe navigation** - TikTok-style full-screen post cards
- **Randomized start** - Different starting post each page load
- **Hybrid recommendations** - Combines TF-IDF content similarity with tag/category matching
- **Wikipedia integration** - LLM-suggested related Wikipedia articles (optional)
- **Keyboard navigation** - Arrow keys, j/k, Enter to open
- **Responsive** - Works on mobile and desktop

## Output

The generated HTML is a single self-contained file with:
- Embedded post metadata and recommendation graph
- CSS-based scroll snapping for smooth swiping
- On-demand content fetching
- No external dependencies
