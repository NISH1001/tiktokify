"""Gradio web interface for TikTokify."""

import asyncio
import base64
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr

# Import tiktokify - add src to path if package not installed (e.g., HF Spaces)
try:
    from tiktokify.cli import _main_async
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from tiktokify.cli import _main_async

MODELS = ["", "gpt-4o-mini", "gpt-4o"]


async def generate(
    base_url: str,
    api_key: str,
    model: str,
    max_depth: int,
    n_key_points: int,
    n_wiki: int,
    temperature: float,
    sources: list[str],
    n_external: int,
    content_weight: float,
    metadata_weight: float,
    embedding_model: str,
    embedding_weight: float,
    top_k: int,
    max_concurrent: int,
    filter_meta_pages: bool,
    min_word_count: int,
    min_similarity: float,
    use_llm_filter: bool,
    skip_url_filter: bool,
    skip_content_filter: bool,
    stealth: bool,
    follow_external: bool,
    external_depth: int,
    limit: int | None,
    save_cache: bool,
    load_cache_file,
):
    """Run the TikTokify pipeline and return HTML file."""
    if not base_url and not load_cache_file:
        raise gr.Error("Blog URL is required (or upload a cache file)")

    # Create temp output files
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output_path = Path(f.name)

    cache_path = None
    if save_cache:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = Path(f.name)

    # Handle load from cache
    load_cache_path = None
    if load_cache_file is not None:
        load_cache_path = Path(load_cache_file)

    # Debug: Log what we're actually passing
    effective_model = model if model else None
    print(f"[DEBUG] API key provided: {bool(api_key)}, Model: {effective_model}")

    # Debug: Check for any API keys in environment
    api_env_vars = [k for k in os.environ.keys() if any(x in k.upper() for x in ['API', 'KEY', 'TOKEN', 'SECRET', 'OPENAI', 'ANTHROPIC', 'HF'])]
    print(f"[DEBUG] API-related env vars: {api_env_vars}")

    try:
        await _main_async(
            base_url=base_url or "https://placeholder.com",
            output_html=output_path,
            model=effective_model,
            n_key_points=n_key_points,
            n_wiki=n_wiki,
            sources=sources or [],
            n_external=n_external,
            content_weight=content_weight,
            metadata_weight=metadata_weight,
            embedding_model=embedding_model if embedding_model else None,
            embedding_weight=embedding_weight,
            top_k=top_k,
            max_concurrent=max_concurrent,
            max_depth=max_depth,
            filter_meta_pages=filter_meta_pages,
            min_word_count=min_word_count,
            min_similarity=min_similarity,
            use_llm_filter=use_llm_filter,
            skip_url_filter=skip_url_filter,
            skip_content_filter=skip_content_filter,
            stealth=stealth,
            headless=True,  # Always headless in web UI
            follow_external=follow_external,
            external_depth=int(external_depth),
            temperature=temperature,
            verbose=False,
            limit=int(limit) if limit else None,
            save_cache=cache_path,
            load_cache=load_cache_path,
            api_key=api_key,
        )
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")

    # Read HTML content for preview (embed in iframe via data URL)
    html_content = output_path.read_text()
    html_b64 = base64.b64encode(html_content.encode()).decode()
    iframe_html = f'<iframe src="data:text/html;base64,{html_b64}" style="width:100%;height:700px;border:1px solid #333;border-radius:8px;"></iframe>'

    return str(output_path), iframe_html, str(cache_path) if cache_path else None


def run_generate(*args):
    """Sync wrapper for async generate."""
    return asyncio.run(generate(*args))


# Custom CSS for VSCode-like tabs
custom_css = """
.tabs {
    border-bottom: none !important;
}
.tab-nav {
    border-bottom: 1px solid #333 !important;
    background: #1e1e1e !important;
}
.tab-nav button {
    background: #2d2d2d !important;
    border: none !important;
    border-radius: 4px 4px 0 0 !important;
    margin-right: 2px !important;
    padding: 8px 16px !important;
}
.tab-nav button.selected {
    background: #3c3c3c !important;
    border-bottom: 2px solid #007acc !important;
}
"""

with gr.Blocks(title="TikTokify", css=custom_css) as demo:
    gr.Markdown("# TikTokify\nTransform any blog into a TikTok-style swipeable viewer")

    # Output tabs at the top (VSCode-like)
    with gr.Tabs() as output_tabs:
        with gr.TabItem("Preview", id="preview"):
            html_preview = gr.HTML(label="Preview", elem_classes=["preview-container"])
        with gr.TabItem("Download HTML", id="html"):
            output_file = gr.File(label="Download HTML")
        with gr.TabItem("Download Cache", id="cache"):
            cache_file = gr.File(label="Download Cache JSON (for reuse)")

    gr.Markdown("---")

    # Input section below
    with gr.Row():
        with gr.Column(scale=2):
            base_url = gr.Textbox(
                label="Blog URL",
                placeholder="https://example.com/blog",
                info="The website to crawl and transform",
            )
            api_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-...",
                info="Required for LLM enrichment (key points, Wikipedia suggestions)",
            )
        with gr.Column(scale=1):
            model = gr.Dropdown(
                MODELS,
                label="LLM Model",
                value="gpt-4o-mini",
                info="Leave empty to skip LLM enrichment",
            )
            max_depth = gr.Slider(
                0, 5, value=1, step=1,
                label="Crawl Depth",
                info="0=seed only, 1=seed+links, 2=two levels",
            )

    with gr.Accordion("LLM Enrichment", open=False):
        with gr.Row():
            n_key_points = gr.Slider(1, 10, value=5, step=1, label="Key Points per Post")
            n_wiki = gr.Slider(0, 10, value=3, step=1, label="Wikipedia Suggestions")
            temperature = gr.Slider(0, 1, value=0.5, step=0.1, label="Temperature")

    with gr.Accordion("External Sources", open=False):
        sources = gr.CheckboxGroup(
            ["hackernews", "hn-frontpage", "links"],
            label="Sources to Fetch",
            info="Fetch related content from external sources",
        )
        n_external = gr.Slider(1, 10, value=3, step=1, label="Items per Source")

    with gr.Accordion("Recommendation Engine", open=False):
        with gr.Row():
            content_weight = gr.Slider(0, 1, value=0.6, step=0.1, label="Content Weight (TF-IDF)")
            metadata_weight = gr.Slider(0, 1, value=0.4, step=0.1, label="Metadata Weight (Tags)")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Recommendations per Post")
        with gr.Row():
            embedding_model = gr.Textbox(
                label="Embedding Model (optional)",
                placeholder="sentence-transformers/all-MiniLM-L6-v2",
                info="Leave empty to disable semantic embeddings",
            )
            embedding_weight = gr.Slider(0, 1, value=0.3, step=0.1, label="Embedding Weight")
        min_similarity = gr.Slider(
            0, 1, value=0.0, step=0.05,
            label="Min Similarity Threshold",
            info="Filter out low-similarity recommendations (0 = no filter)",
        )

    with gr.Accordion("Content Filtering", open=False):
        gr.Markdown("*Filter out junk pages before building recommendations*")
        with gr.Row():
            skip_url_filter = gr.Checkbox(value=False, label="Skip URL Filter")
            skip_content_filter = gr.Checkbox(value=False, label="Skip Content Filter")
            use_llm_filter = gr.Checkbox(value=False, label="Use LLM for Content Filter")
        min_word_count = gr.Slider(
            50, 500, value=100, step=10,
            label="Min Word Count",
            info="Pages with fewer words are filtered out",
        )

    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            max_concurrent = gr.Slider(1, 10, value=5, step=1, label="Max Concurrent Requests")
            filter_meta_pages = gr.Checkbox(value=True, label="Filter Meta Pages")
            stealth = gr.Checkbox(value=True, label="Stealth Mode (anti-detection)")
            limit = gr.Number(label="Post Limit (optional)", precision=0)
        with gr.Row():
            follow_external = gr.Checkbox(value=False, label="Follow External Links")
            external_depth = gr.Slider(0, 3, value=1, step=1, label="External Link Depth")
        with gr.Row():
            save_cache = gr.Checkbox(value=False, label="Save Cache (download JSON to reuse later)")
            load_cache_file = gr.File(
                label="Load from Cache (skip crawling)",
                file_types=[".json"],
                type="filepath",
            )

    generate_btn = gr.Button("Generate TikTokify Viewer", variant="primary", size="lg")

    generate_btn.click(
        fn=run_generate,
        inputs=[
            base_url, api_key, model, max_depth,
            n_key_points, n_wiki, temperature,
            sources, n_external,
            content_weight, metadata_weight,
            embedding_model, embedding_weight, top_k,
            max_concurrent, filter_meta_pages,
            min_word_count, min_similarity, use_llm_filter,
            skip_url_filter, skip_content_filter, stealth,
            follow_external, external_depth,
            limit, save_cache, load_cache_file,
        ],
        outputs=[output_file, html_preview, cache_file],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
