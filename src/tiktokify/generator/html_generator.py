"""HTML generator for TikTok-style swipe UI."""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from tiktokify.models import RecommendationGraph


class HTMLGenerator:
    """Generate standalone HTML with embedded data and swipe UI."""

    def __init__(self, template_dir: Path | None = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True,
        )

    def generate(
        self,
        graph: RecommendationGraph,
        base_url: str,
        output_path: Path,
    ) -> None:
        """Generate HTML file with embedded recommendation data."""
        template = self.env.get_template("swipe.html.jinja2")

        # Prepare data for embedding
        graph_data = graph.to_json_for_embed()
        graph_json = json.dumps(graph_data, indent=2)

        # Sort posts by date for initial list
        sorted_posts = sorted(
            graph.posts.values(),
            key=lambda p: p.metadata.date,
            reverse=True,
        )
        post_slugs = [p.slug for p in sorted_posts]

        html = template.render(
            base_url=base_url.rstrip("/"),
            graph_json=graph_json,
            post_slugs_json=json.dumps(post_slugs),
            post_count=len(sorted_posts),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
