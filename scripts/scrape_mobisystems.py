"""Scrape the MobiSystems (Zendesk) Help Center into Markdown files.

Uses the public Zendesk Help Center API (no auth needed): it returns
categories, sections and full article bodies, so we avoid fragile HTML
scraping. One .md file per article, organised by locale/category/section.

Output (gitignored):
  data/uploads/mobisystems/{locale}/{Category}/{Section}/{id}-{slug}.md

Run:
  python -m scripts.scrape_mobisystems
"""

import re
import time
from pathlib import Path

import requests
from markdownify import markdownify as html_to_md

BASE = "https://support.mobisystems.com/api/v2/help_center"
LOCALES = ["en-us", "bg"]
OUT_ROOT = Path("data/uploads/mobisystems")
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "kb-scraper/1.0"})


def _get_all(url: str, key: str) -> list[dict]:
    """Fetch every page of a Zendesk list endpoint and return items[key]."""
    items: list[dict] = []
    while url:
        resp = _SESSION.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items.extend(data.get(key, []))
        url = data.get("next_page")
        time.sleep(0.15)  # be gentle with the API
    return items


def _slug(text: str, max_len: int = 80) -> str:
    """Filesystem-safe slug from a title."""
    text = (text or "").strip()
    text = re.sub(r"[\\/:*?\"<>|]+", " ", text)  # unsafe chars
    text = re.sub(r"\s+", "-", text).strip("-.")
    return (text[:max_len] or "untitled").strip("-.")


def _clean_dir(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", " ", (name or "").strip())
    return re.sub(r"\s+", " ", name).strip() or "Uncategorized"


def scrape_locale(locale: str) -> int:
    print(f"\n=== {locale} ===")
    categories = _get_all(f"{BASE}/{locale}/categories.json?per_page=100", "categories")
    sections = _get_all(f"{BASE}/{locale}/sections.json?per_page=100", "sections")
    articles = _get_all(f"{BASE}/{locale}/articles.json?per_page=100", "articles")
    print(f"categories={len(categories)} sections={len(sections)} articles={len(articles)}")

    cat_name = {c["id"]: c["name"] for c in categories}
    sec = {s["id"]: s for s in sections}

    written = 0
    for art in articles:
        section = sec.get(art.get("section_id"))
        section_name = _clean_dir(section["name"]) if section else "General"
        category_name = (
            _clean_dir(cat_name.get(section["category_id"], "General"))
            if section
            else "General"
        )

        out_dir = OUT_ROOT / locale / category_name / section_name
        out_dir.mkdir(parents=True, exist_ok=True)

        title = art.get("title", "Untitled")
        # strip=["img"] drops images entirely: the agent is text-only, so image
        # URLs/alt text would only add noise to the embeddings.
        body_md = html_to_md(
            art.get("body") or "", heading_style="ATX", strip=["img"]
        ).strip()

        content = (
            f"# {title}\n\n"
            f"- Source: {art.get('html_url', '')}\n"
            f"- Category: {category_name}\n"
            f"- Section: {section_name}\n"
            f"- Locale: {locale}\n"
            f"- Updated: {art.get('updated_at', '')}\n\n"
            f"---\n\n"
            f"{body_md}\n"
        )

        fname = f"{art['id']}-{_slug(title)}.md"
        (out_dir / fname).write_text(content, encoding="utf-8")
        written += 1

    print(f"wrote {written} files for {locale}")
    return written


def main() -> None:
    total = 0
    for locale in LOCALES:
        total += scrape_locale(locale)
    print(f"\nDONE. Total files: {total}")
    print(f"Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
