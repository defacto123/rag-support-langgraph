"""Brand Streamlit's served HTML so shared links unfurl as MobiHelp.

Link-preview crawlers (Telegram, WhatsApp, Slack, …) read the RAW HTML the
server returns before any JavaScript runs. Streamlit ships a generic
``index.html`` (title "Streamlit", crown favicon, no Open Graph tags), which is
why the app previews as Streamlit. ``st.set_page_config`` only updates the tab
client-side, so it cannot fix link previews.

This script patches Streamlit's static ``index.html`` in place:
  - sets the <title>,
  - points the favicon at the Mobi icon,
  - injects Open Graph + Twitter card meta tags (incl. a 1200x630 image),
  - copies the brand images (favicon + OG card) into Streamlit's static root
    so they're served at ``/mobi_favicon.png`` and ``/mobi_og.png``.

Run once at Docker build time (see Dockerfile). Idempotent: re-running only
refreshes the injected block.

The absolute image URL requires the public origin; set PUBLIC_BASE_URL
(default https://mobisystems.help) at build time.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import streamlit

BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://mobisystems.help").rstrip("/")
TITLE = "MobiHelp – Mobi AI Agent"
DESCRIPTION = "Get instant help with MobiOffice, MobiPDF and MobiDrive."

_MARKER_START = "<!-- mobi-branding:start -->"
_MARKER_END = "<!-- mobi-branding:end -->"

_ASSETS = Path(__file__).resolve().parent.parent / "app" / "ui" / "assets"


def _static_dir() -> Path:
    return Path(streamlit.__file__).parent / "static"


def _meta_block() -> str:
    og_image = f"{BASE_URL}/mobi_og.png"
    return "\n".join(
        [
            _MARKER_START,
            '<meta property="og:type" content="website" />',
            '<meta property="og:site_name" content="MobiHelp" />',
            f'<meta property="og:title" content="{TITLE}" />',
            f'<meta property="og:description" content="{DESCRIPTION}" />',
            f'<meta property="og:url" content="{BASE_URL}/" />',
            f'<meta property="og:image" content="{og_image}" />',
            '<meta property="og:image:width" content="1200" />',
            '<meta property="og:image:height" content="630" />',
            '<meta name="twitter:card" content="summary_large_image" />',
            f'<meta name="twitter:title" content="{TITLE}" />',
            f'<meta name="twitter:description" content="{DESCRIPTION}" />',
            f'<meta name="twitter:image" content="{og_image}" />',
            f'<meta name="description" content="{DESCRIPTION}" />',
            _MARKER_END,
        ]
    )


def _copy_assets(static: Path) -> None:
    shutil.copyfile(_ASSETS / "mobi_favicon.png", static / "mobi_favicon.png")
    shutil.copyfile(_ASSETS / "mobi_og.png", static / "mobi_og.png")


def main() -> None:
    static = _static_dir()
    index = static / "index.html"
    html = index.read_text(encoding="utf-8")

    _copy_assets(static)

    # 1) Title.
    html = re.sub(r"<title>.*?</title>", f"<title>{TITLE}</title>", html, count=1)

    # 2) Favicon -> Mobi icon (served from the static root).
    html = re.sub(
        r'<link rel="shortcut icon"[^>]*/>',
        '<link rel="shortcut icon" href="./mobi_favicon.png" />',
        html,
        count=1,
    )

    # 3) Open Graph / Twitter tags: replace any prior block, else inject before </head>.
    block = _meta_block()
    if _MARKER_START in html:
        html = re.sub(
            re.escape(_MARKER_START) + r".*?" + re.escape(_MARKER_END),
            block,
            html,
            flags=re.DOTALL,
        )
    else:
        html = html.replace("</head>", f"{block}\n</head>", 1)

    index.write_text(html, encoding="utf-8")
    print(f"Branded Streamlit index.html at {index} (og:image base {BASE_URL})")


if __name__ == "__main__":
    main()
