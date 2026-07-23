"""Generate the social-share (Open Graph) preview card for MobiHelp.

Dev-only helper: run locally to (re)generate the branded 1200x630 card that
messengers/social sites show when the app link is shared, plus a PNG favicon.
The generated PNGs are committed under ``app/ui/assets/`` and copied into
Streamlit's static root at Docker build time (see ``scripts/brand_streamlit.py``).

This script uses system fonts (macOS) and is NOT run inside the container.

Usage:
    python scripts/generate_og_card.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ASSETS = Path(__file__).resolve().parent.parent / "app" / "ui" / "assets"
_ICON = _ASSETS / "mobi_icon.jpg"
_OG_OUT = _ASSETS / "mobi_og.png"
_FAVICON_OUT = _ASSETS / "mobi_favicon.png"

# Brand palette (matches the Streamlit theme / mobisystems.com).
BG = (245, 248, 252)        # --bg
SURFACE = (255, 255, 255)   # card surface
HEADING = (36, 50, 120)     # --heading navy
MUTED = (91, 91, 91)        # --muted
PRIMARY = (3, 143, 243)     # --primary blue

# macOS system fonts (present on the dev machine, not in the container).
_FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
_FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _rounded(img: Image.Image, radius: int) -> Image.Image:
    """Return img with rounded corners (RGBA)."""
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [(0, 0), img.size], radius=radius, fill=255
    )
    img.putalpha(mask)
    return img


def generate_og_card() -> None:
    W, H = 1200, 630
    card = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(card)

    # Inner white panel with a blue accent bar at the bottom.
    margin = 56
    panel = [(margin, margin), (W - margin, H - margin)]
    draw.rounded_rectangle(panel, radius=36, fill=SURFACE)
    draw.rounded_rectangle(
        [(margin, H - margin - 16), (W - margin, H - margin)],
        radius=8,
        fill=PRIMARY,
    )

    # Brand icon (red "M"), rounded, top-left of the panel.
    icon = Image.open(_ICON).convert("RGB").resize((150, 150))
    icon = _rounded(icon, 34)
    card.paste(icon, (margin + 60, margin + 70), icon)

    title_font = ImageFont.truetype(_FONT_BOLD, 120)
    tag_font = ImageFont.truetype(_FONT_BOLD, 40)
    sub_font = ImageFont.truetype(_FONT_REG, 40)

    text_x = margin + 60 + 150 + 40
    # Small tagline above the title, aligned with the icon top.
    draw.text((text_x, margin + 78), "Mobi AI Agent", font=tag_font, fill=PRIMARY)
    draw.text((text_x, margin + 128), "Support", font=sub_font, fill=MUTED)

    # Big title.
    draw.text((margin + 60, margin + 260), "MobiHelp", font=title_font, fill=HEADING)

    # Description line.
    draw.text(
        (margin + 62, margin + 410),
        "Get instant help with MobiOffice, MobiPDF & MobiDrive.",
        font=sub_font,
        fill=MUTED,
    )

    card.save(_OG_OUT, "PNG")
    print(f"wrote {_OG_OUT} ({card.size[0]}x{card.size[1]})")


def generate_favicon() -> None:
    """PNG favicon from the brand icon (Streamlit's favicon is a PNG)."""
    icon = Image.open(_ICON).convert("RGB").resize((64, 64))
    icon.save(_FAVICON_OUT, "PNG")
    print(f"wrote {_FAVICON_OUT} (64x64)")


if __name__ == "__main__":
    generate_og_card()
    generate_favicon()
