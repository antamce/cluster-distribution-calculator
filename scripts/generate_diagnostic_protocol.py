from __future__ import annotations

import re
import textwrap
from html.parser import HTMLParser
from pathlib import Path


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT_MARGIN = 44
RIGHT_MARGIN = 44
TOP_MARGIN = 48
BOTTOM_MARGIN = 48


def _ascii(text: str) -> str:
    replacements = {
        "→": "->",
        "–": "-",
        "—": "-",
        "×": "x",
        "µ": "um",
        "…": "...",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        " ": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    return text.encode("ascii", "replace").decode("ascii")


class ProtocolParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._block_kind: str | None = None
        self._text: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._row_is_header = False
        self._ordered_counters: list[int] = []
        self._list_prefix = "- "

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        attributes = dict(attrs)
        if tag == "div" and "pagebreak" in attributes.get("class", "").split():
            self.blocks.append(("pagebreak", ""))
        if tag == "ol":
            self._ordered_counters.append(1)
        elif tag == "ul":
            self._ordered_counters.append(0)
        elif tag in {"h1", "h2", "h3", "p"} and self._row is None:
            self._block_kind = tag
            self._text = []
        elif tag == "li":
            self._block_kind = "li"
            self._text = []
            if self._ordered_counters and self._ordered_counters[-1] > 0:
                self._list_prefix = f"{self._ordered_counters[-1]}. "
                self._ordered_counters[-1] += 1
            else:
                self._list_prefix = "- "
        elif tag == "tr":
            self._row = []
            self._row_is_header = False
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []
            self._row_is_header = self._row_is_header or tag == "th"
        elif tag == "br":
            if self._cell is not None:
                self._cell.append(" ")
            elif self._block_kind is not None:
                self._text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"ol", "ul"}:
            if self._ordered_counters:
                self._ordered_counters.pop()
            return
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append(_ascii("".join(self._cell)))
            self._cell = None
            return
        if tag == "tr" and self._row is not None:
            kind = "table_header" if self._row_is_header else "table_row"
            self.blocks.append((kind, " | ".join(self._row)))
            self._row = None
            self._row_is_header = False
            return
        if tag == self._block_kind and self._block_kind is not None:
            text = _ascii("".join(self._text))
            if text:
                prefix = self._list_prefix if self._block_kind == "li" else ""
                self.blocks.append((self._block_kind, prefix + text))
            self._block_kind = None
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        elif self._block_kind is not None:
            self._text.append(data)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_command(
    text: str,
    *,
    x: float,
    y: float,
    font: str,
    size: float,
    color: tuple[float, float, float],
) -> str:
    red, green, blue = color
    return (
        f"BT /{font} {size:.1f} Tf {red:.3f} {green:.3f} {blue:.3f} rg "
        f"1 0 0 1 {x:.1f} {y:.1f} Tm ({_escape_pdf_text(text)}) Tj ET"
    )


def _layout(blocks: list[tuple[str, str]]) -> list[list[str]]:
    pages: list[list[str]] = [[]]
    y = PAGE_HEIGHT - TOP_MARGIN
    usable_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

    styles = {
        "h1": ("F2", 21.0, 29.0, 8.0, (0.090, 0.212, 0.365), 0),
        "h2": ("F2", 14.0, 20.0, 10.0, (0.137, 0.310, 0.490), 0),
        "h3": ("F2", 11.5, 16.0, 7.0, (0.192, 0.373, 0.553), 0),
        "p": ("F1", 9.5, 13.0, 5.0, (0.080, 0.080, 0.080), 0),
        "li": ("F1", 9.3, 12.5, 3.0, (0.080, 0.080, 0.080), 12),
        "table_header": ("F2", 8.2, 11.0, 4.0, (0.060, 0.150, 0.250), 0),
        "table_row": ("F1", 7.8, 10.5, 3.0, (0.100, 0.100, 0.100), 0),
    }

    def new_page() -> None:
        nonlocal y
        if pages[-1]:
            pages.append([])
        y = PAGE_HEIGHT - TOP_MARGIN

    for kind, text in blocks:
        if kind == "pagebreak":
            new_page()
            continue
        font, size, line_height, spacing, color, indent = styles.get(
            kind, styles["p"]
        )
        width = usable_width - indent
        approximate_character_width = max(
            3.5, size * (0.56 if font == "F2" else 0.50)
        )
        maximum_characters = max(18, int(width / approximate_character_width))
        lines = textwrap.wrap(
            text,
            width=maximum_characters,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        required = spacing + len(lines) * line_height
        if y - required < BOTTOM_MARGIN:
            new_page()
        y -= spacing
        for line_index, line in enumerate(lines):
            x = LEFT_MARGIN + indent
            if kind == "li" and line_index > 0:
                x += 10
            pages[-1].append(
                _text_command(
                    line,
                    x=x,
                    y=y,
                    font=font,
                    size=size,
                    color=color,
                )
            )
            y -= line_height
        if kind in {"table_header", "table_row"}:
            y -= 1.5

    total_pages = len(pages)
    for page_number, commands in enumerate(pages, start=1):
        commands.append(
            _text_command(
                f"Synpo Performance Diagnostic Protocol - page {page_number} of {total_pages}",
                x=LEFT_MARGIN,
                y=24,
                font="F1",
                size=7.5,
                color=(0.35, 0.35, 0.35),
            )
        )
    return pages


def _write_pdf(destination: Path, pages: list[list[str]]) -> None:
    objects: list[bytes] = []

    def add_object(data: str | bytes) -> int:
        objects.append(data.encode("ascii") if isinstance(data, str) else data)
        return len(objects)

    catalog_id = add_object("<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(b"")
    font_regular_id = add_object(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    font_bold_id = add_object(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    )
    page_ids: list[int] = []
    for commands in pages:
        content = ("\n".join(commands) + "\n").encode("ascii")
        content_id = add_object(
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"endstream"
        )
        page_id = add_object(
            "<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )
        page_ids.append(page_id)
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    ).encode("ascii")
    assert catalog_id == 1 and pages_id == 2

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, data in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(data)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    destination.write_bytes(output)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "docs" / "diagnostic_protocol.html"
    destination = (
        root / "src" / "synpo" / "assets" / "Synpo_Diagnostic_Protocol.pdf"
    )
    parser = ProtocolParser()
    parser.feed(source.read_text(encoding="utf-8"))
    pages = _layout(parser.blocks)
    _write_pdf(destination, pages)
    print(
        f"{destination} ({len(pages)} pages; standard PDF Helvetica fonts; ASCII text)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
