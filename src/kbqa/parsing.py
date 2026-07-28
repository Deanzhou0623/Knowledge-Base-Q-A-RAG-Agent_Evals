import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from kbqa.models import DocumentUnit


PARSER_VERSION = "markdown-heading-sections-v2"
ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)[ \t]*|[ \t]*)$")
SETEXT_HEADING_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class _ParsedSection:
    heading: str
    heading_level: int
    heading_path: list[str]
    lines: list[str]


def slugify_heading(heading: str) -> str:
    normalized = unicodedata.normalize("NFKC", heading).strip().lower()
    normalized = "".join(
        character
        for character in normalized
        if character in {"-", "_"}
        or character.isspace()
        or unicodedata.category(character)[0] in {"L", "M", "N"}
    )
    normalized = re.sub(r"\s+", "-", normalized).strip("-")
    return normalized or "section"


def stable_unit_id(source_path: str, anchor: str, suffix: str = "") -> str:
    raw = f"{source_path}#{anchor}:{suffix}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _atx_heading(line: str) -> tuple[int, str] | None:
    match = ATX_HEADING_RE.match(line)
    if match is None:
        return None
    heading = (match.group(2) or "").strip()
    heading = re.sub(r"[ \t]+#+[ \t]*$", "", heading).strip()
    return len(match.group(1)), heading or "Section"


def _has_content(lines: list[str]) -> bool:
    return any(line.strip() for line in lines)


def parse_markdown(path: Path, docs_root: Path) -> list[DocumentUnit]:
    source_path = path.relative_to(docs_root).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[_ParsedSection] = []
    heading = "Document"
    heading_level = 0
    heading_path = ["Document"]
    hierarchy: list[tuple[int, str]] = []
    body: list[str] = []
    seen_heading = False
    active_fence: tuple[str, int] | None = None

    for line in lines:
        fence_match = FENCE_RE.match(line)
        if active_fence is not None:
            body.append(line)
            fence_character, fence_length = active_fence
            if re.match(
                rf"^ {{0,3}}{re.escape(fence_character)}{{{fence_length},}}[ \t]*$",
                line,
            ):
                active_fence = None
            continue
        if fence_match is not None:
            marker = fence_match.group(1)
            active_fence = (marker[0], len(marker))
            body.append(line)
            continue

        parsed_heading = _atx_heading(line)
        setext_match = SETEXT_HEADING_RE.match(line)
        if parsed_heading is None and setext_match is not None and body and body[-1].strip():
            parsed_heading = (1 if setext_match.group(1).startswith("=") else 2, body.pop().strip())

        if parsed_heading is not None:
            if seen_heading or _has_content(body):
                sections.append(
                    _ParsedSection(heading, heading_level, heading_path.copy(), body)
                )
            heading_level, heading = parsed_heading
            while hierarchy and hierarchy[-1][0] >= heading_level:
                hierarchy.pop()
            hierarchy.append((heading_level, heading))
            heading_path = [title for _, title in hierarchy]
            body = []
            seen_heading = True
        else:
            body.append(line)
    if seen_heading or _has_content(body):
        sections.append(_ParsedSection(heading, heading_level, heading_path.copy(), body))

    duplicate_counts: defaultdict[str, int] = defaultdict(int)
    units: list[DocumentUnit] = []
    for section in sections:
        base_anchor = slugify_heading(section.heading)
        duplicate_index = duplicate_counts[base_anchor]
        duplicate_counts[base_anchor] += 1
        anchor = base_anchor if duplicate_index == 0 else f"{base_anchor}-{duplicate_index}"
        text = "\n".join(section.lines).strip()
        units.append(
            DocumentUnit(
                id=stable_unit_id(source_path, anchor),
                source_path=source_path,
                heading=section.heading,
                anchor=anchor,
                citation=f"{source_path}#{anchor}",
                text=text,
                heading_level=section.heading_level,
                heading_path=section.heading_path,
            )
        )
    return units


def load_sections(docs_path: Path) -> tuple[list[DocumentUnit], int]:
    files = sorted(docs_path.rglob("*.md"))
    units = [unit for path in files for unit in parse_markdown(path, docs_path)]
    return units, len(files)


def chunk_sections(
    sections: list[DocumentUnit], chunk_words: int, overlap: int
) -> list[DocumentUnit]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must be non-negative and smaller than chunk_words")

    step = chunk_words - overlap
    chunks: list[DocumentUnit] = []
    for section in sections:
        words = section.text.split()
        if not words:
            continue
        windows = [
            words[start : start + chunk_words] for start in range(0, len(words), step)
        ]
        for index, window in enumerate(windows):
            chunks.append(
                section.model_copy(
                    update={
                        # The chunking configuration is part of the chunk
                        # identity: the same ordinal under a different
                        # chunk_words/overlap covers different text, and
                        # evaluation artifacts join runs on this ID.
                        "id": stable_unit_id(
                            section.source_path,
                            section.anchor,
                            f"{chunk_words}:{overlap}:{index}",
                        ),
                        "text": " ".join(window),
                        "chunk_index": index,
                    }
                )
            )
            if (index + 1) * step + overlap >= len(words):
                break
    return chunks
