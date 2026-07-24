import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from kbqa.models import DocumentUnit


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
NON_SLUG_RE = re.compile(r"[^\w\- ]+", flags=re.UNICODE)


def slugify_heading(heading: str) -> str:
    normalized = unicodedata.normalize("NFKC", heading).strip().lower()
    normalized = NON_SLUG_RE.sub("", normalized)
    normalized = re.sub(r"[\s\-]+", "-", normalized).strip("-")
    return normalized or "section"


def stable_unit_id(source_path: str, anchor: str, suffix: str = "") -> str:
    raw = f"{source_path}#{anchor}:{suffix}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def parse_markdown(path: Path, docs_root: Path) -> list[DocumentUnit]:
    source_path = path.relative_to(docs_root).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[tuple[str, list[str]]] = []
    heading = "Document"
    body: list[str] = []
    seen_heading = False

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            if body or seen_heading:
                sections.append((heading, body))
            heading = match.group(2).strip().rstrip("#").strip()
            body = []
            seen_heading = True
        else:
            body.append(line)
    if body or seen_heading or not sections:
        sections.append((heading, body))

    duplicate_counts: defaultdict[str, int] = defaultdict(int)
    units: list[DocumentUnit] = []
    for section_heading, section_lines in sections:
        base_anchor = slugify_heading(section_heading)
        duplicate_index = duplicate_counts[base_anchor]
        duplicate_counts[base_anchor] += 1
        anchor = base_anchor if duplicate_index == 0 else f"{base_anchor}-{duplicate_index}"
        text = "\n".join(section_lines).strip()
        units.append(
            DocumentUnit(
                id=stable_unit_id(source_path, anchor),
                source_path=source_path,
                heading=section_heading,
                anchor=anchor,
                citation=f"{source_path}#{anchor}",
                text=text,
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
    step = chunk_words - overlap
    chunks: list[DocumentUnit] = []
    for section in sections:
        words = section.text.split()
        windows = [words[start : start + chunk_words] for start in range(0, len(words), step)]
        if not windows:
            windows = [[]]
        for index, window in enumerate(windows):
            chunks.append(
                section.model_copy(
                    update={
                        "id": stable_unit_id(section.source_path, section.anchor, str(index)),
                        "text": " ".join(window),
                        "chunk_index": index,
                    }
                )
            )
            if (index + 1) * step + overlap >= len(words):
                break
    return chunks
