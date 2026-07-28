import pytest

from kbqa.parsing import chunk_sections, parse_markdown


from kbqa.parsing import load_sections, parse_markdown, stable_unit_id


def test_parser_preserves_empty_and_duplicate_heading_sections(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text("# Returns\n# Returns\nSecond section", encoding="utf-8")

    sections = parse_markdown(path, docs)

    assert [section.anchor for section in sections] == ["returns", "returns-1"]
    assert sections[0].text == ""
    assert sections[1].citation == "guide.md#returns-1"


def test_vector_chunks_are_stable_heading_scoped_and_overlap(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "nested" / "guide.md"
    path.parent.mkdir()
    path.write_text(
        "# Empty\n\n## Shipping Rules\none two three four five six seven",
        encoding="utf-8",
    )
    sections = parse_markdown(path, docs)

    first = chunk_sections(sections, chunk_words=4, overlap=1)
    second = chunk_sections(sections, chunk_words=4, overlap=1)

    assert [chunk.text for chunk in first] == [
        "one two three four",
        "four five six seven",
    ]
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.chunk_index for chunk in first] == [0, 1]
    assert {chunk.citation for chunk in first} == {
        "nested/guide.md#shipping-rules"
    }
    assert all(chunk.heading == "Shipping Rules" for chunk in first)


@pytest.mark.parametrize(
    ("chunk_words", "overlap"),
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)],
)
def test_vector_chunking_rejects_invalid_configuration(
    chunk_words, overlap, tmp_path
):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text("# Guide\ncontent", encoding="utf-8")

    with pytest.raises(ValueError):
        chunk_sections(
            parse_markdown(path, docs),
            chunk_words=chunk_words,
            overlap=overlap,
        )


def test_vector_chunk_ids_differ_across_chunking_configurations(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text("# Guide\n" + " ".join(f"w{n}" for n in range(40)), encoding="utf-8")
    sections = parse_markdown(path, docs)

    coarse = chunk_sections(sections, chunk_words=30, overlap=5)
    fine = chunk_sections(sections, chunk_words=10, overlap=2)

    # Same section, same ordinal, different text: the IDs must not collide or a
    # comparison report joining runs on unit ID equates unlike chunks.
    assert coarse[0].text != fine[0].text
    assert coarse[0].id != fine[0].id
    assert [c.id for c in coarse] == [c.id for c in chunk_sections(sections, 30, 5)]


def test_parser_preserves_preamble_hierarchy_raw_markdown_and_empty_sections(
    tmp_path,
):
    docs = tmp_path / "docs"
    path = docs / "nested" / "guide.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Intro with **Markdown**.\n\n"
        "# Returns\n"
        "## Timing\n"
        "Seven to eleven days.\n"
        "## Timing\n"
        "### Exceptions\n"
        "# Empty\n",
        encoding="utf-8",
    )

    sections = parse_markdown(path, docs)

    assert [section.heading for section in sections] == [
        "Document",
        "Returns",
        "Timing",
        "Timing",
        "Exceptions",
        "Empty",
    ]
    assert [section.anchor for section in sections] == [
        "_preamble",
        "returns",
        "timing",
        "timing-1",
        "exceptions",
        "empty",
    ]
    assert sections[0].source_path == "nested/guide.md"
    assert sections[0].text == "Intro with **Markdown**."
    assert sections[1].text == ""
    assert sections[2].heading_level == 2
    assert sections[2].heading_path == ["Returns", "Timing"]
    assert sections[4].heading_path == ["Returns", "Timing", "Exceptions"]
    assert sections[5].heading_path == ["Empty"]
    assert sections[5].text == ""


def test_parser_ignores_blank_preamble_and_supports_setext_headings(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text(
        "\n\nTitle\n=====\nBody\n\nSubtitle\n--------\nMore body\n",
        encoding="utf-8",
    )

    sections = parse_markdown(path, docs)

    assert [(item.heading, item.heading_level) for item in sections] == [
        ("Title", 1),
        ("Subtitle", 2),
    ]
    assert sections[1].heading_path == ["Title", "Subtitle"]


def test_parser_uses_github_style_anchors_and_stable_ids(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text(
        "# Shipping & Returns!\nFirst\n# Shipping & Returns!\nSecond",
        encoding="utf-8",
    )

    first = parse_markdown(path, docs)
    second = parse_markdown(path, docs)

    assert [item.anchor for item in first] == [
        "shipping-returns",
        "shipping-returns-1",
    ]
    assert [item.id for item in first] == [item.id for item in second]
    assert first[0].id == stable_unit_id("guide.md", "shipping-returns")


def test_parser_does_not_split_headings_inside_fenced_code(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text(
        "# Examples\n"
        "```markdown\n"
        "# This is example data\n"
        "```\n"
        "After the fence.\n",
        encoding="utf-8",
    )

    sections = parse_markdown(path, docs)

    assert len(sections) == 1
    assert sections[0].heading == "Examples"
    assert "# This is example data" in sections[0].text


def test_load_sections_discovers_nested_files_in_deterministic_order(tmp_path):
    docs = tmp_path / "docs"
    (docs / "z").mkdir(parents=True)
    (docs / "a.md").write_text("# A\nA", encoding="utf-8")
    (docs / "z" / "b.md").write_text("# B\nB", encoding="utf-8")

    first, file_count = load_sections(docs)
    second, _ = load_sections(docs)

    assert file_count == 2
    assert [item.source_path for item in first] == ["a.md", "z/b.md"]
    assert [item.id for item in first] == [item.id for item in second]


def test_a_real_heading_keeps_its_anchor_when_a_preamble_exists(tmp_path):
    # The preamble previously took the plain "document" slug, pushing a real
    # "## Document" heading to "document-1" and making an Oracle citation of
    # file.md#document point at the preamble instead of the heading.
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text(
        "Intro before any heading.\n\n## Document\nThe real section.\n",
        encoding="utf-8",
    )

    sections = parse_markdown(path, docs)

    by_heading = {section.heading: section.anchor for section in sections}
    assert by_heading["Document"] == "document"
    assert by_heading["Document"] != "_preamble"
    assert sections[0].anchor == "_preamble"


def test_preamble_content_is_not_chunked_for_retrieval(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text(
        "Preamble words that are not citable.\n\n# Real\nReal body text.\n",
        encoding="utf-8",
    )

    chunks = chunk_sections(parse_markdown(path, docs), chunk_words=10, overlap=2)

    assert chunks
    assert all(chunk.anchor != "_preamble" for chunk in chunks)
