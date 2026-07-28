import pytest

from kbqa.parsing import chunk_sections, parse_markdown


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
