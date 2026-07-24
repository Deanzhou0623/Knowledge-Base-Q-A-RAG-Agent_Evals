from kbqa.parsing import parse_markdown


def test_parser_preserves_empty_and_duplicate_heading_sections(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "guide.md"
    path.write_text("# Returns\n# Returns\nSecond section", encoding="utf-8")

    sections = parse_markdown(path, docs)

    assert [section.anchor for section in sections] == ["returns", "returns-1"]
    assert sections[0].text == ""
    assert sections[1].citation == "guide.md#returns-1"
