import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def test_voicereel_doc_exists():
    path = os.path.join(PROJECT_ROOT, "docs", "en", "voicereel.md")
    assert os.path.exists(path), f"Missing documentation: {path}"


def test_mkdocs_nav_contains_voicereel():
    config_path = os.path.join(PROJECT_ROOT, "mkdocs.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "voicereel.md" in content, "voicereel.md not referenced in mkdocs nav"
