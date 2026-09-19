"""Tests for the standalone offline `output`-schema validator
(orchestration-agent-spec.md section 3) — never imported by the agent process."""

from pathlib import Path

import pytest

from orchestration_agent.scripts.validate_skill import (
    ValidationFailure,
    main,
    validate_skill,
)

REPO_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _write_skill(tmp_path: Path, frontmatter: str, body: str = "Body.") -> Path:
    skill_dir = tmp_path / "a-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n{body}\n")
    return skill_dir


BASE_FRONTMATTER = (
    'name: a-skill\ndescription: A skill\ncompatibility: "python>=3.10, openai"'
)


def test_no_output_declared_is_valid_and_unclassified(tmp_path):
    skill_dir = _write_skill(tmp_path, BASE_FRONTMATTER)

    metadata, reasons = validate_skill(str(skill_dir))

    assert metadata.output is None
    assert reasons == []


def test_strict_conformant_schema_classifies_strict(tmp_path):
    frontmatter = BASE_FRONTMATTER + (
        "\noutput:\n"
        "  type: object\n"
        "  properties:\n"
        "    summary: { type: string }\n"
        "  required: [summary]\n"
        "  additionalProperties: false\n"
    )
    skill_dir = _write_skill(tmp_path, frontmatter)

    metadata, reasons = validate_skill(str(skill_dir))

    assert metadata.output is not None
    assert reasons == []


def test_missing_additional_properties_false_classifies_permissive(tmp_path):
    frontmatter = BASE_FRONTMATTER + (
        "\noutput:\n"
        "  type: object\n"
        "  properties:\n"
        "    summary: { type: string }\n"
        "  required: [summary]\n"
    )
    skill_dir = _write_skill(tmp_path, frontmatter)

    metadata, reasons = validate_skill(str(skill_dir))

    assert any("additionalProperties" in r for r in reasons)


def test_optional_field_omitted_from_required_classifies_permissive(tmp_path):
    frontmatter = BASE_FRONTMATTER + (
        "\noutput:\n"
        "  type: object\n"
        "  properties:\n"
        "    summary: { type: string }\n"
        "    note: { type: string }\n"
        "  required: [summary]\n"
        "  additionalProperties: false\n"
    )
    skill_dir = _write_skill(tmp_path, frontmatter)

    metadata, reasons = validate_skill(str(skill_dir))

    assert any("required" in r and "note" in r for r in reasons)


def test_unsupported_keyword_classifies_permissive(tmp_path):
    frontmatter = BASE_FRONTMATTER + (
        "\noutput:\n"
        "  type: object\n"
        "  properties:\n"
        "    code: { type: string, pattern: '^[A-Z]+$' }\n"
        "  required: [code]\n"
        "  additionalProperties: false\n"
    )
    skill_dir = _write_skill(tmp_path, frontmatter)

    metadata, reasons = validate_skill(str(skill_dir))

    assert any("pattern" in r for r in reasons)


def test_root_type_not_object_classifies_permissive(tmp_path):
    frontmatter = BASE_FRONTMATTER + (
        "\noutput:\n"
        "  type: array\n"
        "  items: { type: string }\n"
    )
    skill_dir = _write_skill(tmp_path, frontmatter)

    metadata, reasons = validate_skill(str(skill_dir))

    assert any("root type" in r for r in reasons)


def test_boolean_output_schema_is_legal(tmp_path):
    frontmatter = BASE_FRONTMATTER + "\noutput: true"
    skill_dir = _write_skill(tmp_path, frontmatter)

    metadata, reasons = validate_skill(str(skill_dir))

    assert metadata.output is True
    assert reasons == []


def test_non_schema_output_fails(tmp_path):
    frontmatter = BASE_FRONTMATTER + '\noutput: "not a schema"'
    skill_dir = _write_skill(tmp_path, frontmatter)

    with pytest.raises(ValidationFailure):
        validate_skill(str(skill_dir))


def test_remote_ref_fails(tmp_path):
    frontmatter = BASE_FRONTMATTER + (
        "\noutput:\n"
        "  type: object\n"
        "  properties:\n"
        "    x: { $ref: 'https://example.com/schema.json' }\n"
        "  required: [x]\n"
        "  additionalProperties: false\n"
    )
    skill_dir = _write_skill(tmp_path, frontmatter)

    with pytest.raises(ValidationFailure, match="\\$ref"):
        validate_skill(str(skill_dir))


def test_local_defs_ref_is_supported(tmp_path):
    frontmatter = BASE_FRONTMATTER + (
        "\noutput:\n"
        "  type: object\n"
        "  properties:\n"
        "    x: { $ref: '#/$defs/Thing' }\n"
        "  required: [x]\n"
        "  additionalProperties: false\n"
        "  $defs:\n"
        "    Thing: { type: string }\n"
    )
    skill_dir = _write_skill(tmp_path, frontmatter)

    metadata, reasons = validate_skill(str(skill_dir))

    assert metadata.output is not None


def test_unsupported_dialect_fails(tmp_path):
    frontmatter = BASE_FRONTMATTER + (
        "\noutput:\n"
        "  $schema: 'http://json-schema.org/draft-07/schema#'\n"
        "  type: object\n"
    )
    skill_dir = _write_skill(tmp_path, frontmatter)

    with pytest.raises(ValidationFailure, match="dialect"):
        validate_skill(str(skill_dir))


def test_meta_schema_violation_fails(tmp_path):
    frontmatter = BASE_FRONTMATTER + (
        "\noutput:\n"
        "  type: not-a-real-type\n"
    )
    skill_dir = _write_skill(tmp_path, frontmatter)

    with pytest.raises(ValidationFailure):
        validate_skill(str(skill_dir))


def test_validate_skill_accepts_skill_md_path_directly(tmp_path):
    skill_dir = _write_skill(tmp_path, BASE_FRONTMATTER)

    metadata, reasons = validate_skill(str(skill_dir / "SKILL.md"))

    assert metadata.name == "a-skill"


def test_main_exits_zero_on_success(tmp_path, capsys):
    skill_dir = _write_skill(tmp_path, BASE_FRONTMATTER)

    exit_code = main([str(skill_dir)])

    assert exit_code == 0
    assert "OK a-skill" in capsys.readouterr().out


def test_main_exits_nonzero_and_reports_pointer_on_failure(tmp_path, capsys):
    frontmatter = BASE_FRONTMATTER + '\noutput: "not a schema"'
    skill_dir = _write_skill(tmp_path, frontmatter)

    exit_code = main([str(skill_dir)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "FAIL" in err
    assert "/output" in err


# -- Against the repo's real skills ------------------------------------------------


def test_real_text_summarizer_skill_is_strict():
    metadata, reasons = validate_skill(str(REPO_SKILLS_DIR / "text-summarizer"))

    assert metadata.output is not None
    assert reasons == []


def test_real_recipe_helper_skill_has_no_output():
    metadata, reasons = validate_skill(str(REPO_SKILLS_DIR / "recipe-helper"))

    assert metadata.output is None
    assert reasons == []
