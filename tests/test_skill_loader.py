import pytest

from orchestration_agent.skills.loader import SkillLoader
from orchestration_agent.skills.registry import SkillRegistry

VALID_SKILL = """---
name: example
description: An example skill
compatibility: "python>=3.10, openai"
---

## Instructions

Do the thing.
"""

MISSING_FIELD_SKILL = """---
name: broken
description: Missing compatibility field
---

Body.
"""

NO_FRONTMATTER = "Just markdown, no frontmatter at all."


def test_parse_skill_file_success(tmp_path):
    skill_dir = tmp_path / "example"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(VALID_SKILL)

    loader = SkillLoader()
    metadata = loader.parse_skill_file(skill_dir / "SKILL.md")

    assert metadata.name == "example"
    assert metadata.description == "An example skill"
    assert metadata.compatibility == "python>=3.10, openai"
    assert "Do the thing." in metadata.content


def test_parse_skill_file_missing_field_raises(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text(MISSING_FIELD_SKILL)

    with pytest.raises(ValueError, match="compatibility"):
        SkillLoader().parse_skill_file(path)


def test_parse_skill_file_no_frontmatter_raises(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text(NO_FRONTMATTER)

    with pytest.raises(ValueError, match="must start with"):
        SkillLoader().parse_skill_file(path)


def test_load_skills_from_path_skips_bad_files(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "SKILL.md").write_text(VALID_SKILL)

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text(MISSING_FIELD_SKILL)

    skills = SkillLoader().load_skills_from_path(str(tmp_path))

    assert len(skills) == 1
    assert skills[0].name == "example"


def test_load_skills_from_missing_path_raises():
    with pytest.raises(ValueError, match="not found"):
        SkillLoader().load_skills_from_path("/no/such/path/at/all")


def test_registry_register_and_lookup():
    from orchestration_agent.skills.models import SkillMetadata

    registry = SkillRegistry()
    skill = SkillMetadata(name="a", description="A", compatibility="python", content="x")
    registry.register(skill)

    assert registry.get_by_name("a") is skill
    assert registry.get_by_name("missing") is None
    assert registry.get_all() == [skill]


def test_registry_duplicate_registration_raises():
    from orchestration_agent.skills.models import SkillMetadata

    registry = SkillRegistry()
    skill = SkillMetadata(name="a", description="A", compatibility="python", content="x")
    registry.register(skill)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(skill)


def test_serialize_to_prompt_lists_skills():
    from orchestration_agent.skills.models import SkillMetadata

    registry = SkillRegistry()
    registry.register(
        SkillMetadata(name="a", description="Does A", compatibility="python", content="x")
    )

    catalog = registry.serialize_to_prompt()
    assert "**a**: Does A" in catalog


def test_inject_skills_into_prompt_replaces_placeholder():
    from orchestration_agent.skills.models import SkillMetadata
    from orchestration_agent.utils.prompt import inject_skills_into_prompt

    registry = SkillRegistry()
    registry.register(
        SkillMetadata(name="a", description="Does A", compatibility="python", content="x")
    )

    result = inject_skills_into_prompt("Before {{ skills_catalog }} After", registry)
    assert "Before ## Available Skills" in result
    assert "**a**: Does A" in result
    assert "After" in result
