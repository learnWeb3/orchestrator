"""Discover and parse SKILL.md files from a skills directory."""

from pathlib import Path
from typing import List

import yaml

from .models import SkillMetadata

REQUIRED_FIELDS = ("name", "description", "compatibility")


class SkillLoader:
    """Load and parse SKILL.md files from disk."""

    def load_skills_from_path(self, skills_path: str) -> List[SkillMetadata]:
        """Recursively discover `SKILL.md` files under `skills_path` and parse them.

        Files that fail to parse are skipped (logged to stderr via print, matching
        the spec's loader), so one malformed skill doesn't take down the others.
        """
        skills: List[SkillMetadata] = []
        skills_dir = Path(skills_path)

        if not skills_dir.exists():
            raise ValueError(f"Skills directory not found: {skills_path}")

        for md_file in sorted(skills_dir.rglob("SKILL.md")):
            try:
                metadata = self.parse_skill_file(md_file)
                skills.append(metadata)
                print(f"✓ Loaded skill: {metadata.name}")
            except Exception as e:  # noqa: BLE001 - intentionally broad, per spec
                print(f"✗ Failed to load {md_file}: {e}")

        return skills

    def parse_skill_file(self, file_path: Path) -> SkillMetadata:
        """Parse a single SKILL.md file into a SkillMetadata instance."""
        content = file_path.read_text(encoding="utf-8")

        if not content.startswith("---"):
            raise ValueError(f"SKILL.md must start with --- (found in {file_path})")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError(
                f"SKILL.md must have closing --- delimiter (found in {file_path})"
            )

        yaml_str, markdown_content = parts[1], parts[2].strip()

        try:
            yaml_data = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}") from e

        if not isinstance(yaml_data, dict):
            raise ValueError(f"YAML frontmatter must be a mapping in {file_path}")

        for field_name in REQUIRED_FIELDS:
            if field_name not in yaml_data:
                raise ValueError(f"Missing required field '{field_name}' in {file_path}")
            if not isinstance(yaml_data[field_name], str):
                raise ValueError(f"Field '{field_name}' must be a string in {file_path}")

        try:
            return SkillMetadata(
                name=yaml_data["name"],
                description=yaml_data["description"],
                compatibility=yaml_data["compatibility"],
                content=markdown_content,
            )
        except Exception as e:
            raise ValueError(f"Invalid skill metadata in {file_path}: {e}") from e
