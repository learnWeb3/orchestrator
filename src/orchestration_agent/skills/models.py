"""SkillMetadata: minimal, APM-compliant skill representation."""

from pydantic import BaseModel, ConfigDict


class SkillMetadata(BaseModel):
    """Minimal skill metadata parsed from a SKILL.md file's frontmatter + body."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    compatibility: str
    content: str
