"""In-memory registry of loaded skills."""

from typing import Dict, List, Optional

from .models import SkillMetadata


class SkillRegistry:
    """Registry for loaded skills, keyed by unique name."""

    def __init__(self) -> None:
        self.skills: Dict[str, SkillMetadata] = {}

    def register(self, metadata: SkillMetadata) -> None:
        if metadata.name in self.skills:
            raise ValueError(f"Skill '{metadata.name}' already registered")
        self.skills[metadata.name] = metadata

    def get_by_name(self, name: str) -> Optional[SkillMetadata]:
        return self.skills.get(name)

    def get_all(self) -> List[SkillMetadata]:
        return list(self.skills.values())

    def serialize_to_prompt(self) -> str:
        from ..utils.prompt import serialize_skills_to_prompt

        return serialize_skills_to_prompt(self)
