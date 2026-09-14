"""System prompt injection: replace {{skills_catalog}} with a skill listing."""

import re

from ..skills.registry import SkillRegistry

# Accepts both `{{skills_catalog}}` and `{{ skills_catalog }}` (spec uses both forms
# across its examples).
PLACEHOLDER_RE = re.compile(r"\{\{\s*skills_catalog\s*\}\}")


def serialize_skills_to_prompt(registry: SkillRegistry) -> str:
    """Convert a SkillRegistry into a simple catalog for the system prompt."""
    if not registry.skills:
        return "[No skills available]"

    lines = ["## Available Skills\n"]
    for skill in registry.get_all():
        lines.append(f"- **{skill.name}**: {skill.description}")

    return "\n".join(lines)


def inject_skills_into_prompt(system_prompt: str, skills_registry: SkillRegistry) -> str:
    """Replace `{{skills_catalog}}` in `system_prompt` with the rendered skill catalog."""
    if not PLACEHOLDER_RE.search(system_prompt):
        return system_prompt

    catalog = serialize_skills_to_prompt(skills_registry)
    return PLACEHOLDER_RE.sub(lambda _: catalog, system_prompt)
