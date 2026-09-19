"""SkillMetadata: minimal, APM-compliant skill representation."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class SkillMetadata(BaseModel):
    """Minimal skill metadata parsed from a SKILL.md file's frontmatter + body.

    `output`, where present, is the skill's declared output contract: a raw
    JSON Schema document (or `True`/`False`, the degenerate legal schema),
    read verbatim from the frontmatter and otherwise untouched (orchestration
    spec section 3). It is opaque to the loader/registry — no validation, no
    meta-schema check, no `$ref`/dialect check happens here; that is the
    standalone offline validation script's job, never the runtime's.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    compatibility: str
    content: str
    output: Optional[Any] = None
