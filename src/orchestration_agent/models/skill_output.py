"""`SkillOutput`: the orchestrator's own fixed envelope (spec section 7).

Produced once for a skill's dispatch, and again for the bound completion that
follows it where the skill declares `output` (section 6). The envelope shape
is fixed; only `data` varies by skill.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_VALID_STATUSES = ("success", "error", "partial")


@dataclass
class ErrorDetail:
    """`SkillOutput.error` / `AgentResponse.error` shape."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class SkillOutput:
    """One entry in `AgentResponse.messages`.

    - `type`: the skill's name, matching the catalogue entry.
    - `status`: "success" | "error" | "partial" — determines how `data`/`error`
      are read.
    - `data`: the payload — markdown on a dispatch entry, or the validated
      structured result on a bound-completion entry. Must be `None` on
      `error`.
    - `input`: `{"skill_name": ...}` on a dispatch entry; omitted (`None`) on
      a bound-completion entry, since it did not originate from a distinct
      call the model made.
    - `error`: present when `status` is `error` or `partial`.
    - `duration_ms`: wall-clock for this entry, including validation and any
      internal repair.
    """

    type: str
    status: str
    data: Any = None
    input: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetail] = None
    duration_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("SkillOutput.type must be a non-empty string")
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Invalid SkillOutput.status: {self.status!r}")
        if self.status == "error":
            if self.error is None:
                raise ValueError("SkillOutput with status='error' requires `error`")
            if self.data is not None:
                raise ValueError("SkillOutput with status='error' must have data=None")
        if self.status == "partial" and self.error is None:
            raise ValueError("SkillOutput with status='partial' requires `error`")


__all__ = ["SkillOutput", "ErrorDetail"]
