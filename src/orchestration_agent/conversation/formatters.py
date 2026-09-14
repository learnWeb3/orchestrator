"""Default conversation formatter for LLM-facing serialization."""

from typing import Any, Dict, List


def default_formatter(messages: List[Dict[str, Any]]) -> str:
    """Produce a clean, readable "Role: content [Used: skill]" transcript."""
    if not messages:
        return ""

    lines = []
    for msg in messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]

        if msg.get("skills_invoked"):
            skill_names = ", ".join(s["name"] for s in msg["skills_invoked"])
            content = f"{content} [Used: {skill_names}]"

        lines.append(f"{role_label}: {content}")

    return "\n".join(lines)
