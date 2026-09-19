"""Offline validation of a SKILL.md's `output` schema (orchestration-agent-spec.md,
section 3, "Offline validation").

This is the *only* place a malformed `output` schema is caught. It is a
standalone script, run separately from the agent process — e.g. in CI, against
every skill before deployment — never during `Agent.__init__` or
`SkillLoader.load_skills_from_path`, which treat `output` as opaque.

Usage:
    python -m orchestration_agent.scripts.validate_skill /abs/path/to/a/skill
    python -m orchestration_agent.scripts.validate_skill /abs/path/to/a/skill/SKILL.md
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jsonschema

from ..skills.loader import SkillLoader
from ..skills.models import SkillMetadata

_DEFAULT_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_SUPPORTED_DIALECTS: Dict[str, Any] = {
    _DEFAULT_DIALECT: jsonschema.Draft202012Validator,
}

# Constraints for `strict` (spec section 4): numeric/string constraints and
# structural keywords commonly rejected or ignored by native structured-output
# compilers.
_UNSUPPORTED_STRICT_KEYWORDS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "if",
    "then",
    "else",
    "not",
    "patternProperties",
    "dependentSchemas",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
}


class ValidationFailure(Exception):
    """A hard failure: parse error, non-schema `output`, meta-schema violation,
    or an unsupported `$ref`/dialect (spec section 3, item 5)."""

    def __init__(self, pointer: str, keyword: str, message: str) -> None:
        self.pointer = pointer
        self.keyword = keyword
        self.message = message
        super().__init__(message)


def _resolve_skill_file(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "SKILL.md"
        if not candidate.exists():
            raise ValidationFailure(str(path), "path", f"No SKILL.md found under {path}")
        return candidate
    return path


def _check_refs(node: Any, pointer: str = "") -> None:
    """Reject remote (`http://`, `https://`, file paths) and cross-skill `$ref`s.
    Only local `#/$defs/...`-style references are supported (spec section 3)."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            raise ValidationFailure(
                pointer, "$ref", f"Remote or cross-skill $ref is not supported: {ref!r}"
            )
        for key, value in node.items():
            _check_refs(value, f"{pointer}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _check_refs(value, f"{pointer}/{index}")


def _classify_strict(schema: Any, pointer: str = "/output", root: bool = True) -> List[str]:
    """Return human-readable reasons `schema` falls outside the strict subset
    (spec section 4). An empty list means the schema is `strict`."""
    reasons: List[str] = []

    if isinstance(schema, bool):
        if root:
            reasons.append(f"{pointer}: root type must be 'object' (found boolean schema)")
        return reasons

    if not isinstance(schema, dict):
        reasons.append(f"{pointer}: expected a schema object (found {type(schema).__name__})")
        return reasons

    schema_type = schema.get("type")
    if root and schema_type != "object":
        reasons.append(f"{pointer}: root type must be 'object' (found {schema_type!r})")

    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            reasons.append(f"{pointer}: additionalProperties must be false at every depth")
        required = set(schema.get("required", []))
        missing = sorted(set(properties.keys()) - required)
        if missing:
            reasons.append(f"{pointer}: required must list every property (missing: {missing})")
        for key, value in properties.items():
            reasons.extend(_classify_strict(value, f"{pointer}/properties/{key}", root=False))

    if "items" in schema:
        reasons.extend(_classify_strict(schema["items"], f"{pointer}/items", root=False))

    for defs_key in ("$defs", "definitions"):
        for name, value in schema.get(defs_key, {}).items():
            reasons.extend(_classify_strict(value, f"{pointer}/{defs_key}/{name}", root=False))

    for combinator in ("anyOf", "allOf"):
        for index, value in enumerate(schema.get(combinator, [])):
            reasons.extend(_classify_strict(value, f"{pointer}/{combinator}/{index}", root=False))

    if "oneOf" in schema:
        reasons.append(f"{pointer}: oneOf support varies by provider; anyOf is safer")
        for index, value in enumerate(schema["oneOf"]):
            reasons.extend(_classify_strict(value, f"{pointer}/oneOf/{index}", root=False))

    for keyword in sorted(_UNSUPPORTED_STRICT_KEYWORDS & schema.keys()):
        reasons.append(f"{pointer}: unsupported keyword in strict mode: {keyword!r}")

    return reasons


def validate_skill(skill_path: str) -> Tuple[SkillMetadata, List[str]]:
    """Validate one skill's `output` schema.

    Returns `(metadata, permissive_reasons)` — an empty `permissive_reasons`
    means the schema is `strict` (or absent). Raises `ValidationFailure` on
    any hard failure.
    """
    path = Path(skill_path).resolve()
    file_path = _resolve_skill_file(path)

    try:
        metadata = SkillLoader().parse_skill_file(file_path)
    except ValueError as e:
        raise ValidationFailure(str(file_path), "frontmatter", str(e)) from e

    output: Optional[Union[Dict[str, Any], bool]] = metadata.output
    if output is None:
        return metadata, []

    if not isinstance(output, (dict, bool)):
        raise ValidationFailure(
            "/output", "output", "`output` must be a JSON Schema object or boolean"
        )

    if isinstance(output, bool):
        return metadata, []

    dialect = output.get("$schema", _DEFAULT_DIALECT)
    validator_cls = _SUPPORTED_DIALECTS.get(dialect)
    if validator_cls is None:
        raise ValidationFailure(
            "/output/$schema", "$schema", f"Unsupported dialect: {dialect!r}"
        )

    try:
        validator_cls.check_schema(output)
    except jsonschema.exceptions.SchemaError as e:
        pointer = "/output/" + "/".join(str(part) for part in e.path)
        raise ValidationFailure(pointer.rstrip("/"), "meta-schema", e.message) from e

    _check_refs(output, "/output")

    return metadata, _classify_strict(output)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "skill_path", help="Absolute path to a skill directory, or directly to its SKILL.md file"
    )
    args = parser.parse_args(argv)

    try:
        metadata, permissive_reasons = validate_skill(args.skill_path)
    except ValidationFailure as e:
        print(f"FAIL {args.skill_path}", file=sys.stderr)
        print(f"  at: {e.pointer}", file=sys.stderr)
        print(f"  keyword: {e.keyword}", file=sys.stderr)
        print(f"  {e.message}", file=sys.stderr)
        return 1

    if metadata.output is None:
        print(f"OK {metadata.name}: no output declared (dispatch-only)")
        return 0

    classification = "permissive" if permissive_reasons else "strict"
    print(f"OK {metadata.name}: output={classification}")
    for reason in permissive_reasons:
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
