"""Serialization validation helpers reused by the CLI and notebooks."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import yaml


def _strip_fence(text: str, fmt: str) -> str:
    """Extract a fenced block if present (mirrors WIIAT_EHRCON_EVAL)."""

    fmt = fmt.lower()
    if fmt == "json":
        # ```json\n ... ```
        pattern = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
    elif fmt == "yaml":
        # ```yaml\n ... ``` or ```yml\n ... ```
        pattern = re.compile(r"```(?:yaml|yml)?\s*\n(.*?)```", re.DOTALL)
    elif fmt == "xml":
        # ```xml\n ... ```
        pattern = re.compile(r"```(?:xml)?\s*\n(.*?)```", re.DOTALL)
    else:
        return text

    match = pattern.search(text)
    return match.group(1) if match else text


def validate_serialization(payload: str, fmt: str) -> None:
    """Raise ``ValueError`` if ``payload`` cannot be parsed as ``fmt``.

    This mirrors the WIIAT_EHRCON evaluation helpers: it first attempts to parse
    the raw payload, then falls back to extracting a fenced code block (```json
    ... ``` / ```yaml ... ``` / ```xml ... ```).
    """

    fmt_normalized = fmt.lower()
    try:
        if fmt_normalized == "json":
            try:
                json.loads(payload)
                return
            except json.JSONDecodeError:
                json.loads(_strip_fence(payload, "json"))
        elif fmt_normalized == "yaml":
            try:
                yaml.safe_load(payload)
                return
            except yaml.YAMLError:
                yaml.safe_load(_strip_fence(payload, "yaml"))
        elif fmt_normalized == "xml":
            try:
                ET.fromstring(payload)
                return
            except ET.ParseError:
                ET.fromstring(_strip_fence(payload, "xml"))
        else:
            raise ValueError(f"Unsupported format for validation: {fmt}")
    except Exception as exc:  # pragma: no cover - surfacing raw error is useful
        raise ValueError(f"Failed to parse {fmt} payload") from exc
