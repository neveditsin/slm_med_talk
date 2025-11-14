"""Prompt helpers for converting clinical notes via local LLMs.

These prompts intentionally mirror the open information-extraction style used
in the WIIAT_EHRCON notebook: no fixed schema, just "extract all data" into
JSON/YAML/XML with fenced output for easy regex extraction.
"""

from __future__ import annotations

from textwrap import dedent

SUPPORTED_FORMATS = ("json", "yaml", "xml")

# Kept for configuration symmetry; the default user prompt does not enforce
# a schema and simply asks for open IE, as in WIIAT_EHRCON.
DEFAULT_SYSTEM_PROMPT = ""


def _mk_prompt_extract_data_json(doc: str) -> str:
    """Match mk_prompt_extract_data from WIIAT_EHRCON.ipynb (JSON)."""
    return (
        "    Given the following document: \\n "
        + doc
        + """.
        
        Extract all data in JSON format.
    Make sure that the JSON document is valid, provide reasonably detailed names for fields.

    If some data has text in free form without specific explicit fields, keep it as unstructured text. 
    Make a proper fence for JSON so that it can be extraxted from response with regular expression.
    """
    )


def _mk_prompt_extract_data_yaml(doc: str) -> str:
    """Match mk_prompt_extract_data_yml from WIIAT_EHRCON.ipynb (YAML)."""
    return (
        "    Given the following document: \\n "
        + doc
        + """.
        
        Extract all data in YAML format.
    Make sure that the YAML document is valid, provide reasonably detailed names for fields.

    If some data has text in free form without specific explicit fields, keep it as unstructured text. 
    Make a proper fence for YAML so that it can be extraxted from response with regular expression.
    """
    )


def _mk_prompt_extract_data_xml(doc: str) -> str:
    """Match mk_prompt_extract_data_xml from WIIAT_EHRCON.ipynb (XML)."""
    return (
        "    Given the following document: \\n "
        + doc
        + """.
        
        Extract all data in XML format.
    Make sure that the XML document is valid, provide reasonably detailed names for fields.

    If some data has text in free form without specific explicit fields, keep it as unstructured text. 
    Make a proper fence for XML so that it can be extraxted from response with regular expression.
    """
    )


def build_conversion_prompt(
    note_text: str,
    *,
    note_id: str,  # kept for API compatibility; not used in the prompt
    output_format: str,
    schema_description: str | None = None,  # unused, preserved for CLI compatibility
    extra_requirements: str | None = None,
) -> str:
    """Return an open-IE prompt in the style of WIIAT_EHRCON (no fixed schema)."""

    fmt = output_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{output_format}'. Choose from {SUPPORTED_FORMATS}.")

    doc = note_text
    if fmt == "json":
        prompt = _mk_prompt_extract_data_json(doc)
    elif fmt == "yaml":
        prompt = _mk_prompt_extract_data_yaml(doc)
    else:
        prompt = _mk_prompt_extract_data_xml(doc)

    if extra_requirements:
        prompt = prompt.rstrip() + "\n\n" + dedent(extra_requirements).strip() + "\n"

    return prompt


__all__ = ["SUPPORTED_FORMATS", "DEFAULT_SYSTEM_PROMPT", "build_conversion_prompt"]

