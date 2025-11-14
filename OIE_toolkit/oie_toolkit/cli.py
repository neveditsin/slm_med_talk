"""Command-line interface for LLM-powered clinical note conversion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

from . import (
    LocalGenerationConfig,
    LocalLLMExtractor,
    build_conversion_prompt,
    validate_serialization,
)
from .prompts import DEFAULT_SYSTEM_PROMPT


def _iter_note_files(path: Path) -> Iterable[Path]:
    """Yield single note file(s) from a path that can be a folder or file."""
    if path.is_file():
        if path.suffix.lower() == ".txt":
            yield path
        return
    yield from sorted(path.glob("*.txt"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw clinical notes to JSON/XML/YAML via LLM prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, default=Path("samples"), help="Note file or directory of .txt files")
    parser.add_argument("--output", "-o", type=Path, default=Path("outputs"), help="Directory for serialized artifacts")
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "xml", "yaml", "all"],
        default="all",
        help="Serialization format(s) to store",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip parseability checks on each artifact")
    parser.add_argument(
        "--model",
        "-m",
        default="microsoft/Phi-3.5-mini-instruct",
        help="Hugging Face model id (e.g., microsoft/Phi-3.5-mini-instruct, Qwen/Qwen3-14B, meta-llama/Meta-Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (omit for greedy decoding as in WIIAT_EHRCON)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling value (omit to mirror WIIAT_EHRCON settings)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate per file (8192 in WIIAT_EHRCON)",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", help="Torch dtype (e.g., bfloat16, float16, float32)")
    parser.add_argument("--device-map", default="auto", help="Device map passed to transformers (auto, cuda, cpu, etc.)")
    parser.add_argument("--system-prompt", type=str, default=None, help="Override the default system instructions")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat templates (falls back to plain prompts)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model/tokenizer",
    )
    parser.add_argument(
        "--schema-file",
        type=Path,
        default=None,
        help="Optional path to a JSON/YAML snippet describing the desired schema",
    )
    parser.add_argument(
        "--extra-requirements",
        type=Path,
        default=None,
        help="Optional text file with extra bullet points to append to the user prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first generated prompt instead of calling the LLM (useful for debugging)",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not args.input.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    files = list(_iter_note_files(args.input))
    if not files:
        logging.warning("No .txt files found in %s", args.input)
        return 0

    schema_description = args.schema_file.read_text(encoding="utf-8") if args.schema_file else None
    extra_requirements = args.extra_requirements.read_text(encoding="utf-8") if args.extra_requirements else None

    config = LocalGenerationConfig(
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        schema_description=schema_description,
        extra_requirements=extra_requirements,
        torch_dtype=None if args.torch_dtype.lower() in {"none", "null"} else args.torch_dtype,
        device_map=None if args.device_map.lower() in {"none", "null"} else args.device_map,
        trust_remote_code=args.trust_remote_code,
        use_chat_template=not args.no_chat_template,
    )

    extractor: LocalLLMExtractor | None = None
    if not args.dry_run:
        extractor = LocalLLMExtractor(config=config)

    def _formats() -> Sequence[str]:
        return ["json", "xml", "yaml"] if args.format == "all" else [args.format]

    for idx, path in enumerate(files):
        logging.info("Processing %s", path.name)
        note_text = path.read_text(encoding="utf-8")
        note_id = path.stem

        for fmt in _formats():
            if args.dry_run:
                prompt_preview = build_conversion_prompt(
                    note_text,
                    note_id=note_id,
                    output_format=fmt,
                    schema_description=schema_description,
                    extra_requirements=extra_requirements,
                )
                logging.info("Dry run prompt for %s (%s):\n%s", note_id, fmt, prompt_preview)
                continue

            assert extractor is not None  # for type-checkers
            serialized = extractor.convert_note(note_text, note_id=note_id, fmt=fmt)
            target = args.output / f"{note_id}.{fmt}"
            target.write_text(serialized + "\n", encoding="utf-8")
            logging.info("Wrote %s", target.relative_to(args.output.parent))
            if not args.no_validate:
                validate_serialization(serialized, fmt)
                logging.debug("Validated %s output for %s", fmt, note_id)
        if args.dry_run:
            logging.info("Dry run complete; exiting without calling the LLM.")
            break
    if not args.dry_run:
        logging.info("Completed %d note(s)", len(files))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
