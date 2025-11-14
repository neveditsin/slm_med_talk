
"""Command-line helpers for the Open ASR tutorial toolkit."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

from . import AudioTranscriber, TranscriptionConfig

SUPPORTED_SUFFIXES = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma', '.aiff', '.aif', '.opus'}


def _iter_audio_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
        return
    for candidate in sorted(path.rglob('*')):
        if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            yield candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Transcribe audio files into text using the tutorial toolkit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=Path, default=Path('samples'), help='Audio file or directory')
    parser.add_argument('--output', '-o', type=Path, default=Path('transcripts'), help='Destination folder for transcripts')
    parser.add_argument('--model', '-m', default='openai/whisper-small', help='Hugging Face model identifier')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size for the pipeline')
    parser.add_argument('--chunk-length', type=float, default=None, help='Optional chunk length (seconds)')
    parser.add_argument('--stride-length', type=float, default=None, help='Optional stride overlap (seconds)')
    parser.add_argument('--overwrite', action='store_true', help='Rewrite transcript files even if they already exist')
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if not args.input.exists():
        raise FileNotFoundError(f'Input path not found: {args.input}')

    config = TranscriptionConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        chunk_length_s=args.chunk_length,
        stride_length_s=args.stride_length,
    )
    transcriber = AudioTranscriber(config)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(_iter_audio_files(args.input))
    if not audio_files:
        logging.warning('No supported audio files found in %s', args.input)
        return 0

    logging.info('Found %d audio file(s)', len(audio_files))

    for audio_file in audio_files:
        target = output_dir / f'{audio_file.stem}.txt'
        if target.exists() and not args.overwrite:
            logging.info('Skipping %s (already exists)', target.name)
            continue
        result = transcriber.transcribe_file(audio_file)
        result.save_text(target)
        metadata = {
            'audio_file': str(audio_file.resolve()),
            'transcript_file': str(target.resolve()),
            'model_name': args.model,
            'elapsed_seconds': round(result.elapsed_seconds, 2),
            'language': result.language,
        }
        target.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        logging.info('Saved transcript to %s', target)
    logging.info('Completed transcription job')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
