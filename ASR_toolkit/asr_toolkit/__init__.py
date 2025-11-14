"""
Lightweight ASR toolkit primitives.

This module exposes the :class:`AudioTranscriber` and supporting dataclasses
so they can be reused by both the CLI scripts and instructional notebooks.
"""

from .transcriber import AudioTranscriber, TranscriptionConfig, TranscriptionResult, list_audio_files

__all__ = [
    "AudioTranscriber",
    "TranscriptionConfig",
    "TranscriptionResult",
    "list_audio_files",
]
