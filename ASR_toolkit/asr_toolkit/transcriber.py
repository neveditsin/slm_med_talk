"""Reusable audio transcription utilities.

The goal of this module is to expose a small, well documented API that both the
command-line scripts and the educational notebooks can share.  It wraps the
🤗 Transformers ASR pipeline with light ergonomics for working with folders of
audio files and persisting transcripts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is optional at runtime
    import torch
except Exception:  # torch is not always available in CPU-only environments
    torch = None  # type: ignore

from transformers import pipeline

try:  # pragma: no cover - tqdm is only used for user feedback
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_EXTENSIONS: Tuple[str, ...] = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".opus",
)


@dataclass
class TranscriptionConfig:
    """Friendly configuration container for :class:`AudioTranscriber`."""

    model_name: str = "openai/whisper-small"
    batch_size: int = 1
    device: Optional[int] = None
    chunk_length_s: Optional[float] = None
    stride_length_s: Optional[float] = None
    generate_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class TranscriptionResult:
    """Return payload from a transcription run."""

    audio_path: Path
    text: str
    language: Optional[str]
    elapsed_seconds: float
    raw: Dict[str, Any]

    def save_text(self, output_path: Path) -> Path:
        """Persist the transcript to ``output_path``."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.text.strip() + "\n", encoding="utf-8")
        return output_path


def list_audio_files(
    input_dir: Path | str, extensions: Sequence[str] = DEFAULT_EXTENSIONS
) -> List[Path]:
    """Collect supported audio files in ``input_dir`` recursively."""
    base = Path(input_dir).expanduser().resolve()
    files: List[Path] = []
    for ext in extensions:
        files.extend(base.rglob(f"*{ext}"))
    # Remove duplicates and enforce sorted order for deterministic processing.
    return sorted(set(files))


class AudioTranscriber:
    """Load-and-go utility for both folder and single-file transcription."""

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing with config: %s", asdict(self.config))
        self._pipeline = self._build_pipeline()

    def _resolve_device(self) -> int:
        """Pick the device ID understood by ``transformers.pipeline``."""
        if self.config.device is not None:
            return self.config.device
        if torch is not None and torch.cuda.is_available():  # pragma: no cover
            return 0
        return -1

    def _build_pipeline(self):
        """Instantiate the 🤗 pipeline once so it can be reused everywhere."""
        pipeline_kwargs: Dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": self.config.model_name,
            "batch_size": self.config.batch_size,
            "device": self._resolve_device(),
        }
        if self.config.chunk_length_s is not None:
            pipeline_kwargs["chunk_length_s"] = self.config.chunk_length_s
        if self.config.stride_length_s is not None:
            pipeline_kwargs["stride_length_s"] = self.config.stride_length_s
        if self.config.generate_kwargs:
            pipeline_kwargs["generate_kwargs"] = self.config.generate_kwargs

        self.logger.info(
            "Loading ASR pipeline for model %s (device=%s)",
            self.config.model_name,
            pipeline_kwargs["device"],
        )
        return pipeline(**pipeline_kwargs)

    def transcribe_file(self, audio_path: Path | str) -> TranscriptionResult:
        """Transcribe a single audio file."""
        path = Path(audio_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        self.logger.debug("Transcribing %s", path)
        start = time.perf_counter()
        raw_result = self._pipeline(str(path))
        elapsed = time.perf_counter() - start

        text = raw_result.get("text", "").strip()
        language = raw_result.get("language") or raw_result.get("language_code")

        if not text:
            raise RuntimeError(f"No transcription text returned for {path}")

        return TranscriptionResult(
            audio_path=path,
            text=text,
            language=language,
            elapsed_seconds=elapsed,
            raw=raw_result,
        )

    def transcribe_folder(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        skip_existing: bool = True,
        save_metadata: bool = True,
    ) -> List[TranscriptionResult]:
        """Transcribe every supported audio file inside ``input_dir``."""
        audio_files = list_audio_files(input_dir, extensions=extensions)
        if not audio_files:
            self.logger.warning("No audio files found in %s", input_dir)
            return []

        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        iterator: Iterable[Path]
        if tqdm is not None:
            iterator = tqdm(audio_files, desc="Transcribing audio", unit="file")
        else:  # pragma: no cover - fallback without tqdm
            iterator = audio_files

        results: List[TranscriptionResult] = []
        for audio_file in iterator:
            target = output_root / f"{audio_file.stem}.txt"
            if skip_existing and target.exists():
                self.logger.info("Skipping existing transcript for %s", audio_file.name)
                continue

            try:
                result = self.transcribe_file(audio_file)
                result.save_text(target)
                if save_metadata:
                    metadata = {
                        "audio_file": str(result.audio_path),
                        "transcript_file": str(target),
                        "model_name": self.config.model_name,
                        "language": result.language,
                        "elapsed_seconds": round(result.elapsed_seconds, 2),
                    }
                    meta_path = target.with_suffix(".json")
                    meta_path.write_text(
                        json.dumps(metadata, indent=2), encoding="utf-8"
                    )
                results.append(result)
                self.logger.info("Saved transcript to %s", target)
            except Exception as exc:  # pragma: no cover - user feedback
                self.logger.exception("Failed to transcribe %s: %s", audio_file, exc)
        return results
