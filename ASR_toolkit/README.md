# Open ASR Tutorial Toolkit

This repository demonstrates how to transcribe audio with Hugging Face ASR
models (e.g., Whisper) using a single reusable code path.  Everything has been
organized into the `asr_toolkit` Python package so notebooks, CLI commands, and
any future scripts import the same logic.

## Repository Layout

```text
ASR_toolkit/
├── asr_toolkit/    # Python package (AudioTranscriber + CLI helpers)
├── notebooks/      # Interactive walkthroughs (Single_File_ASR_Demo.ipynb)
├── samples/        # Example audio clip (samples/6_m.wav)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Tip: If you prefer editable installs, run `pip install -e .` from this folder
> so `asr_toolkit` can be imported globally.

## Batch Transcription

```bash
python -m asr_toolkit.cli \
  --input ./samples \
  --output ./demo_transcripts \
  --model openai/whisper-small
```

- Point `--input` at a single file or a directory of audio recordings.
- Each audio file produces a `.txt` transcript plus a `.json` metadata summary
  inside the specified output folder.

## Single-File Notebook

```bash
jupyter lab notebooks/Single_File_ASR_Demo.ipynb
```

The notebook loads `samples/6_m.wav`, ensures the repository root is on
`sys.path`, imports `AudioTranscriber` from the package, and walks through a
single inference run. The transcript is saved to `notebooks/output/`.

## Customization Tips

- `asr_toolkit/transcriber.py` centralizes model loading, batching, and output
  serialization. Point `TranscriptionConfig.model_name` at any compatible Hugging
  Face checkpoint (Whisper, wav2vec2, etc.).
- `asr_toolkit/cli.py` exposes CLI arguments for chunk length, stride, and batch
  size so you can adapt to longer recordings.
- Replace `samples/6_m.wav` with your own recordings once you have validated the
  workflow end to end.
