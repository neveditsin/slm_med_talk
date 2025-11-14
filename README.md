# ASR, OCR, and Open IE Tutorial

This repository contains three small, self‑contained tutorial toolkits:

1. **ASR** – transcribe audio into text.
2. **OCR** – extract text from document images.
3. **OIE** – turn clinical notes into structured JSON/XML/YAML.

Each stage lives in its own folder with a focused `README.md`, sample data, and
CLI/notebook entry points.

## Repository Layout

```text
slm_med_talk/
├── ASR_toolkit/   # Audio → text with Hugging Face ASR models
├── OCR_toolkit/   # Images → text with multiple OCR engines
└── OIE_toolkit/   # Clinical notes → structured outputs via local LLMs
```

## Prerequisites

- Python 3.10+ recommended.
- Basic familiarity with virtual environments (`python -m venv`).
- (Optional) GPU + CUDA for larger ASR/OCR/LLM models.

Each toolkit has its own `requirements.txt`. The safest approach is to create a
fresh virtual environment per toolkit.

---

## Stage 1 – ASR: Audio to Text

Folder: `ASR_toolkit/`  
Local docs: `ASR_toolkit/README.md`

This stage demonstrates how to transcribe audio using Hugging Face ASR models
through a reusable Python package and CLI.

**Quick start**

```bash
cd ASR_toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m asr_toolkit.cli \
  --input ./samples \
  --output ./demo_transcripts \
  --model openai/whisper-small
```

- The sample audio `samples/6_m.wav` is included so you can validate the flow.
- See `ASR_toolkit/notebooks/Single_File_ASR_Demo.ipynb` for an interactive
  walkthrough of a single recording.

Once you are comfortable transcribing the sample, swap in your own audio files
and re‑run the CLI or notebook.

---

## Stage 2 – OCR: Images to Text

Folder: `OCR_toolkit/`  
Local docs: `OCR_toolkit/README.md`

This stage runs one or more OCR engines over document images and writes out raw
text files per model.

**Quick start**

```bash
cd OCR_toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_ocr.py \
  --input-dir samples \
  --models qwen25vl7b \
  --output-text-dir outputs
```

- The `samples/` folder contains a small set of demo images with different
  artifacts (blur, noise, rotation, etc.).
- You can switch `--models` to any supported engine(s) listed in
  `OCR_toolkit/README.md`.

Use this stage to generate OCR text that you can later feed into downstream
evaluation or information‑extraction workflows.

---

## Stage 3 – OIE: Clinical Notes to Structured Data

Folder: `OIE_toolkit/`  
Local docs: `OIE_toolkit/README.md`

This stage shows how to convert free‑text clinical notes into structured JSON,
YAML, or XML using local LLMs, and how to check that the outputs are
parseable.

**Quick start**

```bash
cd OIE_toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

huggingface-cli login  # once per machine
huggingface-cli download microsoft/Phi-3-mini-128k-instruct --local-dir ./models/phi3-mini

python -m oie_toolkit \
  --input samples \
  --output outputs \
  --model ./models/phi3-mini
```

- `samples/` contains a few synthetic notes for practice.
- Notebooks under `OIE_toolkit/notebooks/` walk through single‑note conversion
  and parseability checks in more detail.

You can replace the sample notes with your own, point `--model` at a different
checkpoint, or import the `oie_toolkit` package directly in your own scripts.

---

Each stage is designed to be small and "hackable" - feel free to copy the code,
add new models, or plug in your own data at any point in the tutorial.

