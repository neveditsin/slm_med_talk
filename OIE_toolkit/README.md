# Clinical Note OIE Tutorial Toolkit

This folder packages lightweight prompt templates plus companion notebooks so
you can walk through the information-extraction workflow using only the sample
notes provided here. Everything runs against locally-hosted models (Phi 3.5,
Qwen2.5, Llama 3, etc.) so you can stay offline while producing JSON/XML/YAML
artifacts and parseability checks.

## Folder Structure

```
OIE_toolkit/
├── notebooks/                  # Tutorial + single-note parseability demos
├── oie_toolkit/                # Prompt + LLM helpers and CLI
├── samples/                    # Three synthetic notes for practice
├── README.md                   # You are here
└── requirements.txt            # Minimal dependencies (PyYAML + Jupyter)
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login  # once per machine
# ensure torch sees your GPU, then pre-download your preferred model:
huggingface-cli download microsoft/Phi-3-mini-128k-instruct --local-dir ./models/phi3-mini
python -m oie_toolkit --input samples --output outputs --model ./models/phi3-mini
```

The command above generates `outputs/noteX.json|yaml|xml` and reruns the
parseability check after each write.

## Tutorial Flow

1. **Explore the samples** – open `samples/note*.txt` to see the raw content
   you will be parsing. The structure mirrors common clinical note templates.
2. **Single-note conversion** – open
   `notebooks/OIE_Single_Note_Conversion.ipynb` to walk through sending one
   sample note to your configured LLM and exporting JSON/YAML/XML artifacts to
   `outputs_single_note/`.
3. **Single-note parseability** – run
   `notebooks/OIE_Single_Note_Parseability.ipynb` to confirm that each format
   round-trips without errors using `validate_serialization`.
4. **Run the extractor** – use the CLI (or import `oie_toolkit.LocalLLMExtractor`)
   to process a folder of notes in one shot. These files then serve as the
   “model output” for downstream tools.
5. **Validate parseability at scale** – the CLI performs JSON/YAML/XML
   round-trip tests for every file. Pass `--no-validate` to skip them when
   prototyping.

## CLI Usage

```bash
python -m oie_toolkit \
  --input samples \
  --output outputs \
  --model microsoft/Phi-3.5-mini-instruct \
  --torch-dtype bfloat16 \
  --device-map auto \
  --format all \
  --max-new-tokens 8192
```

- `--input` accepts a single `.txt` file or a folder (defaults to `samples/`).
- `--model` points to any local or Hugging Face-hosted checkpoint (Phi 3.5, Qwen2.5, Llama 3, etc.).
- `--torch-dtype` and `--device-map` let you target CUDA, ROCm, or CPU boxes.
- `--format` can be `json`, `xml`, `yaml`, or `all` (default) per note.
- `--no-validate` disables the parseability check if you need raw speed.
- Add `--dry-run` to print the generated prompt without loading a model.

## API Snippet

```python
from pathlib import Path
from oie_toolkit.generation import LocalGenerationConfig, LocalLLMExtractor
from oie_toolkit import validate_serialization

text = Path("samples/note1.txt").read_text()
config = LocalGenerationConfig(
    model_name="microsoft/Phi-3-mini-128k-instruct",
    torch_dtype="bfloat16",
    device_map="cuda",
)
extractor = LocalLLMExtractor(config=config)
json_payload = extractor.convert_note(text, note_id="demo-note", fmt="json")
validate_serialization(json_payload, "json")
print(json_payload[:400])
```

## Requirements

See `requirements.txt` for the exact list (`pyyaml`, `transformers`, `torch`,
`jupyterlab`, etc.). Install GPU-enabled PyTorch/ROCm that matches your system
before running the CLI or notebooks.

## Extending the Tutorial

- Drop your own clinical notes into `samples/` and rerun the CLI.
- Attach extraction results to the notebooks to compare against the LLM
  baseline.
- Inspect the prompt helpers in `oie_toolkit/prompts.py` to adjust the schema or
  add additional guardrails.
- Extend `oie_toolkit/validation.py` with domain-specific schema checks if you
  need stricter guarantees.
