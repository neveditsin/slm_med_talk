# OCR Evaluation Tutorial Toolkit

This directory now serves as a lightweight toolkit for running OCR models over
example images and collecting their raw text outputs.

## Tutorial Map

| Component | Purpose |
| --- | --- |
| `run_ocr.py` | Runs the supported OCR engines over a directory of images and saves raw text files per model. |
| `models/` | Lightweight wrappers or configs for engines that need them (e.g., Surya, GOT-OCR, docTR). |


## Getting Started

1. **Clone / open this folder** and make sure you can see the demo assets in
   the `samples/` directory.
2. **Create a fresh environment** for the base workflow:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Optional environments**
   - PaddleOCR only: `pip install -r requirements_paddleocr.txt` inside a clean env.
4. **System deps**
   - Install Tesseract binaries (and set `TESSERACT_CMD` if not in PATH).
   - Configure CUDA if you plan to run GPU-heavy models (Qwen, InternVL, Phi, Surya, etc.).

## Tutorial Flow

1. **Run OCR over the sample images**
   ```bash
   python run_ocr.py \
     --input-dir samples \
     --models qwen25vl7b \
     --output-text-dir outputs
   ```
The `run_ocr.py` script emits raw OCR text files per model under the directory
given via `--output-text-dir`. You can feed those files into your own evaluation
or analytics pipeline.

## Supported OCR Engines

The runner script understands the following keys (aliases in parentheses):

- `tesseract`
- `paddleocr` (`paddle`)
- `doctr`
- `surya` (`suryaocr`)
- `gotocr2` (`got-ocr2`, `got-ocr2.0`, `ocr2`)
- `qwen25vl7b`
- `phi4mm` (`phi4`)
- `internvl35`

Feel free to register additional engines inside `models/`—the notebooks simply
consume the CSV/JSON outputs, so every new engine instantly shows up in the
visualizations.

## Tips & Troubleshooting

- Use smaller subsets of your dataset when experimenting with GPU-intensive
  models; once the flow works, scale to your full image set.

Happy experimenting! This toolkit is meant to be tinkered with, so feel
free to copy the scripts, adapt them, and integrate them into your own OCR
evaluation pipeline. Start with the `samples/` mini-dataset to explore the
command above, then swap in your real images when you are ready.
