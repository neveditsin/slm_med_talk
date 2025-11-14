#!/usr/bin/env python3
"""
Run one or more OCR models over all images in a directory and save raw text
outputs per model.

This script does not require a manifest and does not compute any accuracy or
noise-related metrics. It simply walks an input directory, runs the requested
models, and writes one ``.txt`` file per image per model.

Example:
    python run_ocr.py \
        --input-dir samples \
        --models tesseract qwen25vl7b \
        --output-text-dir outputs/text
"""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from huggingface_hub import snapshot_download


# ---------------------------------------------------------------------------
# OCR runners

Runner = Callable[[Path], str]

NO_CACHE_MODELS = {}#{"phi4mm", "qwen25vl7b"}

_PADDLE_TEXT_FIELDS: Tuple[str, ...] = (
    "rec_texts",
    "text_lines",
    "text",
    "ocr_text",
    "document_text",
    "transcription",
    "line_texts",
    "content",
    "ocr_results",
    "rec_results",
)


def _build_tesseract_runner() -> Runner:
    from PIL import Image
    from models.ocr_tesseract import run_ocr_tesseract

    def run(image_path: Path) -> str:
        with Image.open(image_path) as img:
            return run_ocr_tesseract(img)

    return run


def _build_donut_runner() -> Runner:
    from PIL import Image
    from models.ocr_donut_base import run_ocr_Donut

    def run(image_path: Path) -> str:
        with Image.open(image_path) as img:
            return run_ocr_Donut(img)

    return run


def _build_got_ocr_runner() -> Runner:
    from models.ocr_OCR2 import run_ocr_OCR2

    def run(image_path: Path) -> str:
        return run_ocr_OCR2(str(image_path))

    return run


def _build_paddleocr_runner() -> Runner:
    try:
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
        from paddleocr import PaddleOCR
        import paddle
    except ImportError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "PaddleOCR is not installed. Install `paddleocr` to use this runner."
        ) from exc

    if not hasattr(paddle, "device"):
        raise ModuleNotFoundError(
            "PaddleOCR requires the `paddlepaddle` runtime. Install it with, e.g.,"
            " `pip install paddlepaddle==2.6.2` (CPU) or the GPU wheel for CUDA-enabled"
            " systems before using the PaddleOCR runner."
        )

    try:
        use_gpu = bool(
            paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        )
    except Exception:  # pragma: no cover - conservative fallback
        use_gpu = False

    paddle_device = "gpu" if use_gpu else "cpu"

    try:
        ocr = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
            device=paddle_device,
            enable_mkldnn=False,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PaddleOCR requires PaddlePaddle's inference components (`paddle.inference`)."
            " Install the full `paddlepaddle` (or `paddlepaddle-gpu`) wheel before running."
        ) from exc

    def run(image_path: Path) -> str:
        result = ocr.ocr(str(image_path))
        if not result:
            return ""

        collected: List[str] = []
        seen: Set[str] = set()

        def _split_and_store(text: str) -> None:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if not any(ch.isalpha() for ch in line):
                    continue
                if line in seen:
                    continue
                seen.add(line)
                collected.append(line)

        def _visit(obj) -> None:  # type: ignore[no-untyped-def]
            if obj is None:
                return
            if isinstance(obj, str):
                _split_and_store(obj)
                return
            if isinstance(obj, (list, tuple, set)):
                for item in obj:
                    _visit(item)
                return
            if isinstance(obj, dict):
                for key in _PADDLE_TEXT_FIELDS:
                    if key in obj:
                        _visit(obj[key])
                        return
                for value in obj.values():
                    if isinstance(value, (str, list, tuple, set, dict)):
                        _visit(value)
                return
            if isinstance(obj, (bytes, bytearray)):
                try:
                    decoded = obj.decode("utf-8", errors="ignore")
                except Exception:
                    return
                _split_and_store(decoded)
                return
            # Ignore other scalar values (numbers, custom objects, etc.).
            return

        _visit(result)

        if not collected:
            return ""

        return "\n".join(collected)

    def cleanup() -> None:
        nonlocal ocr
        ocr = None
        gc.collect()
        try:  # pragma: no cover - optional CUDA cleanup
            empty_cache = getattr(paddle.device.cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception:
            pass

    setattr(run, "_cleanup", cleanup)  # type: ignore[attr-defined]
    return run



def _build_doctr_runner() -> Runner:
    """OCR using Mindee's Doctr library (https://mindee.github.io/doctr/)."""
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except ImportError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "Doctr is not installed. Install with: pip install 'python-doctr[torch]'"
        ) from exc

    # Load a generic end-to-end predictor (DB text detection + CRNN text recognition)
    predictor = ocr_predictor(pretrained=True)

    def run(image_path: Path) -> str:
        # Build a one-page document from the image path
        doc = DocumentFile.from_images([str(image_path)])
        result = predictor(doc)
        exported = result.export()
        lines = []
        try:
            for page in exported.get("pages", []):
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        words = [w.get("value", "") for w in line.get("words", []) if w.get("value")]
                        if words:
                            lines.append(" ".join(words))
        except Exception:
            # Fallback to string representation if structure differs
            return str(exported)
        return "\n".join(lines).strip()

    return run


def _build_surya_ocr_runner() -> Runner:
    """OCR via Surya OCR (https://github.com/chaosdomain/surya or PyPI 'surya')."""
    try:
        import torch  # noqa: F401
        from PIL import Image
        from surya.foundation import FoundationPredictor
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.common.surya.schema import TaskNames
    except ImportError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "Surya OCR is not installed. Try: pip install surya"
        ) from exc

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'

    foundation = FoundationPredictor(device=device)
    detection = DetectionPredictor(device=device)
    recognizer = RecognitionPredictor(foundation)

    def run(image_path: Path) -> str:
        with Image.open(image_path) as img:
            image = img.convert('RGB')

        try:
            results = recognizer(
                [image],
                task_names=[TaskNames.ocr_with_boxes],
                det_predictor=detection,
                sort_lines=True,
                return_words=False,
                drop_repeated_text=True,
            )
        except Exception:
            return ""

        if not results:
            return ""
        result = results[0]
        try:
            lines = []
            for line in getattr(result, 'text_lines', []) or []:
                text = getattr(line, 'text', '')
                if isinstance(text, str) and text.strip():
                    lines.append(text.strip())
            return "\n".join(lines)
        except Exception:
            return ""

    return run

def _build_mmocr_runner() -> Runner:
    """OCR via OpenMMLab's MMOCR (https://github.com/open-mmlab/mmocr)."""
    try:
        from mmocr.apis import MMOCR
    except ImportError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "MMOCR is not installed. Try: pip install 'mmocr>=1.0.0' mmengine mmcv"
        ) from exc

    try:
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'

    # Common, well-supported detector/recognizer pair
    ocr = MMOCR(det='DB_r18', recog='CRNN', device=device)

    def run(image_path: Path) -> str:
        # Prefer simple API that returns list of strings when available
        try:
            lines = ocr.readtext(str(image_path), details=False)
            if isinstance(lines, list):
                return "\n".join([str(x) for x in lines if isinstance(x, (str, int, float)) or x]).strip()
            # Fallback into details=True parsing below
        except Exception:
            lines = None  # type: ignore[assignment]

        try:
            details = ocr.readtext(str(image_path), details=True)
        except Exception as exc:  # pragma: no cover - best effort
            return ""

        collected: List[str] = []

        def _visit(obj) -> None:  # type: ignore[no-untyped-def]
            if obj is None:
                return
            if isinstance(obj, str):
                s = obj.strip()
                if s:
                    collected.append(s)
                return
            if isinstance(obj, (list, tuple, set)):
                for item in obj:
                    _visit(item)
                return
            if isinstance(obj, dict):
                # Common keys across MMOCR versions
                for key in ("text", "texts", "value"):
                    if key in obj and isinstance(obj[key], (str, list)):
                        _visit(obj[key])
                for key in ("result", "res", "predictions", "blocks", "lines", "words"):
                    if key in obj:
                        _visit(obj[key])
                return

        _visit(details)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in collected:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return "\n".join(unique).strip()

    return run


def _build_idefics2_runner() -> Runner:
    """Backwards-compatible runner now powered by InternVL2-4B.

    This replaces the previous Idefics2-based implementation with
    OpenGVLab/InternVL2-4B using its chat API, while preserving the
    canonical name "idefics2" for CLI compatibility.
    """
    import torch
    from PIL import Image
    from transformers import AutoTokenizer, AutoModel

    model_id = "OpenGVLab/InternVL2-4B"

    use_cuda = torch.cuda.is_available()
    # Prefer BF16 on capable GPUs, otherwise fall back to FP16/FP32
    torch_dtype = torch.float32
    if use_cuda:
        try:
            torch_dtype = torch.bfloat16  # type: ignore[attr-defined]
        except Exception:
            torch_dtype = torch.float16

    # Some InternVL2 repos expect slow tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    # Try loading with preferred dtype; if that fails, retry with a safer dtype
    def _load_model(dtype):
        return AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()

    try:
        model = _load_model(torch_dtype)
    except Exception:
        fallback_dtype = torch.float16 if use_cuda else torch.float32
        model = _load_model(fallback_dtype)

    if use_cuda:
        try:
            model = model.to("cuda")  # type: ignore[assignment]
        except Exception:
            pass

    base_prompt = (
        "You are performing OCR on this document. "
        "Transcribe all visible text verbatim as plain text."
    )

    def run(image_path: Path) -> str:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Ensure the <image> sentinel is present so the model links pixels to text
        question = "<image>\n" + base_prompt

        def _to_text(out_obj) -> str:
            try:
                # chat returns (response, history)
                if isinstance(out_obj, (tuple, list)) and len(out_obj) >= 1:
                    out = out_obj[0]
                else:
                    out = out_obj
                return (out if isinstance(out, str) else str(out)).strip()
            except Exception:
                return ""

        with torch.no_grad():
            try:
                out = model.chat(
                    tokenizer,
                    question,
                    image,
                    history=None,
                    do_sample=False,
                    temperature=0.0,
                    num_beams=1,
                    max_new_tokens=1024,
                )
                text = _to_text(out)
                if text:
                    return text
            except Exception:
                pass

            # Optional fallback: some variants expose batch_chat
            try:
                outs = model.batch_chat(
                    tokenizer,
                    [question],
                    [image],
                    history=None,
                    do_sample=False,
                    temperature=0.0,
                    num_beams=1,
                    max_new_tokens=1024,
                )
                if isinstance(outs, (list, tuple)) and outs:
                    text = _to_text(outs[0])
                    if text:
                        return text
            except Exception:
                pass

        return ""

    def cleanup() -> None:
        nonlocal model, tokenizer
        try:
            if model is not None:
                model.to("cpu")
        except Exception:
            pass
        model = None
        tokenizer = None
        gc.collect()
        if use_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    setattr(run, "_cleanup", cleanup)  # type: ignore[attr-defined]
    return run


def _ensure_phi4mm_patch() -> None:
    """
    Patch the dynamically loaded Phi-4 multimodal modules so that they expose
    ``prepare_inputs_for_generation`` which is required by PEFT LoRA wrappers.
    """
    modules_root = Path.home() / ".cache" / "huggingface" / "modules"
    target_root = (
        modules_root
        / "transformers_modules"
        / "microsoft"
        / "Phi_hyphen_4_hyphen_multimodal_hyphen_instruct"
    )
    if not target_root.exists():
        try:
            snapshot_download(
                "microsoft/Phi-4-multimodal-instruct",
                allow_patterns=["*.py"],
            )
        except Exception as exc:  # pragma: no cover - best effort
            logging.debug("snapshot_download for Phi-4 multimodal failed: %s", exc)
        return

    if str(modules_root) not in sys.path:
        sys.path.insert(0, str(modules_root))

    candidates = sorted(
        (p for p in target_root.iterdir() if p.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for subdir in candidates:
        module_name = (
            "transformers_modules.microsoft.Phi_hyphen_4_hyphen_multimodal_hyphen_instruct."
            f"{subdir.name}.modeling_phi4mm"
        )
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        model_cls = getattr(module, "Phi4MMModel", None)
        causal_cls = getattr(module, "Phi4MMForCausalLM", None)

        if model_cls is None or causal_cls is None:
            continue

        if not hasattr(model_cls, "prepare_inputs_for_generation"):
            def _prepare_inputs_for_generation(self, input_ids, **kwargs):
                return {"input_ids": input_ids, **kwargs}

            model_cls.prepare_inputs_for_generation = _prepare_inputs_for_generation
            causal_cls.prepare_inputs_for_generation = _prepare_inputs_for_generation
            logging.debug("Injected prepare_inputs_for_generation into %s", module_name)

        original_forward = getattr(causal_cls, "_phi4mm_original_forward", None)
        if original_forward is None:
            original_forward = causal_cls.forward

            def patched_forward(self, *args, **kwargs):
                if "num_logits_to_keep" not in kwargs or kwargs["num_logits_to_keep"] is None:
                    kwargs["num_logits_to_keep"] = 0
                return original_forward(self, *args, **kwargs)

            causal_cls._phi4mm_original_forward = original_forward
            causal_cls.forward = patched_forward
            logging.debug("Patched Phi4MM forward in %s to enforce num_logits_to_keep", module_name)
        break
    else:
        return

    try:
        from peft.tuners.lora.model import LoraModel
    except Exception:  # pragma: no cover
        return

    if getattr(LoraModel, "_phi4mm_prepare_patch", False):
        return

    original_getattr = LoraModel.__getattr__

    def patched_getattr(self, name):
        if name == "prepare_inputs_for_generation":
            attr = getattr(self.model, name, None)
            if attr is None:
                def _prepare_inputs_for_generation(base_self, input_ids, **kwargs):
                    return {"input_ids": input_ids, **kwargs}
                bound = _prepare_inputs_for_generation.__get__(self.model, self.model.__class__)
                setattr(self.model, name, bound)
                return bound
            return attr
        return original_getattr(self, name)

    LoraModel.__getattr__ = patched_getattr
    LoraModel._phi4mm_prepare_patch = True

def _build_phi4_multimodal_runner() -> Runner:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

    _ensure_phi4mm_patch()  # Keep this for potential LoRA/prepare_inputs compatibility, but it's optional if not using LoRA.

    model_id = "microsoft/Phi-4-multimodal-instruct"
    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            f"Processor for {model_id} does not expose a tokenizer attribute required for chat templates."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if use_cuda else None,
    )
    model.eval()
    model.config.use_cache = False
    generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)
    generation_config.use_cache = False
    generation_config.max_new_tokens = 1024
    generation_config.do_sample = False
    generation_config.num_logits_to_keep = 1  # Keep this as-is.
    generation_config.min_new_tokens = 1
    if not use_cuda:
        device = torch.device("cpu")
        model.to(device)
    else:
        device = model.device

    # Patch the model instance's forward method to enforce num_logits_to_keep.
    if not hasattr(model, '_phi4mm_original_forward'):
        original_forward = model.forward
        def patched_forward(*args, **kwargs):
            if "num_logits_to_keep" not in kwargs or kwargs["num_logits_to_keep"] is None:
                kwargs["num_logits_to_keep"] = 0
            return original_forward(*args, **kwargs)
        model.forward = patched_forward
        model._phi4mm_original_forward = original_forward
        logging.debug("Patched Phi-4 multimodal model instance forward to enforce num_logits_to_keep.")

    # Optionally, ensure prepare_inputs_for_generation is present (for completeness, though not strictly needed here).
    if not hasattr(model, 'prepare_inputs_for_generation'):
        def _prepare_inputs_for_generation(input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}
        model.prepare_inputs_for_generation = _prepare_inputs_for_generation
        logging.debug("Injected prepare_inputs_for_generation into Phi-4 multimodal model instance.")

    image_token = getattr(processor, "image_token", None)
    if not image_token:
        image_token = "<|image_1|>"

    def run(image_path: Path) -> str:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        user_text = (
            f"{image_token}\n"
            "You are performing OCR on this document. "
            "Transcribe all visible text verbatim as plain text."
        )

        conversation = [{"role": "user", "content": user_text}]

        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = getattr(model.generation_config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(model.generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        with torch.no_grad():
            generation = model.generate(
                **inputs,
                generation_config=generation_config,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                num_logits_to_keep=generation_config.num_logits_to_keep,
                use_cache=False,
            )

        continuation = generation[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if decoded.startswith("assistant"):
            decoded = decoded.split("\n", 1)[-1]
        return decoded.strip()

    def cleanup() -> None:
        nonlocal model, processor, tokenizer
        try:
            if model is not None:
                model.to("cpu")
        except Exception:
            pass
        model = None
        processor = None
        tokenizer = None
        gc.collect()
        if use_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    setattr(run, "_cleanup", cleanup)  # type: ignore[attr-defined]
    return run


def _build_qwen25vl_runner() -> Runner:
    import torch
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if use_cuda else None,
    )
    model.eval()
    generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)
    generation_config.use_cache = False
    generation_config.max_new_tokens = 1024
    generation_config.do_sample = False
    generation_config.min_new_tokens = 1
    if not use_cuda:
        device = torch.device("cpu")
        model.to(device)
    else:
        device = model.device

    def run(image_path: Path) -> str:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "You are performing OCR on this document. "
                            "Transcribe all visible text verbatim as plain text."
                        ),
                    },
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                generation_config=generation_config,
                use_cache=False,
            )

        continuation = generated[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if decoded.startswith("assistant"):
            decoded = decoded.split("\n", 1)[-1]
        return decoded.strip()

    def cleanup() -> None:
        nonlocal model, processor
        try:
            if model is not None:
                model.to("cpu")
        except Exception:
            pass
        model = None
        processor = None
        gc.collect()
        if use_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    setattr(run, "_cleanup", cleanup)  # type: ignore[attr-defined]
    return run


def _build_internvl25_runner() -> Runner:
    """OCR using OpenGVLab InternVL2.5-8B via its chat API (pixel_values path)."""
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_id = "OpenGVLab/InternVL2_5-8B"
    use_cuda = torch.cuda.is_available()
    try:
        torch_dtype = torch.bfloat16 if use_cuda and hasattr(torch, 'bfloat16') else (torch.float16 if use_cuda else torch.float32)
    except Exception:  # pragma: no cover
        torch_dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if use_cuda else None,
    )
    model.eval()
    device = model.device if use_cuda else torch.device("cpu")
    if not use_cuda:
        model.to(device)

    # Determine target image size from config; fall back to 448
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        image_size = int(getattr(getattr(cfg, 'vision_config', cfg), 'image_size', 448))
    except Exception:
        image_size = 448

    # Reasonable defaults for ViT-style backbones
    preprocess = T.Compose([
        T.ConvertImageDtype(torch.float32),
    ])
    # Build resize pipeline preserving aspect ratio (no cropping)
    def _prepare_pixel_values(pil: Image.Image) -> torch.Tensor:
        img = pil
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize shortest side to image_size while preserving aspect ratio
        w, h = img.size
        if min(w, h) != image_size:
            scale = image_size / min(w, h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = img.resize((new_w, new_h), Image.BICUBIC)
        # To tensor [0,1] then normalize with ImageNet stats
        tensor = T.ToTensor()(img)
        tensor = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(tensor)
        return tensor.unsqueeze(0)  # (1,3,H,W)

    # InternVL chat expects a dict-like generation_config (it mutates keys internally)
    gen_cfg = {"max_new_tokens": 1024, "do_sample": False}

    base_prompt = (
        "You are performing OCR on this document. "
        "Transcribe all visible text verbatim as plain text."
    )

    def run(image_path: Path) -> str:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        pixel_values = _prepare_pixel_values(image).to(device=device, dtype=model.dtype)

        # Ensure the question contains the <image> sentinel so InternVL links pixels to the prompt
        question = "<image>\n" + base_prompt

        def _decode_out(out_obj) -> str:
            txt = out_obj[0] if isinstance(out_obj, tuple) else out_obj
            if not isinstance(txt, str):
                try:
                    txt = str(txt)
                except Exception:
                    txt = ""
            return txt.strip()

        # Try chat; if empty, fall back to batch_chat
        try:
            out = model.chat(tokenizer, pixel_values, question, gen_cfg)
            text = _decode_out(out)
            if text:
                return text
        except Exception:
            pass

        try:
            outs = model.batch_chat(tokenizer, pixel_values, [question], gen_cfg)
            if isinstance(outs, (list, tuple)) and outs:
                text = _decode_out(outs[0])
                if text:
                    return text
        except Exception:
            pass

        return ""

    def cleanup() -> None:
        nonlocal model, tokenizer
        try:
            if model is not None:
                model.to("cpu")
        except Exception:
            pass
        model = None
        tokenizer = None
        gc.collect()
        if use_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    setattr(run, "_cleanup", cleanup)  # type: ignore[attr-defined]
    return run


def _build_internvl35_runner() -> Runner:
    """OCR using OpenGVLab InternVL3.5-4B via its chat API.

    Reference: https://huggingface.co/OpenGVLab/InternVL3_5-4B
    """
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer

    model_id = "OpenGVLab/InternVL3_5-4B"

    use_cuda = torch.cuda.is_available()
    # Prefer BF16 on CUDA when available
    if use_cuda and hasattr(torch, "bfloat16"):
        torch_dtype = torch.bfloat16
    elif use_cuda:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=use_cuda,
        device_map="auto" if use_cuda else None,
    ).eval()

    # Prepare image tiling to 448px grid with thumbnail, per README
    image_size = 448
    max_tiles = 12
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    def _find_closest_aspect_ratio(aspect_ratio: float, target_ratios, width: int, height: int):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = max_tiles):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height if orig_height else 1.0
        target_ratios = sorted({
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }, key=lambda x: x[0] * x[1])
        target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        if processed_images and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _prepare_pixel_values(pil: Image.Image) -> torch.Tensor:
        tiles = _dynamic_preprocess(pil, max_num=max_tiles)
        pixel_values = torch.stack([transform(im) for im in tiles])
        # place on same device/dtype as model params if CUDA
        try:
            p = next(model.parameters())
            return pixel_values.to(device=p.device, dtype=p.dtype, non_blocking=use_cuda)
        except Exception:
            return pixel_values

    base_prompt = (
        "You are performing OCR on this document. "
        "Transcribe all visible text verbatim as plain text."
    )

    def _clean_text(text: str) -> str:
        s = text.strip()
        if s.lower().startswith("assistant:"):
            s = s.split(":", 1)[-1].lstrip()
        return s

    def run(image_path: Path) -> str:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        pixel_values = _prepare_pixel_values(image)
        question = "<image>\n" + base_prompt

        try:
            # Model's chat API handles both text-only and image-text prompts
            out = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config={"max_new_tokens": 1024, "do_sample": False},
            )
            if isinstance(out, (list, tuple)) and out:
                out = out[0]
            if isinstance(out, str) and out.strip():
                return _clean_text(out)
        except Exception:
            pass
        return ""

    def cleanup() -> None:
        nonlocal model, tokenizer
        try:
            if model is not None:
                model.to("cpu")
        except Exception:
            pass
        model = None
        tokenizer = None
        gc.collect()
        if use_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    setattr(run, "_cleanup", cleanup)  # type: ignore[attr-defined]
    return run


_RUNNER_BUILDERS: Dict[str, Callable[[], Runner]] = {
    "tesseract": _build_tesseract_runner,
    "donut": _build_donut_runner,
    "gotocr2": _build_got_ocr_runner,
    "paddleocr": _build_paddleocr_runner,
    "doctr": _build_doctr_runner,
    "mmocr": _build_mmocr_runner,
    "surya": _build_surya_ocr_runner,
    "internvl25": _build_internvl25_runner,
    "internvl35": _build_internvl35_runner,
    "phi4mm": _build_phi4_multimodal_runner,
    "qwen25vl7b": _build_qwen25vl_runner,
    # Back-compat canonical name kept as "idefics2" but powered by InternVL2-4B
    "idefics2": _build_idefics2_runner,
}

_MODEL_ALIASES: Dict[str, str] = {
    "tesseract": "tesseract",
    "donut": "donut",
    "gotocr2": "gotocr2",
    "gotocr20": "gotocr2",
    "gotocr": "gotocr2",
    "got-ocr2": "gotocr2",
    "got-ocr2.0": "gotocr2",
    "got-ocr20": "gotocr2",
    "ocr2": "gotocr2",
    "ocr-2": "gotocr2",
    "paddleocr": "paddleocr",
    "paddle": "paddleocr",
    "paddleocrv4": "paddleocr",
    "paddleocrv3": "paddleocr",
    "doctr": "doctr",
    "mmocr": "mmocr",
    "surya": "surya",
    "suryaocr": "surya",
    # InternVL3.5 4B aliases
    "opengvlabinternvl354b": "internvl35",
    "internvl354b": "internvl35",
    "internvl35": "internvl35",
    "internvl3p54b": "internvl35",
    "internvl3_5_4b": "internvl35",
    "internvl3.5-4b": "internvl35",
    "opengvlabinternvl258b": "internvl25",
    "internvl258b": "internvl25",
    "internvl25": "internvl25",
    "internvl2p58b": "internvl25",
    "internvl2_5_8b": "internvl25",
    "phi4mm": "phi4mm",
    "phi4": "phi4mm",
    "phi4multimodal": "phi4mm",
    "phi4multimodalinstruct": "phi4mm",
    "microsoftphi4multimodalinstruct": "phi4mm",
    "phi4instruct": "phi4mm",
    "qwen25vl7b": "qwen25vl7b",
    "qwen25vl7binstruct": "qwen25vl7b",
    "qwen2p5vl7b": "qwen25vl7b",
    "qwenvl": "qwen25vl7b",
    "qwenqwen25vl7binstruct": "qwen25vl7b",
    # Map InternVL2-4B names (and old idefics2 aliases) to the idefics2 canonical
    # so users can pass either and get InternVL2-4B.
    "opengvlabinternvl24b": "idefics2",
    "internvl24b": "idefics2",
    "internvl2": "idefics2",
    "internvl2_4b": "idefics2",
    "internvl2-4b": "idefics2",
    "idefics2": "idefics2",
    "idefics": "idefics2",
    "idefics28b": "idefics2",
    "idefics2-8b": "idefics2",
    "huggingfacem4idefics2": "idefics2",
    "huggingfacem4idefics28b": "idefics2",
}


def _canonical_model_name(model_name: str) -> str:
    cleaned = (
        model_name.lower()
        .strip()
        .replace(" ", "")
    )
    cleaned = cleaned.replace("_", "").replace("-", "").replace(".", "").replace("/", "")
    if cleaned not in _MODEL_ALIASES:
        supported = sorted(set(_MODEL_ALIASES.keys()))
        raise KeyError(
            f"Unsupported model '{model_name}'. Supported values: {supported}"
        )
    canonical = _MODEL_ALIASES[cleaned]
    if canonical not in _RUNNER_BUILDERS:
        raise KeyError(
            f"Model '{model_name}' maps to unsupported canonical name '{canonical}'."
        )
    return canonical


def make_runners(
    model_names: Iterable[str],
) -> tuple[Dict[str, Callable[[], Runner]], Dict[str, str], Dict[str, str]]:
    runners: Dict[str, Callable[[], Runner]] = {}
    display_names: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    for requested_name in model_names:
        canonical_name = _canonical_model_name(requested_name)
        if canonical_name in runners:
            # Preserve the first display name encountered.
            continue
        display_names[canonical_name] = requested_name
        runners[canonical_name] = _RUNNER_BUILDERS[canonical_name]
    return runners, display_names, skipped


def _cleanup_runner(canonical_name: str, runner: Runner) -> None:
    cleanup = getattr(runner, "_cleanup", None)
    if callable(cleanup):
        try:
            cleanup()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logging.debug("Cleanup for model '%s' raised an error: %s", canonical_name, exc)


def cleanup_runner(canonical_name: str, runner: Runner) -> None:
    _cleanup_runner(canonical_name, runner)


# ---------------------------------------------------------------------------
# Directory-based processing


def _iter_image_paths(input_dir: Path) -> List[Path]:
    """
    Return all image files directly under ``input_dir``.

    This is intentionally simple: if you need recursion or custom filtering,
    pre-select files and point this script at that directory.
    """
    supported_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_exts
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR models over all images in a directory and save raw text outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "tesseract",
            "donut",
            "got-ocr2.0",
            "doctr",
            "phi-4-multimodal-instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "paddleocr",
            "OpenGVLab/InternVL3_5-4B",
        ],
        help="List of OCR models to run.",
    )
    parser.add_argument(
        "--output-text-dir",
        type=Path,
        required=True,
        help="Directory to store raw OCR outputs per model.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity level.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        logging.error("Input directory does not exist or is not a directory: %s", input_dir)
        return 1

    image_paths = _iter_image_paths(input_dir)
    if not image_paths:
        logging.error("No images found in input directory: %s", input_dir)
        return 1

    try:
        runners, display_names, skipped_models = make_runners(args.models)
    except KeyError as exc:
        logging.error("%s", exc)
        return 2

    if skipped_models:
        for model_name, reason in skipped_models.items():
            logging.warning("Model '%s' was skipped: %s", model_name, reason)

    if not runners:
        logging.error("No OCR models were loaded. Install missing dependencies or adjust --models.")
        return 3

    output_root: Path = args.output_text_dir
    output_root.mkdir(parents=True, exist_ok=True)

    for canonical_name, runner_builder in runners.items():
        display_name = display_names.get(canonical_name, canonical_name)
        runner: Optional[Runner] = None
        logging.info("Initializing model '%s'...", display_name)
        try:
            runner = runner_builder()
        except ModuleNotFoundError as exc:
            logging.error(
                "Skipping model '%s' due to missing dependency: %s",
                display_name,
                exc,
            )
            continue
        except (RuntimeError, FileNotFoundError, EnvironmentError) as exc:
            logging.error("Skipping model '%s': %s", display_name, exc)
            continue
        except Exception:
            logging.exception("Skipping model '%s' due to unexpected error.", display_name)
            continue

        try:
            model_output_dir = output_root / canonical_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            for image_path in image_paths:
                document_id = image_path.stem
                output_file = model_output_dir / f"{document_id}.txt"

                if output_file.exists() and canonical_name not in NO_CACHE_MODELS:
                    try:
                        # Ensure the file is readable; if not, fall back to recomputing.
                        output_file.read_text(encoding="utf-8")
                        logging.info(
                            "Using cached OCR for model '%s' on '%s'.",
                            display_name,
                            image_path,
                        )
                        continue
                    except Exception:
                        logging.warning(
                            "Failed to read cached OCR output '%s'. Recomputing.",
                            output_file,
                        )

                logging.info(
                    "Model '%s' processing image '%s' (document_id='%s').",
                    display_name,
                    image_path,
                    document_id,
                )

                try:
                    raw_text = runner(image_path)  # type: ignore[arg-type]
                    if raw_text is None:
                        raw_text = ""
                    else:
                        raw_text = str(raw_text)
                except Exception:
                    logging.exception(
                        "Model '%s' failed on '%s'.",
                        display_name,
                        image_path,
                    )
                    raw_text = ""

                try:
                    output_file.write_text(raw_text, encoding="utf-8")
                except Exception:
                    logging.warning(
                        "Failed to write OCR output to '%s'.",
                        output_file,
                    )
        finally:
            if runner is not None:
                _cleanup_runner(canonical_name, runner)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
