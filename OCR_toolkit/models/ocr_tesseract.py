# ocr_tesseract.py
import os
import shutil
from pathlib import Path

import pytesseract
from PIL import Image

_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "tesseract"
_VENDOR_CMD = _VENDOR_ROOT / "usr" / "bin" / "tesseract"
_VENDOR_LIB = _VENDOR_ROOT / "usr" / "lib" / "x86_64-linux-gnu"
_VENDOR_TESSDATA = _VENDOR_ROOT / "usr" / "share" / "tesseract-ocr" / "4.00" / "tessdata"

_tesseract_cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")

if not _tesseract_cmd and _VENDOR_CMD.exists():
    existing_ld = os.environ.get("LD_LIBRARY_PATH")
    vendor_lib_str = str(_VENDOR_LIB)
    if _VENDOR_LIB.exists():
        if existing_ld:
            if vendor_lib_str not in existing_ld.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{vendor_lib_str}:{existing_ld}"
        else:
            os.environ["LD_LIBRARY_PATH"] = vendor_lib_str
if _VENDOR_TESSDATA.exists():
    os.environ.setdefault("TESSDATA_PREFIX", str(_VENDOR_TESSDATA))
    _tesseract_cmd = str(_VENDOR_CMD)

if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
else:
    raise RuntimeError(
        "Tesseract executable not found. Install Tesseract OCR or set the "
        "TESSERACT_CMD environment variable to the binary path."
    )

def run_ocr_tesseract(image: Image.Image) -> str:
    """
    Extract text using Tesseract OCR engine.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return pytesseract.image_to_string(image)
