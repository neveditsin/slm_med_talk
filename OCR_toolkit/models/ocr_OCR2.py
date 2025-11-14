
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch

# have to use transformers 4.37.2
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id
if hasattr(model, "generation_config"):
    model.generation_config.pad_token_id = tokenizer.pad_token_id
model = model.eval().cuda()

#General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model
def run_ocr_OCR2(image_path: str) -> str:
    
    text = model.chat(tokenizer, image_path, ocr_type='ocr')
    return text.strip()


