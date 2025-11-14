#base donut models

from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def run_ocr_Donut(image: Image.Image) -> str:
    image = image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
   
    prompt = processor.tokenizer("<s_docvqa>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    output = model.generate(pixel_values, decoder_input_ids=prompt, max_length=512)

    return processor.batch_decode(output, skip_special_tokens=True)[0].strip()
