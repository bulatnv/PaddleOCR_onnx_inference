from ocr_inference.predictor import TextPredictor
import numpy as np
import time
import json
import base64
import io
from PIL import Image
import os


track_time = os.getenv('OCR_TRACK_TIME', False)
debug = os.getenv('OCR_DEBUG', False)

def init_context(context):
    context.logger.info("Init context...  0%")
    req_dict = {'det_framework': 'paddle', 'rec_framework': 'paddle', 'det_lang': 'eng', 'rec_lang': 'rus'}
    context.user_data.model_handler = TextPredictor(**req_dict)
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run OCR model")
    data = event.body
    
    threshold = float(data.get("threshold", 0.5))
    buf = io.BytesIO(base64.b64decode(data["image"]))
    pil_image = Image.open(buf)
    # The model uses BGR image format
    image = np.array(pil_image)[:,:,:3][:, :, ::-1]

    start_time = time.time()
    _, result_dict = context.user_data.model_handler(image, debug)
    if track_time:
        context.logger.info('Total processing time: ', time.time() - start_time)

    results = []
    for poly_dict in result_dict:
        label = "text_polygon"
        if poly_dict['confidence'] >= threshold:
            results.append({
                "confidence": str(float(poly_dict['confidence'])),
                "label": label,
                "points": [x for x1x2 in poly_dict['bbox'] for x in x1x2],
                "type": "polygon",
                "attributes": [
                    {"name": "text", "value": poly_dict['text']}
                ]
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
