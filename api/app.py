from flask import Flask, request, Response
from ocr_inference.predictor import text_predictor
import numpy as np
import jsonpickle
import cv2
import time


# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/ocr', methods=['POST'])
def detect_and_recognize_text():
    # extract parameters from the request
    req_dict = {'det_framework': 'paddle', 'rec_framework': 'paddle', 'det_lang': 'mult', 'rec_lang': 'rus', 'output': 'json'}
    req_dict.update(request.form.to_dict())
    # convert string of image data to uint8
    nparr = np.frombuffer(request.files['image'].read(), np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # process image with OCR engine
    start_time = time.time()
    result_image, result_dict = text_predictor(img, req_dict['det_framework'], req_dict['rec_framework'], req_dict['det_lang'], req_dict['rec_lang'])
    print('Total processing time: ', time.time() - start_time)
    # return image or json
    if req_dict['output'] == 'image':
        # compress and prepare image for transmission via http
        _, img_encoded = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 20])
        res = Response(response=img_encoded.tobytes(), status=200, mimetype="image/jpeg")
    else:
        response_pickled = jsonpickle.encode(result_dict)
        res = Response(response=response_pickled, status=200, mimetype="application/json")
    return res


# start flask app
app.run(host="0.0.0.0", port=5000, use_reloader=True)