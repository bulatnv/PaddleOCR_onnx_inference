import cv2

from ocr_inference.text_detection import TextDetector
from ocr_inference.utils import get_rotate_crop_image, sorted_boxes
from ocr_inference.text_recognition import TextRecognizer
import time


# Press the green button in the gutter to run the script.
class TextPredictor(object):
    def __init__(self, det_framework='paddle', rec_framework='paddle', det_lang='eng', rec_lang='eng'):
        assert det_framework in ['paddle'], \
            "Currently supported text detection frameworks are 'paddle'"
        assert rec_framework in ['paddle'], \
            "Currently supported text recognition frameworks are 'paddle'"
        assert rec_lang in ['eng'], "Currently supported languages are 'eng'"

        # Initialise engines
        self.det_framework = det_framework
        self.det_lang = det_lang
        self.rec_framework = rec_framework
        self.rec_lang = rec_lang
        if det_framework == 'paddle':
            self.detector = TextDetector(detector_type=self.det_lang)
        if rec_framework == 'paddle':
            self.recognizer = TextRecognizer(framework=rec_framework, language=self.rec_lang)

    def __call__(self, image_orig, debug=False):
        detect_result = list()
        text_boxes = list()
        img = image_orig.copy()

        if self.det_framework == 'paddle':
            # 1) Detect text
            '''
                    Существует 2 базовые модели детекторов текста английский 'eng' и мультиязычный 'mult'
            '''

            text_bboxes = self.detector(image_orig, debug=False)

            # 2) Sort text bboxes from top to bottom, from left to right
            text_bboxes = sorted_boxes(text_bboxes)

            # 3) Crop text blocks from image (4 point transform)
            text_crops = []
            for points in text_bboxes:
                text_crops.append(get_rotate_crop_image(image_orig, points))

            # 4) Recognize text crops
            '''
                На данный момент сделана поддержка 2-х моделей для распознавания русского 'rus' и английского 'eng' языка 
            '''

            for crop, bbox in zip(text_crops, text_bboxes):
                text, conf = self.recognizer(crop)
                if text:
                    points = bbox.tolist()
                    if text:
                        detect_result.append({'text': text, 'confidence': float(conf), 'bbox': points})
                        text_boxes.append(points)

        # Postprocessing (drawing bboxes and text on image)
        # print(len(text_boxes))
        img_with_bboxes = TextDetector.draw_boxes(img, text_boxes)
        for detection in detect_result:
            text_str = detection['text']
            conf = detection['confidence']
            points = detection['bbox']
            img_with_bboxes = TextRecognizer.put_text(img_with_bboxes, text_str, conf, points[0])

        return img_with_bboxes, detect_result

    @staticmethod
    def test():
        import cv2
        req_dict = {'det_framework': 'paddle', 'rec_framework': 'paddle', 'det_lang': 'eng', 'rec_lang': 'rus'}
        predictor = TextPredictor(**req_dict)
        image = cv2.imread('ocr_inference/test_images/718_0_2021-04-29.08-03-02.154065.jpg')
        start_time = time.time()
        _, _ = predictor(image)
        print('Total processing time: ', time.time() - start_time)
        return 0


if __name__ == '__main__':
    # Example usage:
    # cd paddle_ocr/ocr_projects/ocr_onnx_inference/
    # export PYTHONPATH=./
    # python ocr_inference/predictor.py
    TextPredictor.test()
