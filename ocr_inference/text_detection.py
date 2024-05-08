import cv2
import numpy as np
import onnxruntime as rt
from ocr_inference.db_postprocessor import DistillationDBPostProcess
import os


class TextDetector(object):
    def __init__(self, detector_type='eng'):
        assert detector_type in ['eng', 'mult'], "Currently support 2 text detection models: 'eng' and 'mult'"
        # Standardization params
        self.scale = np.float32(1.0 / 255.0)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider = ['CUDAExecutionProvider']

        # Initialise onnxruntime
        if detector_type == 'eng':
            self.det_sess = rt.InferenceSession('ocr_inference/models/det_dyn_eng.onnx',
                                                sess_options=sess_options,
                                                providers=provider)
        elif detector_type == 'mult':
            self.det_sess = rt.InferenceSession('ocr_inference/models/det_dyn_mult.onnx',
                                                sess_options=sess_options,
                                                providers=provider)
        else:
            print("Currently support 2 text detection models: 'eng' and 'mult'")
            raise NotImplementedError
        self.det_outputs = self.det_sess.get_outputs()
        self.det_output_names = list(map(lambda output: output.name, self.det_outputs))
        self.det_input_name = self.det_sess.get_inputs()[0].name

    def to_tensor(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = (img.astype('float32') * self.scale - self.mean) / self.std
        return np.expand_dims(tensor.transpose((2, 0, 1)), axis=0)

    @staticmethod
    def draw_boxes(image, boxes, scores=None, drop_score=0.5):
        if scores is None:
            scores = [1] * len(boxes)

        for (box, score) in zip(boxes, scores):
            if score < drop_score:
                continue
            box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
            image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
        return image

    @staticmethod
    def order_points_clockwise(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def __call__(self, img, debug=False):

        # 1) Scale image
        height, width, _ = img.shape
        if height % 32 != 0:
            height = height + 32 - height % 32
        if width % 32 != 0:
            width = width + (32 - width % 32)
        sample = np.zeros((height, width, 3), dtype='uint8')
        sample[0:img.shape[0], 0:img.shape[1], :] = img

        if debug:
            cv2.imshow('sample', sample)

        # 2) Preprocess Image
        tensor = self.to_tensor(sample)

        # 3) Predict
        detections = self.det_sess.run(self.det_output_names, {self.det_input_name: tensor})

        if debug:
            cv2.imshow('prediction', np.reshape(detections[0].astype(np.float32), (height, width)))

        # 4) Postprocess prediction
        db = DistillationDBPostProcess(unclip_ratio=1.8)
        bboxes = db(detections, [[height, width, 1, 1]])
        bboxes = bboxes[0][0]['points']

        # 5) Order points
        bboxes = [self.order_points_clockwise(pts) for pts in bboxes]

        if debug:
            img_bboxes = self.draw_boxes(img, bboxes)
            cv2.imshow('prediction_with_bboxes', img_bboxes)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return bboxes
