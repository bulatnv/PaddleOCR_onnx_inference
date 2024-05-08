import cv2
import numpy as np
import onnxruntime as rt
import os
import pymorphy2


class TextRecognizer(object):
    def __init__(self, framework='paddle', language='eng'):
        # assert language in ['eng', 'rus'], "Currently supported languages are 'eng' and 'rus'"
        assert language in ['eng'], "Currently supported languages are 'eng'"
        self.framework = framework
        self.language = language

        providers = os.getenv('OV_EXEC_PROVIDER', 'CPUExecutionProvider').split(';')

        if self.framework == 'paddle':
            self.letters = []
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            # This ORT build has ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] enabled.
            provider = ['CPUExecutionProvider']

            if self.language == 'eng':
                with open('ocr_inference/dictionaries/dict_eng.txt', 'r', encoding='utf8') as fr:
                    self.letters = list(map(lambda item: item.strip('\n\r'), fr.readlines()))

                self.rec_sess = rt.InferenceSession('ocr_inference/models/rec_dyn_eng.onnx',
                                                    sess_options=sess_options,
                                                    providers=provider)

            # elif self.language == 'rus':
            #     with open('ocr_inference/dictionaries/rus_char.txt', 'r', encoding='utf8') as fr:
            #         self.letters = list(map(lambda item: item.strip('\n'), fr.readlines()))
            #
            #     self.rec_sess = rt.InferenceSession('ocr_inference/models/rec.onnx',
            #                                         sess_options=sess_options,
            #                                         providers=provider)

            else:
                print("Currently supported languages are 'eng' and 'rus'")
                raise NotImplementedError

            self.rec_outputs = self.rec_sess.get_outputs()
            self.rec_output_names = list(map(lambda output: output.name, self.rec_outputs))
            self.rec_input_name = self.rec_sess.get_inputs()[0].name

        self.morph = pymorphy2.MorphAnalyzer(lang='ru')

    @staticmethod
    # Base approach for tensor normalization with decolorization
    # Tij = [-1, +1]
    def to_tensor(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.stack((img, img, img), axis=2)
        norm_img = img.astype('float32') / 128. - 1.
        return np.expand_dims(norm_img.transpose((2, 0, 1)), axis=0)

    @staticmethod
    # Approach without decolorization
    # Tensor normalization for SVTR algorithm
    # Tij = [-1, +1]
    def norm_svtr_default(img):
        norm_img = img.astype('float32') / 255.
        norm_img -= 0.5
        norm_img /= 0.5
        norm_img = norm_img.transpose((2, 0, 1))
        return np.expand_dims(norm_img, axis=0)

    @staticmethod
    def put_text(image, text, conf, point):
        x_min = int(point[0])
        y_min = int(point[1])
        text = text + '  ' + str(round(conf * 100, 1)) + '%'
        ((text_width, text_height), _) = cv2.getTextSize(text,
                                                         cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        cv2.rectangle(image, (x_min, y_min - int(1.5 * text_height)), (x_min + text_width, y_min),
                      (0, 0, 255), -1)

        cv2.putText(
            image,
            text=text,
            org=(int(x_min), int(y_min) - int(0.3 * 15)),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
        )
        return image

    def __call__(self, img, debug=False):
        if self.framework == 'paddle':
            # 1) Preprocess Text Crop
            # Scale image, Crop height is fixed.
            # Crop width should be proportional to height and divisible by 12
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            new_width = int(w * (new_height / h))
            img = cv2.resize(img, (new_width, new_height), cv2.INTER_LINEAR)

            # 2) Normalize crop
            tensor = self.norm_svtr_default(img)
            # 3) Predict
            raw_pred = self.rec_sess.run(self.rec_output_names, {self.rec_input_name: tensor})[0][0]

            # 4) Postprocess prediction
            # Row prediction to word and confidence
            letter_confs = np.max(raw_pred, axis=1)
            letter_indices = np.argmax(raw_pred, axis=1)

            # Filter empty ad duplicated letters
            # accepted = [letter_indices != 0]
            accepted = []
            last = -1
            for indx, letter_index in enumerate(letter_indices):
                if letter_index != 0 and letter_index != last:
                    accepted.append(indx)
                last = letter_index
            if not accepted:
                return [], 0

            word_confs = letter_confs[np.array(accepted)]
            word_letters = [self.letters[letter_index] for letter_index in letter_indices[np.array(accepted)]]
            word = ''.join(word_letters).strip()

            if debug:
                print(word, round(float(np.mean(word_confs)), 5))

            return word, round(float(np.mean(word_confs)), 5)
