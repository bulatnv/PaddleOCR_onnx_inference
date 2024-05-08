# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os.path
import time

import cv2
import datetime
from ocr_inference.predictor import TextPredictor
import glob
from tqdm import tqdm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predictor = TextPredictor(det_framework='paddle', rec_framework='paddle', det_lang='eng', rec_lang='eng')
    img_paths = glob.glob('imgs/*.jpg')

    start = time.time()

    for file_path in tqdm(img_paths):
        image_orig = cv2.imread(file_path)
        result_image, result_dict = predictor(image_orig, debug=True)
        time_stamp = datetime.datetime.now().strftime('%H%M%S%f')[:-2]
        img_name = os.path.basename(file_path)
        cv2.imwrite(f'results/{time_stamp}_{img_name}', result_image)

    finish = time.time()
    print('Inference time:', round((finish - start) * 1000, 3), 'ms')
    print('Average time per image:', round((finish - start) * 1000 / len(img_paths), 3), 'ms per img')

    # Object destructor required, to eliminate TRT errors
    del predictor

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
