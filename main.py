# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os.path
import time

import cv2
import datetime
from ocr_inference.predictor import TextPredictor
from pathlib import Path
from tqdm import tqdm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_paths = list(Path('imgs').glob('*.jpg'))
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = TextPredictor(det_framework='paddle', rec_framework='paddle', det_lang='eng', rec_lang='eng')

    start = time.time()

    for file_path in tqdm(img_paths):
        image_orig = cv2.imread(str(file_path))
        result_image, result_dict = predictor(image_orig, debug=True)
        time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        cv2.imwrite(f'{output_dir}/{time_stamp}_{file_path.name}', result_image)

    finish = time.time()
    print('Inference time:', round((finish - start) * 1000, 3), 'ms')
    print('Average time per image:', round((finish - start) * 1000 / len(img_paths), 3), 'ms per img')

    # Object destructor required, to eliminate TRT errors
    del predictor

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
