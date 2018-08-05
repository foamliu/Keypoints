import json
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm

from config import train_image_folder, train_annotations_filename

if __name__ == '__main__':
    names = [n for n in os.listdir(train_image_folder) if n.lower().endswith('.jpg')]
    print('train sample number: ' + str(len(names)))

    with open(train_annotations_filename, 'r') as file:
        data = json.load(file)

    width_list = []
    height_list = []
    human_list = []
    keypoint_list = []
    for item in tqdm(data):
        image_id = item['image_id']
        human = item['human_annotations']
        keypoint = item['keypoint_annotations']
        filename = os.path.join(train_image_folder, '{}.jpg'.format(image_id))
        img = cv.imread(filename)
        width_list.append(img.shape[1])
        height_list.append(img.shape[0])
        human_list.append(len(human))
        keypoint_list.append(len(keypoint))
        assert(len(keypoint['human1']) == 21)

    print('avg width: ' + str(np.mean(width_list)))
    print('avg height: ' + str(np.mean(height_list)))
    print('avg num_human of boxes: ' + str(np.mean(human_list)))
    print('max num_human of boxes: ' + str(np.max(human_list)))
    print('min num_human of boxes: ' + str(np.min(human_list)))
    print('avg num_human of keypoints: ' + str(np.mean(keypoint_list)))
