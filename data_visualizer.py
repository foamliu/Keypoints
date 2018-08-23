import json
import os
import random

import cv2 as cv
import numpy as np

from config import image_h, image_w, train_image_folder, train_annotations_filename, num_joints_and_bkg, num_connections
from data_utils import from_raw_keypoints, create_heatmap, create_paf

if __name__ == '__main__':
    image_folder = train_image_folder
    annot_filename = train_annotations_filename

    with open(annot_filename, 'r') as file:
        samples = json.load(file)
    items = random.sample(samples, 1)

    for k, item in enumerate(items):
        image_id = item['image_id']
        human_annots = item['human_annotations']
        keypoint_annots = item['keypoint_annotations']
        filename = os.path.join(train_image_folder, '{}.jpg'.format(image_id))
        image = cv.imread(filename)
        image = cv.resize(image, (image_h, image_w))
        cv.imwrite('images/image_datav_{}.png'.format(k), image)
        image = image[:, :, ::-1]

        all_joints = from_raw_keypoints(human_annots, keypoint_annots)
        heatmap = create_heatmap(num_joints_and_bkg, 46, 46, all_joints, sigma=7.0, stride=8)
        pafmap = create_paf(num_connections, 46, 46, all_joints, 1, stride=8)

        part_index = random.choice(range(num_joints_and_bkg))
        heatmap = heatmap[:, :, part_index]
        frame = np.zeros((image_h, image_w), np.uint8)
        for i in range(46):
            for j in range(46):
                left = j * 8
                top = i * 8
                right = left + 7
                bottom = top + 7
                cv.rectangle(frame, (left, top), (right, bottom), int(heatmap[i, j] * 255), cv.FILLED)
                print(heatmap[i, j])
        cv.imwrite('images/PCM_datav_{}.png'.format(k), frame)
