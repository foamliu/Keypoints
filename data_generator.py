# encoding=utf-8
import json
import os

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from config import image_h, image_w, batch_size, train_image_folder, train_annotations_filename, valid_image_folder, \
    valid_annotations_filename, num_joints_and_bkg, num_joints


def from_raw_keypoints(human_annots, keypoint_annots):
    all_joints = []
    num_human = len(human_annots)
    for i in range(1, num_human + 1):
        human_key = 'human' + str(i)
        keypoints = keypoint_annots[human_key]
        joints = []
        for j in range(num_joints):
            x = keypoints[j * 3]
            y = keypoints[j * 3 + 1]
            v = keypoints[j * 3 + 2]
            # only visible and occluded keypoints are used
            if v <= 2:
                joints.append((x, y))
            else:
                joints.append(None)
        all_joints.append(joints)
    return all_joints


def put_heatmap_on_plane(heatmap, plane_idx, joint, height, width, stride):
    pass


def create_heatmap(num_maps, height, width, all_joints, stride):
    heatmap = np.zeros((height, width, num_maps), dtype=np.float64)

    for joints in all_joints:
        for plane_idx, joint in enumerate(joints):
            if joint:
                put_heatmap_on_plane(heatmap, plane_idx, joint, height, width, stride)

    # background
    heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)

    return heatmap


def create_paf():
    pass


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        if usage == 'train':
            self.image_folder = train_image_folder
            annot_filename = train_annotations_filename
        else:
            self.image_folder = valid_image_folder
            annot_filename = valid_annotations_filename

        with open(train_annotations_filename, 'r') as file:
            self.samples = json.load(file)

        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        batch_input = np.empty((length, image_h, image_w, 3), dtype=np.float32)
        batch_outputs = []
        text_input = []

        for i_batch in range(length):
            item = self.samples[i + i_batch]
            image_id = item['image_id']
            human_annots = item['human_annotations']
            keypoint_annots = item['keypoint_annotations']
            filename = os.path.join(train_image_folder, '{}.jpg'.format(image_id))
            image = cv.imread(filename)
            image = cv.resize(image, (image_h, image_w))
            image = image[:, :, ::-1]
            batch_input[i_batch] = image / 256 - 0.5

            all_joints = from_raw_keypoints(human_annots, keypoint_annots)
            heatmap = create_heatmap(num_joints_and_bkg, 46, 46, stride=8)

        return batch_input, batch_outputs

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
