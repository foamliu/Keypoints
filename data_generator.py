# encoding=utf-8
import json
import os

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from config import image_h, image_w, batch_size, train_image_folder, train_annotations_filename, valid_image_folder, \
    valid_annotations_filename, num_joints_and_bkg, num_connections, stages
from data_utils import from_raw_keypoints, create_heatmap, create_paf, ALL_PAF_MASK, ALL_HEATMAP_MASK


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        if usage == 'train':
            self.image_folder = train_image_folder
            annot_filename = train_annotations_filename
        else:
            self.image_folder = valid_image_folder
            annot_filename = valid_annotations_filename

        with open(annot_filename, 'r') as file:
            self.samples = json.load(file)

        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        batch_images = np.empty((length, image_h, image_w, 3), dtype=np.float32)
        batch_paf_masks = np.empty((length, 46, 46, num_connections * 2), dtype=np.uint8)
        batch_heatmap_masks = np.empty((length, 46, 46, num_joints_and_bkg), dtype=np.uint8)
        batch_pafmaps = np.empty((length, 46, 46, num_connections * 2), dtype=np.float32)
        batch_heatmaps = np.empty((length, 46, 46, num_joints_and_bkg), dtype=np.float32)

        for i_batch in range(length):
            item = self.samples[i + i_batch]
            image_id = item['image_id']
            human_annots = item['human_annotations']
            keypoint_annots = item['keypoint_annotations']
            filename = os.path.join(self.image_folder, '{}.jpg'.format(image_id))
            image = cv.imread(filename)
            orig_shape = image.shape[:2]
            image = cv.resize(image, (image_h, image_w))
            image = image[:, :, ::-1]
            batch_images[i_batch] = image / 256 - 0.5
            batch_paf_masks[i_batch] = ALL_PAF_MASK
            batch_heatmap_masks[i_batch] = ALL_HEATMAP_MASK

            all_joints = from_raw_keypoints(human_annots, keypoint_annots, orig_shape)
            heatmap = create_heatmap(num_joints_and_bkg, 46, 46, all_joints, sigma=7.0, stride=8)
            pafmap = create_paf(num_connections, 46, 46, all_joints, 1, stride=8)
            batch_heatmaps[i_batch] = heatmap
            batch_pafmaps[i_batch] = pafmap

        batch_outputs = []
        for _ in range(stages):
            batch_outputs.append(batch_pafmaps)
            batch_outputs.append(batch_heatmaps)

        return [batch_images, batch_paf_masks, batch_heatmap_masks], batch_outputs

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
