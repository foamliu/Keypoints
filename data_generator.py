# encoding=utf-8
import json
import os
import keras
import numpy as np
from keras.preprocessing import sequence
from keras.utils import Sequence
import cv2 as cv
from config import image_h, image_w, batch_size, train_image_folder, train_annotations_filename, valid_image_folder, \
    valid_annotations_filename, num_joints_and_bkg


def create_heatmap(num_maps, height, width):
    pass


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
            human = item['human_annotations']
            keypoint = item['keypoint_annotations']
            filename = os.path.join(train_image_folder, '{}.jpg'.format(image_id))
            image = cv.imread(filename)
            image = cv.resize(image, (image_h, image_w))
            image = image[:, :, ::-1]
            batch_input[i_batch] = image/256 - 0.5

            heatmap = create_heatmap(num_joints_and_bkg, 46, 46)


        return batch_input, batch_outputs

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
