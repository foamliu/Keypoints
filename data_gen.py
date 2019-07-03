# encoding=utf-8
import json
import os

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import im_size, train_image_folder, train_annotations_filename, valid_image_folder, \
    valid_annotations_filename

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def adjust_human_annot(human_annot, w_ratio, h_ratio):
    human_annot[0] = human_annot[0] * w_ratio
    human_annot[1] = human_annot[1] * h_ratio
    human_annot[2] = human_annot[2] * w_ratio
    human_annot[3] = human_annot[3] * h_ratio
    return human_annot


def adjust_keypoint_annot(keypoint_annot, w_ratio, h_ratio):
    for i in range(14):
        keypoint_annot[i][0] = keypoint_annot[i][0] * w_ratio
        keypoint_annot[i][1] = keypoint_annot[i][1] * h_ratio
    return keypoint_annot


class KpDataset(Dataset):
    def __init__(self, split):
        if split == 'train':
            self.image_folder = train_image_folder
            annot_filename = train_annotations_filename
        else:
            self.image_folder = valid_image_folder
            annot_filename = valid_annotations_filename

        self.transformer = data_transforms[split]

        with open(annot_filename, 'r') as file:
            self.samples = json.load(file)

    def __getitem__(self, i):
        item = self.samples[i]
        image_id = item['image_id']
        human_annots = item['human_annotations']
        keypoint_annots = item['keypoint_annotations']
        filename = os.path.join(self.image_folder, '{}.jpg'.format(image_id))
        img = cv.imread(filename)
        img = cv.resize(img, (im_size, im_size))
        h, w = img.shape[:2]
        w_ratio, h_ratio = im_size / w, im_size / h
        x = torch.zeros((3, im_size, im_size), dtype=torch.float)
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        x[:, :, :] = img

        num_humen = len(human_annots)
        boxes = np.zeros((num_humen, 4), dtype=np.float)
        labels = np.zeros((num_humen, 1), dtype=np.float)
        keypoints = np.zeros((num_humen, 14, 3), dtype=np.float)

        for i in range(num_humen):
            key = 'human' + str(i + 1)
            human_annot = human_annots[key]
            boxes[i] = adjust_human_annot(np.array(human_annot), w_ratio, h_ratio)
            keypoint_annot = keypoint_annots[key]
            keypoints[i] = adjust_keypoint_annot(np.array(keypoint_annot).reshape(14, 3), w_ratio, h_ratio)
            labels[i] = 1

        target = dict()
        target['boxes'] = torch.from_numpy(boxes)
        target['labels'] = torch.from_numpy(labels)
        target['keypoints'] = torch.from_numpy(keypoints)

        return x, target

    def __len__(self):
        return len(self.samples)
