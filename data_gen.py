# encoding=utf-8
import json
import os

import cv2 as cv
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import train_image_folder, train_annotations_filename, valid_image_folder, \
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
        h, w = img.shape[:2]
        x = torch.zeros((3, h, w), dtype=torch.float)
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        x[:, :, :] = img

        num_humen = len(human_annots)
        boxes = torch.zeros((num_humen, 4), dtype=torch.int)
        labels = torch.zeros((num_humen, 1), dtype=torch.int)
        keypoints = torch.zeros((num_humen, 14, 3), dtype=torch.int)

        for i in range(num_humen):
            key = 'human' + str(i + 1)
            human_annot = human_annots[key]
            boxes[i] = human_annot
            keypoint_annot = keypoint_annots[i]
            keypoints[i] = keypoint_annot
            labels[i] = 1

        target = dict()
        target['boxes'] = boxes
        target['labels'] = labels
        target['keypoints'] = keypoints

        return x, target

    def __len__(self):
        return len(self.samples)
