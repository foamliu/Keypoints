# encoding=utf-8
import json
import os

import cv2 as cv
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

        with open(annot_filename, 'r') as file:
            self.samples = json.load(file)

    def __getitem__(self, i):
        item = self.samples[i]
        image_id = item['image_id']
        human_annots = item['human_annotations']
        keypoint_annots = item['keypoint_annotations']
        filename = os.path.join(self.image_folder, '{}.jpg'.format(image_id))
        image = cv.imread(filename)
        h, w = image.shape[:2]

        print('human_annots: ' + str(human_annots))
        print('keypoint_annots: ' + str(keypoint_annots))

        print(h, w)

        target = dict()
        target['boxes'] = None
        target['labels'] = None
        target['keypoints'] = None

        return image, target

    def __len__(self):
        return len(self.samples)
