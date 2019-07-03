# import the necessary packages
import cv2 as cv
import torch
import torchvision
from torchvision import transforms

from utils import draw_bboxes

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

if __name__ == '__main__':
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    transformer = data_transforms['valid']

    test_image = 'images/ski.jpg'
    bgr_img = cv.imread(test_image)  # B,G,R order
    h, w = bgr_img.shape[:2]

    x_test = torch.zeros((3, h, w), dtype=torch.float)
    img = transforms.ToPILImage()(bgr_img)
    img = transformer(img)
    x_test[:, :, :] = img

    with torch.no_grad():
        predictions = model([x_test])[0]

    boxes = predictions['boxes'].cpu().numpy().tolist()
    labels = predictions['labels'].cpu().numpy().tolist()
    scores = predictions['scores'].cpu().numpy().tolist()
    keypoints = predictions['keypoints'].cpu().numpy().tolist()

    # print('boxes.size(): ' + str(boxes.size()))
    # print('labels.size(): ' + str(labels.size()))
    # print('scores.size(): ' + str(scores.size()))
    # print('keypoints.size(): ' + str(keypoints.size()))

    img = draw_bboxes(bgr_img, boxes, labels, scores, keypoints)
    cv.imshow('image', img)
    cv.waitKey(0)
