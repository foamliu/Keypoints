# import the necessary packages

import cv2 as cv
import keras.backend as K
import numpy as np
import pylab as plt

from config import image_h, image_w, scale
from model import build_model
from utils import get_best_model

if __name__ == '__main__':
    model = build_model()
    model.load_weights(get_best_model())

    test_image = 'images/ski.jpg'
    oriImg = cv.imread(test_image)  # B,G,R order
    plt.imshow(oriImg[:, :, [2, 1, 0]])

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 30))

    imageToTest = cv.resize(oriImg, (image_h, image_w), interpolation=cv.INTER_CUBIC)
    plt.imshow(imageToTest[:, :, [2, 1, 0]])

    input_img = np.transpose(np.float32(imageToTest[:, :, :, np.newaxis]),
                             (3, 0, 1, 2))  # required shape (1, width, height, channels)
    print("Input shape: " + str(input_img.shape))

    output_blobs = model.predict(input_img)
    print("Output shape (heatmap): " + str(output_blobs[1].shape))

    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
    heatmap = cv.resize(heatmap, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

    paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
    paf = cv.resize(paf, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

    # visualization
    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(heatmap[:, :, 3], alpha=.5)  # right elbow
    plt.show()

    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(paf[:, :, 16], alpha=.5)  # right elbow
    plt.show()

    K.clear_session()
