# import the necessary packages

import cv2 as cv
import keras.backend as K
import numpy as np
import pylab as plt

from config import image_h, image_w, num_connections, num_joints_and_bkg
from data_utils import ALL_PAF_MASK, ALL_HEATMAP_MASK
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
    plt.show()

    input_img = np.transpose(np.float32(imageToTest[:, :, :, np.newaxis]),
                             (3, 0, 1, 2))  # required shape (1, width, height, channels)
    input_img = input_img / 256 - 0.5
    print("Input shape: " + str(input_img.shape))

    batch_paf_masks = np.empty((1, 46, 46, num_connections * 2), dtype=np.uint8)
    batch_paf_masks[0] = ALL_PAF_MASK
    batch_heatmap_masks = np.empty((1, 46, 46, num_joints_and_bkg), dtype=np.uint8)
    batch_heatmap_masks[0] = ALL_HEATMAP_MASK

    output_blobs = model.predict([input_img, batch_paf_masks, batch_heatmap_masks])
    print("Output shape (heatmap): " + str(output_blobs[1].shape))

    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
    heatmap = cv.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    print("Shape after resize (heatmap): " + str(heatmap.shape))

    heatmap = np.expand_dims(heatmap[:, :, 1], -1)
    image = imageToTest * 0.5 + heatmap * 0.5
    image = image.astype(np.uint8)
    cv.imwrite('images/demo.png', image)

    # visualization
    plt.imshow(imageToTest[:, :, [2, 1, 0]])
    plt.imshow(heatmap[:, :, 1], alpha=.5)  # right elbow
    plt.show()

    # paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
    # paf = cv.resize(paf, (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    #
    # plt.imshow(imageToTest[:, :, [2, 1, 0]])
    # plt.imshow(paf[:, :, 6], alpha=.5)  # right elbow
    # plt.show()

    K.clear_session()
