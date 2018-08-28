# import the necessary packages

import cv2 as cv
import keras.backend as K
import numpy as np
import pylab as plt

from config import image_h, image_w
from data_utils import ALL_PAF_MASK, ALL_HEATMAP_MASK
from model import build_model
from utils import get_best_model

if __name__ == '__main__':
    model = build_model()
    model.load_weights(get_best_model())

    test_image = 'images/ski.jpg'
    oriImg = cv.imread(test_image)  # B,G,R order
    imageToTest = cv.resize(oriImg, (image_h, image_w), interpolation=cv.INTER_CUBIC)

    input_img = np.expand_dims(imageToTest, 0)
    input_img = input_img / 256 - 0.5

    batch_paf_masks = np.expand_dims(ALL_PAF_MASK, 0)
    batch_heatmap_masks = np.expand_dims(ALL_HEATMAP_MASK, 0)

    output_blobs = model.predict([input_img, batch_paf_masks, batch_heatmap_masks])

    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
    heatmap = cv.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    print("Output shape (heatmap): " + str(heatmap.shape))

    # visualization
    plt.imshow(imageToTest[:, :, ::-1])
    plt.imshow(heatmap[:, :, 1], alpha=.5)  # right elbow
    np.set_printoptions(threshold=np.inf)
    print(heatmap[:, :, 1])
    plt.savefig('images/demo.png')

    K.clear_session()
