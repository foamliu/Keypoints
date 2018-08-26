# import the necessary packages

import cv2 as cv
import keras.backend as K
import pylab as plt

from model import build_model
from utils import get_best_model

if __name__ == '__main__':
    model = build_model()
    model.load_weights(get_best_model())

    test_image = 'images/ski.jpg'
    oriImg = cv.imread(test_image)  # B,G,R order
    plt.imshow(oriImg[:, :, [2, 1, 0]])

    K.clear_session()
