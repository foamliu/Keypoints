import multiprocessing
import os

import cv2 as cv
import keras.backend as K
import numpy as np
from tensorflow.python.client import device_lib

from config import batch_size


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_loss_funcs():
    """
    Euclidean loss as implemented in caffe
    https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
    :return:
    """

    def _eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2

    losses = {}
    losses["weight_stage1_L1"] = _eucl_loss
    losses["weight_stage1_L2"] = _eucl_loss
    losses["weight_stage2_L1"] = _eucl_loss
    losses["weight_stage2_L2"] = _eucl_loss
    losses["weight_stage3_L1"] = _eucl_loss
    losses["weight_stage3_L2"] = _eucl_loss
    losses["weight_stage4_L1"] = _eucl_loss
    losses["weight_stage4_L2"] = _eucl_loss
    losses["weight_stage5_L1"] = _eucl_loss
    losses["weight_stage5_L2"] = _eucl_loss
    losses["weight_stage6_L1"] = _eucl_loss
    losses["weight_stage6_L2"] = _eucl_loss

    return losses


def get_best_model():
    import re
    pattern = 'model.(?P<epoch>\d+)-(?P<val_loss>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    ensure_folder('models')
    files = [f for f in os.listdir('models/') if p.match(f)]
    filename = None
    if len(files) > 0:
        losses = [float(p.match(f).groups()[1]) for f in files]
        best_index = int(np.argmin(losses))
        filename = os.path.join('models', files[best_index])
    return filename


def create_all_mask(mask, num, stride):
    """
    Helper function to create a stack of scaled down mask.
    :param mask: mask image
    :param num: number of layers
    :param stride: parameter used to scale down the mask image because it has
    the same size as orginal image. We need the size of network output.
    :return:
    """
    scale_factor = 1.0 / stride
    small_mask = cv.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
    small_mask = small_mask[:, :, np.newaxis]
    return np.repeat(small_mask, num, axis=2)