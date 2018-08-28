import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from config import num_joints
from data_generator import DataGenSequence


def _get_bgimg(inp, target_size=None):
    """
    Get a RGB image from cv2 BGR
    :param inp:
    :param target_size:
    :return: RGB image
    """
    inp = cv.cvtColor(inp.astype(np.uint8), cv.COLOR_BGR2RGB)
    if target_size:
        inp = cv.resize(inp, target_size, interpolation=cv.INTER_AREA)
    return inp


def display_image(img, heatmap, vectmap):
    """
    Displays an image and associated heatmaps and pafs (all)
    :param img:
    :param heatmap:
    :param vectmap:
    :return:
    """
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Image')
    plt.imshow(_get_bgimg(img))

    a = fig.add_subplot(2, 2, 2)
    a.set_title('Heatmap')
    plt.imshow(_get_bgimg(img, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
    tmp = np.amax(heatmap, axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = vectmap.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('paf-x')
    plt.imshow(_get_bgimg(img, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('paf-y')
    plt.imshow(_get_bgimg(img, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    plt.show()


def display_masks(center, img, mask):
    """
    Displays mask for a given image and marks the center of a main person.
    :param center:
    :param img:
    :param mask:
    :return:
    """
    fig = plt.figure()

    a = fig.add_subplot(2, 2, 1)
    a.set_title('Image')
    i = _get_bgimg(img)
    cv.circle(i, (int(center[0, 0]), int(center[0, 1])), 9, (0, 255, 0), -1)
    plt.imshow(i)

    if mask is not None:
        a = fig.add_subplot(2, 2, 2)
        a.set_title('Mask')
        plt.imshow(mask * 255, cmap=plt.cm.gray)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Image + Mask')
        plt.imshow(_get_bgimg(img), alpha=0.5)
        plt.imshow(mask * 255, cmap=plt.cm.gray, alpha=0.5)

    plt.show()


def show_image_mask_center_of_main_person(g):
    img = g[0].img
    mask = g[0].mask
    center = g[0].aug_center
    display_masks(center, img, mask)


def show_image_heatmap_paf(g):
    img = g[0].img
    paf = g[3].astype(np.float32)
    heatmap = g[4].astype(np.float32)

    display_image(img, heatmap, paf)


def display_heatmap(img, heatmap):
    body_part = 0

    heatmap1 = cv.resize(heatmap[:, :, body_part], (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)

    plt.imshow(img[:, :, [2, 1, 0]])
    plt.imshow(heatmap1[:, :], alpha=.5)
    plt.show()

    print(heatmap.dtype)
    print(np.max(heatmap))
    print(np.min(heatmap))


if __name__ == '__main__':
    datagen = DataGenSequence('train')
    batch_inputs, batch_outputs = datagen.__getitem__(0)
    batch_images, batch_paf_masks, batch_heatmap_masks = batch_inputs[0], batch_inputs[1], batch_inputs[2]
    batch_pafmaps, batch_heatmaps = batch_outputs[0], batch_outputs[1]

    for j in range(num_joints):
        body_part = j
        paf_num = j

        image = batch_images[j]
        pafmap = batch_pafmaps[j]
        heatmap = batch_heatmaps[j]

        image = ((image + 0.5) * 256).astype(np.uint8)
        cv.imwrite('images/datav_image_{}.png'.format(j), image[:, :, ::-1])

        heatmap1 = cv.resize(heatmap[:, :, body_part], (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
        plt.imshow(image)
        plt.imshow(heatmap1, alpha=.5)
        plt.savefig('images/datav_heatmap_{}.png'.format(j))

        pafmap1 = cv.resize(pafmap[:, :, paf_num * 2], (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
        plt.imshow(image)
        plt.imshow(pafmap1, alpha=.5)
        plt.savefig('images/datav_paf_dx_{}.png'.format(j))

        pafmap2 = cv.resize(pafmap[:, :, paf_num * 2 + 1], (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
        plt.imshow(image)
        plt.imshow(pafmap2, alpha=.5)
        plt.savefig('images/datav_paf_dy_{}.png'.format(j))
