import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from config import num_joints
from data_gen import DataGenSequence

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

        cv.imwrite('images/datav_image_{}.png'.format(j), image)

        heatmap1 = cv.resize(heatmap[:, :, body_part], (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
        plt.imshow(image[:, :, ::-1])
        plt.imshow(heatmap1, alpha=.5)
        plt.savefig('images/datav_heatmap_{}.png'.format(j))

        pafmap1 = cv.resize(pafmap[:, :, paf_num * 2], (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
        plt.imshow(image[:, :, ::-1])
        plt.imshow(pafmap1, alpha=.5)
        plt.savefig('images/datav_paf_dx_{}.png'.format(j))

        pafmap2 = cv.resize(pafmap[:, :, paf_num * 2 + 1], (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
        plt.imshow(image[:, :, ::-1])
        plt.imshow(pafmap2, alpha=.5)
        plt.savefig('images/datav_paf_dy_{}.png'.format(j))

    from numpy import ma

    image = batch_images[0]
    pafmap = batch_pafmaps[0]

    paf_num1 = 4  # RShoulder
    paf_num2 = 5  # RElbow

    U = cv.resize(pafmap[:, :, paf_num1], (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC) * -1
    V = cv.resize(pafmap[:, :, paf_num2], (0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
    M = np.zeros(U.shape, dtype='bool')
    M[U ** 2 + V ** 2 < 0.5 * 0.5] = True
    U = ma.masked_array(U, mask=M)
    V = ma.masked_array(V, mask=M)

    plt.figure()
    plt.imshow(image[:, :, ::-1], alpha=.5)
    s = 5
    Q = plt.quiver(X[::s, ::s], Y[::s, ::s], U[::s, ::s], V[::s, ::s],
                   scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.savefig('images/datav_paf_vectors.png')
