# encoding=utf-8
import math

import numpy as np

from hparams import num_joints, joint_pairs, num_connections, num_joints_and_bkg, im_size

ALL_PAF_MASK = np.ones((46, 46, num_connections * 2), dtype=np.uint8)
ALL_HEATMAP_MASK = np.ones((46, 46, num_joints_and_bkg), dtype=np.uint8)


def from_raw_keypoints(human_annots, keypoint_annots, orig_shape):
    orig_h, orig_w = orig_shape
    all_joints = []
    num_human = len(human_annots)
    for i in range(1, num_human + 1):
        human_key = 'human' + str(i)
        keypoints = keypoint_annots[human_key]
        joints = []
        for j in range(num_joints):
            x = keypoints[j * 3]
            y = keypoints[j * 3 + 1]
            v = keypoints[j * 3 + 2]
            x = x * im_size / orig_w
            y = y * im_size / orig_h
            # only visible and occluded keypoints are used
            if v <= 2:
                joints.append((x, y))
            else:
                joints.append(None)
        all_joints.append(joints)
    return all_joints


def put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride):
    start = stride / 2.0 - 0.5

    center_x, center_y = joint

    for g_y in range(height):
        for g_x in range(width):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue

            heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
            if heatmap[g_y, g_x, plane_idx] > 1.0:
                heatmap[g_y, g_x, plane_idx] = 1.0


def put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2, threshold, height, width):
    min_x = max(0, int(round(min(x1, x2) - threshold)))
    max_x = min(width, int(round(max(x1, x2) + threshold)))

    min_y = max(0, int(round(min(y1, y2) - threshold)))
    max_y = min(height, int(round(max(y1, y2) + threshold)))

    vec_x = x2 - x1
    vec_y = y2 - y1

    norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
    if norm < 1e-8:
        return

    vec_x /= norm
    vec_y /= norm

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - x1
            bec_y = y - y1
            dist = abs(bec_x * vec_y - bec_y * vec_x)

            if dist > threshold:
                continue

            cnt = countmap[y][x][plane_idx]

            if cnt == 0:
                vectormap[y][x][plane_idx * 2 + 0] = vec_x
                vectormap[y][x][plane_idx * 2 + 1] = vec_y
            else:
                vectormap[y][x][plane_idx * 2 + 0] = (vectormap[y][x][plane_idx * 2 + 0] * cnt + vec_x) / (cnt + 1)
                vectormap[y][x][plane_idx * 2 + 1] = (vectormap[y][x][plane_idx * 2 + 1] * cnt + vec_y) / (cnt + 1)

            countmap[y][x][plane_idx] += 1


def create_heatmap(num_maps, height, width, all_joints, sigma, stride):
    heatmap = np.zeros((height, width, num_maps), dtype=np.float64)

    for joints in all_joints:
        for plane_idx, joint in enumerate(joints):
            if joint:
                put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride)

    # background
    heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)

    return heatmap


def create_paf(num_maps, height, width, all_joints, threshold, stride):
    vectormap = np.zeros((height, width, num_maps * 2), dtype=np.float64)
    countmap = np.zeros((height, width, num_maps), dtype=np.uint8)
    for joints in all_joints:
        for plane_idx, (j_idx1, j_idx2) in enumerate(joint_pairs):
            center_from = joints[j_idx1]
            center_to = joints[j_idx2]

            # skip if no valid pair of keypoints
            if center_from is None or center_to is None:
                continue

            x1, y1 = (center_from[0] / stride, center_from[1] / stride)
            x2, y2 = (center_to[0] / stride, center_to[1] / stride)

            put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2,
                             threshold, height, width)

    return vectormap
