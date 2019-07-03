import os

import cv2 as cv


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def draw_bboxes(img, boxes, labels, scores, keypoints):
    for i, b in enumerate(boxes):
        cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

        person = keypoints[i]
        for kp in person:
            cv.circle(img, (int(kp[0]), int(kp[1])), 1, (0, 255, 0), -1)

        l = labels[i]
        s = scores[i]

        print(l)
        print(s)

    return img
