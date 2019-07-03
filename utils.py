import os

import cv2 as cv


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def draw_bboxes(img, boxes, scores, keypoints):
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        print(x0, y0, x1, y1)
        cv.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)

        s = scores[i]
        print(s)

        cv.putText(img, s, (x0 + 1, y0 + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
        cv.putText(img, s, (x0, y0), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

        person = keypoints[i]
        for kp in person:
            cv.circle(img, (int(kp[0]), int(kp[1])), 1, (0, 255, 0), -1)

    return img
