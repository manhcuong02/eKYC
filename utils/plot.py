import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_box_and_label(
    image, lw, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)
):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    return image


def plot_landmarks_mtcnn(
    image, landmarks, left_angle=None, right_angle=None, orientation=None
):

    for i, p in enumerate(landmarks.astype(np.uint32)):
        cv2.circle(image, (p[0], p[1]), 3, (0, 0, 255), -1)

    left_eye = np.array(landmarks[0]).astype(np.uint32)
    right_eye = np.array(landmarks[1]).astype(np.uint32)
    nose = np.array(landmarks[2]).astype(np.uint32)

    cv2.line(image, left_eye, right_eye, (255, 0, 0), 1)
    cv2.line(image, left_eye, nose, (0, 255, 0), 1)
    cv2.line(image, nose, right_eye, (0, 0, 255), 1)

    if left_angle:
        cv2.putText(
            image,
            f"left angle: {left_angle: .2f}",
            (20, 20),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            1,
        )

    if right_angle:
        cv2.putText(
            image,
            f"right angle: {right_angle: .2f}",
            (20, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            1,
        )

    if orientation:
        cv2.putText(
            image,
            "orientation: {}".format(orientation),
            (20, 80),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            1,
        )

    return image
