import random

import cv2 as cv
import numpy as np
import torch

from facenet.models.mtcnn import MTCNN
from liveness_detection.blink_detection import *
from liveness_detection.emotion_prediction import *
from liveness_detection.face_orientation import *
from utils.functions import extract_face


def random_challenge():
    return random.choice(["smile", "surprise", "blink eyes", "right", "left"])


def get_question(challenge):
    """
    Generate a question or instruction based on the challenge.

    Parameters:
        challenge (str): The current challenge, which can be 'smile', 'surprise', 'right', 'left', 'front', or 'blink eyes'.

    Returns:
        str or list: A question or instruction related to the challenge.
                     If the challenge is 'blink eyes', returns a list containing the instruction and the required number of blinks.
    """
    if challenge in ["smile", "surprise"]:
        return "Please put on a {} expression".format(challenge)

    elif challenge in ["right", "left", "front"]:
        return "Please turn your face to the {}".format(challenge)

    elif challenge == "blink eyes":

        num = random.randint(2, 4)
        return ["Blink your eyes {} times".format(num), num]


def get_challenge_and_question():
    challenge = random_challenge()

    question = get_question(challenge)

    return challenge, question


def blink_response(image, box, question, model: BlinkDetector):

    thresh = question[1]
    blink_success = model.eye_blink(image, box, thresh)

    return blink_success


def face_response(challenge: str, landmarks: list, model: FaceOrientationDetector):

    orientation = model.detect(landmarks)

    return orientation == challenge


def emotion_response(face, challenge: str, model: EmotionPredictor):

    emotion = model.predict(face)

    return emotion == challenge


def result_challenge_response(
    frame: np.ndarray, challenge: str, question, model: list, mtcnn: MTCNN
):
    """
    Process the response to a challenge based on the input frame.

    Parameters:
        frame (np.ndarray): RGB color image.
        challenge (str): The current challenge, which can be 'smile', 'surprise', 'right', 'left', 'front', or 'blink eyes'.
        question:  A question or instruction related to the challenge.
        model (list): List of models used, including [blink_model, face_orientation_model, emotion_model].
        mtcnn (MTCNN): MTCNN object used for face extraction.

    Returns:
        bool: The result of the challenge (True if correct, False if incorrect).
    """
    face, box, landmarks = extract_face(frame, mtcnn, padding=10)
    if box is not None:
        if challenge in ["smile", "surprise"]:
            isCorrect = emotion_response(face, challenge, model[2])

        elif challenge in ["right", "left", "front"]:
            isCorrect = face_response(challenge, landmarks, model[1])

        elif challenge == "blink eyes":
            isCorrect = blink_response(frame, box, question, model[0])

        return isCorrect
    return False


if __name__ == "__main__":

    video = cv.VideoCapture(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN()
    blink_detector = BlinkDetector()
    emotion_predictor = EmotionPredictor()
    face_orientation_detector = FaceOrientationDetector()

    model = [blink_detector, face_orientation_detector, emotion_predictor]

    challenge, question = get_challenge_and_question()
    challengeIsCorrect = False

    count = 0
    while True:
        ret, frame = video.read()

        if ret:
            frame = cv.flip(frame, 1)
            if challengeIsCorrect is False:

                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                challengeIsCorrect = result_challenge_response(
                    rgb_frame, challenge, question, model, mtcnn
                )

                if isinstance(question, list):
                    cv.putText(
                        frame,
                        "Question: {}".format(question[0]),
                        (20, 20),
                        cv.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                else:
                    cv.putText(
                        frame,
                        "Question: {}".format(question),
                        (20, 20),
                        cv.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            cv.imshow("", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

            count += 1

            if challengeIsCorrect is True and count >= 100:
                challenge, question = get_challenge_and_question()
                print(question)
                challengeIsCorrect = False

                count = 0
        else:
            break
