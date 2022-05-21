from typing import Union

import cv2
import numpy as np
from PIL import Image
from keras.backend import expand_dims
from keras.models import load_model
from numpy import ndarray

emotions: dict[int, str] = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

classCascade = cv2.CascadeClassifier("ClassifierForOpenCV/frontalface_default.xml")

model = load_model("./Models/trained_models/savedModel")


def get_label_from_id(id: int) -> str:
    return emotions[id]


def get_most_probably_emotion(pred: np.ndarray) -> str:
    return get_label_from_id(np.argmax(pred, axis=-1)[0])


def get_emotion_probability_from_id(pred: np.ndarray, id) -> float:
    return pred[0][id]


def sort_dict_and_return_tuple_of_scores_sorted(
        dict_pred: dict[str, float]
) -> list[str]:
    return sorted(dict_pred.items(), key=lambda x: x[1])[::-1]


def get_sorted_results(pred: np.ndarray) -> list:
    proba = {
        emotion: get_emotion_probability_from_id(pred, id)
        for id, emotion in emotions.items()
    }

    return sort_dict_and_return_tuple_of_scores_sorted(proba)


def get_face_from_frame_with_classcascade(
        frame: np.ndarray, class_cascade, shape: tuple
) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, None]]:
    frame = cv2.flip(frame, 1)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = class_cascade.detectMultiScale(
        image=frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    for (x, y, w, h) in faces:
        if x is not None and y is not None:
            face = Image.fromarray(frame[y: y + h, x: x + w]).resize(shape)
            face = expand_dims(np.asarray(face), 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return frame, face
    return frame, None


def get_face_from_frame(
        frame: np.ndarray, shape: tuple
) -> Union[tuple[ndarray, ndarray], tuple[ndarray, None]]:
    return get_face_from_frame_with_classcascade(frame, classCascade, shape)


def get_emotions_from_face(face) -> Union[list, None]:
    if face is not None:
        pred = model.predict(x=face)
        return get_sorted_results(pred)
    return None


def camera_modified(face_shape: tuple):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame, face = get_face_from_frame(cv2.flip(frame, 1), face_shape)
            yield frame, get_emotions_from_face(face)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # below it's just an example of how to use this file
    face_shape = (48, 48)
    for frame, emotion in camera_modified(face_shape):
        # update all the interface here
        cv2.imshow("frame", frame)
        # the "emotion" array is a sorted array of all emotions with their probabilities
        if emotion is not None:
            print(emotion)
