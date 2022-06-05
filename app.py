import contextlib
import cv2
from flask import Flask, render_template, Response, request
from keras.models import load_model
from emoji import emojize
from emotion_recognition.prediction import get_face_from_frame, get_emotions_from_face
from keras.applications.resnet import ResNet50
from emotion_recognition.Models.common_functions import (
    create_model,
    get_data,
    fit,
    evaluation_model,
    saveModel,
)

switch, out, capture, rec_frame = (
    1,
    0,
    0,
    0,
)

face_shape = (80, 80)


# ## Load model
# model = load_model("./emotion_recognition/Models/trained_models/resnet50_ferplus")

## Load weights
parameters = {
    "shape": [80, 80],
    "nbr_classes": 7,
    "batch_size": 8,
    "epochs": 50,
    "number_of_last_layers_trainable": 10,
    "learning_rate": 0.001,
    "nesterov": True,
    "momentum": 0.9,
}
model = create_model(architecture=ResNet50, parameters=parameters)
model.load_weights("emotion_recognition/Models/trained_models/resnet50_ferplus.h5")


class_cascade = cv2.CascadeClassifier(
    "./emotion_recognition/ClassifierForOpenCV/frontalface_default.xml"
)
face = None
emotions = None

# instatiate flask app
app = Flask(__name__, template_folder="./templates", static_folder="./staticFiles")

camera = cv2.VideoCapture(0)

emotions_with_smiley = {
    "happy": f"{emojize(':face_with_tears_of_joy:')} HAPPY",
    "angry": f"{emojize(':pouting_face:')} ANGRY",
    "fear": f"{emojize(':fearful_face:')} FEAR",
    "neutral": f"{emojize(':neutral_face:')} NEUTRAL",
    "sad": f"{emojize(':loudly_crying_face:')} SAD",
    "surprise": f"{emojize(':face_screaming_in_fear:')} SURPRISE",
    "disgust": f"{emojize(':nauseated_face:')} DISGUST",
}


def gen_frames():  # generate frame by frame from camera
    global face
    while camera.isOpened():
        success, frame = camera.read()
        if success:
            frame, face = get_face_from_frame(
                cv2.flip(frame, 1), face_shape, class_cascade=class_cascade
            )
            with contextlib.suppress(Exception):
                ret, buffer = cv2.imencode(".jpg", cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )


def magnify_emotion(emotion):
    return f"<p>{emotions_with_smiley[emotion[0]]} :{int(emotion[1] * 100)} %</p>"


def magnify_results(emotions):
    return "".join(list(map(magnify_emotion, emotions)))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/time_feed")
def time_feed():
    def generate():
        success, frame = camera.read()
        if success:
            frame, face = get_face_from_frame(
                cv2.flip(frame, 1), face_shape, class_cascade=class_cascade
            )
            emotions = get_emotions_from_face(face, model)
            yield magnify_results(
                emotions
            ) if emotions is not None else "no faces found"

    return Response(generate(), mimetype="text")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/requests", methods=["POST", "GET"])
def tasks():
    global switch, camera
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        if request.form.get("stop") == "Stop/Start":
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3134)
    camera.release()
    cv2.destroyAllWindows()
