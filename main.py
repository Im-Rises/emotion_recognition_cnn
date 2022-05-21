from datetime import datetime

import cv2
from flask import Flask, render_template, Response, request
from keras.models import load_model

from emotion_recognition.prediction import get_face_from_frame, get_emotions_from_face

switch, out, capture, rec_frame = (
    1,
    0,
    0,
    0,
)

face_shape = (80, 80)
model = load_model("./emotion_recognition/Models/trained_models/resnet50")
class_cascade = cv2.CascadeClassifier(
    "./emotion_recognition/ClassifierForOpenCV/frontalface_default.xml"
)
face = None
emotions = None

# instatiate flask app
app = Flask(__name__, template_folder="./templates")

camera = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    global face
    while camera.isOpened():
        success, frame = camera.read()
        if success:
            frame, face = get_face_from_frame(
                cv2.flip(frame, 1), face_shape, class_cascade=class_cascade
            )
            try:
                ret, buffer = cv2.imencode(".jpg", cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            except Exception as e:
                pass


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
            yield str(emotions) if emotions is not None else "no faces found"

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
    app.run()
    camera.release()
    cv2.destroyAllWindows()
