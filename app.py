import socket

from flask import Flask, render_template
from flask.wrappers import Response
from object_detection.capture import VideoStreamCapture

from object_detection.webcam_detection import WebcamObjectDetection
from object_detection.video_capture_detection import VideoCaptureObjectDetection
from object_detection.yolo_config import YoloV3Config

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")


def gen(camera):

    display_frame = ""

    while True:
        frame = camera.main()
        if frame != "":
            display_frame = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + display_frame + b"\r\n\r\n"
            )


@app.route("/video_feed")
def video_feed():
    return Response(
        gen(
            VideoCaptureObjectDetection(
                0, VideoStreamCapture("dog-cycle-car.png"), YoloV3Config()
            )
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    # Serve the app with gevent
    app.run(host="0.0.0.0", threaded=True, debug=True)
