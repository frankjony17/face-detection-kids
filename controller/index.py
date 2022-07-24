from flask import Flask, render_template, Response
from service.face_detection_v1 import FaceDetectionService
import cv2
import os

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    fd = FaceDetectionService()
    fd.detection()
    frame = fd.processing_frame()

    return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_frame')
def get_frame():
    if os.path.exists("controller/static/images/face.jpg"):
        os.remove("controller/static/images/face.jpg")

    while True:
        image = cv2.imread('controller/static/images/frame.jpg')
        if image.shape:
            height, _, _ = image.shape
            if height == 480:
                cv2.imwrite("controller/static/images/face.jpg", image)
                break

    frame = (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + open('controller/static/images/face.jpg',
                                                                    'rb').read() + b'\r\n')
    return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
