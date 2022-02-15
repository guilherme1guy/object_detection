from abc import ABC, abstractmethod

import cv2
from imutils.video import WebcamVideoStream


class Capture(ABC):
    @abstractmethod
    def get_frame(self):
        raise NotImplementedError()


class WebcamCapture(Capture):
    def __init__(self, src):
        self.cap = WebcamVideoStream(src=src).start()

    def get_frame(self):
        return self.cap.read()


class VideoStreamCapture(Capture):
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.frame = None

    def get_frame(self):
        status, frame = self.cap.read()

        if status:
            self.frame = frame

        return self.frame
