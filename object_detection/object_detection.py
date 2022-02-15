import queue
import random
import threading
from abc import ABC
from functools import lru_cache
from typing import Any, List, Tuple, Union, cast

import cv2
import torch
from imutils.video import FPS
from object_detection.detection import Detection
from pytorchyolo import detect, models

from object_detection.capture import Capture
from object_detection.distance_calculator import DistanceCalculator
from object_detection.yolo_config import YoloConfig

from threading import Lock

CUDA_lock = Lock()


class ObjectDetection(ABC):
    def __init__(self, id, capture_source: Capture, yolo_cfg: YoloConfig):

        self.id = id

        self.cap = capture_source
        self.yolo_cfg = yolo_cfg
        self.distance_calculator = DistanceCalculator()

        self.confidence = float(0.5)
        self.nms_thesh = float(0.4)
        self.num_classes = 80

        self.model = models.load_model(
            self.yolo_cfg.cfg_file, self.yolo_cfg.weights_file
        )
        self.CUDA = torch.cuda.is_available()

        self.width = 640  # 640#
        self.height = 480  # 360#

        print("Loading network.....")
        if self.CUDA:
            self.model.cuda()
        print("Network successfully loaded")

        self.model.eval()

    def draw_on_image(self, detection: Detection, img):
        """
        Draws the bounding box over the objects that the model detects
        """

        # as a personal choice you can modify this to get distance as accurate as possible:
        # detection.x1 += 150
        # detection.y1 += 100
        # detection.x2 += 200
        # detection.y2 += 200

        label = self.yolo_cfg.classes[detection.class_index]
        color = self.yolo_cfg.colors[detection.class_index]

        # draw rectangle around detected object
        img = cv2.rectangle(
            img,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            1,
        )

        # draw rectangle for label
        cv2.rectangle(
            img,
            (detection.x1 - 2, detection.y2 + 25),
            (detection.x2 + 2, detection.y2),
            color,
            -1,
        )

        # write label to image
        img = cv2.putText(
            img,
            label,
            (detection.x1 + 2, detection.y2 + 20),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            [225, 255, 255],
            1,
        )

        # returns image with bounding box and label drawn on it
        return img

    def process_output(self, frame, output: torch.Tensor):

        # Output is a numpy array in the following format:
        # [[x1, y1, x2, y2, confidence, class]]

        for out in output:

            detection = Detection.from_output(out)

            self.draw_on_image(detection, frame)
            distance = self.distance_calculator.calc(*detection.bounding_box)
            feedback = f"{detection} is {distance} cm"
            print(feedback)

    def main(self):

        fps = FPS().start()

        frame = self.cap.get_frame()

        if frame is None:
            return ""

        # Localize the objects in a frame
        with CUDA_lock:
            output = detect.detect_image(
                self.model, frame, conf_thres=self.confidence, nms_thres=self.nms_thesh
            )
            self.process_output(frame, output)

        fps.update()
        fps.stop()

        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.1f}".format(fps.fps()))

        _, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tostring()
