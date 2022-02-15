from object_detection.capture import Capture
from object_detection.object_detection import ObjectDetection
from object_detection.yolo_config import YoloConfig


class VideoCaptureObjectDetection(ObjectDetection):
    def __init__(self, id, capture_source: Capture, yolo_cfg: YoloConfig):
        super().__init__(id, capture_source, yolo_cfg)

        self.confidence = float(0.5)
        self.nms_thesh = float(0.4)
        self.num_classes = 80

        self.width = 640  # 640#
        self.height = 480  # 360#
