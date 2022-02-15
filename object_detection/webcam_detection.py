import torch
from object_detection.capture import Capture, WebcamCapture
from object_detection.object_detection import ObjectDetection
from object_detection.yolo_config import YoloConfig, YoloV3Config

torch.multiprocessing.set_start_method("spawn", force=True)

# ##  Setting up torch for gpu utilization
# if torch.cuda.is_available():
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
#     torch.set_default_tensor_type("torch.cuda.FloatTensor")


class WebcamObjectDetection(ObjectDetection):
    def __init__(
        self,
        id,
        capture_source: Capture = WebcamCapture(0),
        yolo_cfg: YoloConfig = YoloV3Config(),
    ):
        super().__init__(id, capture_source, yolo_cfg)

        self.confidence = float(0.6)
        self.nms_thesh = float(0.8)
        self.num_classes = 80

        self.width = 1280  # 640#1280
        self.height = 720  # 360#720


if __name__ == "__main__":
    id = 0
    WebcamObjectDetection(id).main()
