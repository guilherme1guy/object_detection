from abc import ABC
from typing import List

import numpy as np
from pytorchyolo.utils.utils import load_classes


class YoloConfig(ABC):
    def __init__(self):
        self._cfgfile = ""
        self._weightsfile = ""
        self._classes = [""]
        self._colors = []

    @property
    def cfg_file(self):
        return self._cfgfile

    @property
    def weights_file(self):
        return self._weightsfile

    @property
    def classes(self):
        return self._classes

    @property
    def colors(self):
        return self._colors


class YoloV3Config(YoloConfig):
    def __init__(self):
        super().__init__()

        self._cfgfile = "cfg/yolov3.cfg"
        self._weightsfile = "weights/yolov3.weights"
        self._classes = load_classes("data/coco.names")
        self._colors = np.random.uniform(0, 255, size=(len(self._classes), 3))
