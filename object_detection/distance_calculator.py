from functools import lru_cache


@lru_cache
def _get_object_distance(x, y, w, h):
    distance = (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3
    distance = round(distance * 2.54, 1)

    return distance


class DistanceCalculator:
    def __init__(self) -> None:
        pass

    def calc(self, x, y, w, h):
        return _get_object_distance(x, y, w, h)
