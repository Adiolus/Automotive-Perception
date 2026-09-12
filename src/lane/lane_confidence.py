"""Lane confidence components and visibility quality."""


def lane_confidence(detection, temporal, geometry, bev, continuity, visibility):
    values = [detection, temporal, geometry, bev, continuity, visibility]
    return max(0.0, min(1.0, sum(values) / len(values)))


def visibility_quality(pixel_count, occluded=False):
    quality = min(1.0, pixel_count / 1800.0)
    return quality * (0.55 if occluded else 1.0)
