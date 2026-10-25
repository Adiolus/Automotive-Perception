"""Lane confidence components and visibility quality."""


def lane_confidence(*evidence):
    """Cap confidence at the weakest observed lane-evidence component."""
    if not evidence:
        return 0.0
    return max(0.0, min(1.0, min(evidence)))


def visibility_quality(pixel_count, occluded=False):
    quality = min(1.0, pixel_count / 1800.0)
    return quality * (0.55 if occluded else 1.0)
