"""Canonical lane geometry and lane/object relation helpers."""

import math
import numpy as np


def polyline_from_coefficients(coefficients, y_values):
    return [[float(np.polyval(coefficients, y)), float(y)] for y in y_values]


def fit_curve(points, degree=2):
    if len(points) < degree + 1:
        return None
    points = np.asarray(points, dtype=np.float32)
    return np.polyfit(points[:, 1], points[:, 0], degree).tolist()


def curve_x(coefficients, y):
    return float(np.polyval(np.asarray(coefficients), y))


def curve_rmse(first, second, y_values):
    if not first or not second:
        return float("inf")
    errors = [curve_x(first, y) - curve_x(second, y) for y in y_values]
    return float(math.sqrt(sum(error * error for error in errors) / len(errors)))


def lane_style(support_ratio, threshold):
    return "solid" if support_ratio >= threshold else "dashed"


def lane_color(hue_fraction, yellow_low, yellow_high, saturation):
    if saturation >= 45 and yellow_low <= hue_fraction <= yellow_high:
        return "yellow"
    if saturation < 90:
        return "white"
    return "unknown"


def get_lane_relation(track_bbox, lane_state):
    """Return object-to-lane relation using bottom-center image geometry."""
    x1, _, x2, y2 = track_bbox
    center_x = (x1 + x2) * 0.5
    candidates = []
    for lane in lane_state.get("lanes", []):
        x = curve_x(lane["coefficients"], y2)
        candidates.append((abs(center_x - x), lane, center_x - x))
    if not candidates:
        return {
            "lane_id": None, "lane_relation": "unknown", "lateral_offset": None,
            "lane_center_distance": None, "boundary_distance": None,
            "lane_conflict": None, "lane_confidence": None,
        }
    _, nearest, signed = min(candidates, key=lambda item: item[0])
    relation = nearest.get("ego_relation", "unknown")
    return {
        "lane_id": nearest["lane_id"],
        "lane_relation": relation,
        "lateral_offset": signed,
        "lane_center_distance": abs(signed),
        "boundary_distance": abs(signed),
        "lane_conflict": False,
        "lane_confidence": nearest["confidence"],
    }
