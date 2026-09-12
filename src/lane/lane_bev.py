"""Projective BEV helper. This is geometry normalization, not metric calibration."""

import cv2
import numpy as np


def default_homography(width, height, config):
    source = np.float32([
        [config["roi_left"] * width, config["roi_top_y"] * height],
        [config["roi_right"] * width, config["roi_top_y"] * height],
        [config["roi_right"] * width, config["roi_bottom_y"] * height],
        [config["roi_left"] * width, config["roi_bottom_y"] * height],
    ])
    target = np.float32([
        [0, 0], [config["bev_width"], 0],
        [config["bev_width"], config["bev_height"]], [0, config["bev_height"]],
    ])
    return cv2.getPerspectiveTransform(source, target)


def warp(frame, homography, config):
    return cv2.warpPerspective(
        frame,
        homography,
        (int(config["bev_width"]), int(config["bev_height"])),
    )


def project_points(points, homography):
    if not points:
        return []
    values = np.float32(points).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(values, homography).reshape(-1, 2).tolist()
