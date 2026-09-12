"""Fast classical lane-marking detector for forward road video."""

import cv2
import numpy as np

from lane_bev import default_homography, warp
from lane_confidence import lane_confidence, visibility_quality
from lane_geometry import fit_curve, lane_color, lane_style, polyline_from_coefficients


class ClassicalLaneDetector:
    def __init__(self, config):
        self.config = config
        self.homography = None

    def detect(self, frame, frame_index):
        height, width = frame.shape[:2]
        if self.homography is None:
            self.homography = default_homography(width, height, self.config)
        bev = warp(frame, self.homography, self.config)
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        white = cv2.inRange(hsv, np.array([0, 0, 135], dtype=np.uint8), np.array([180, int(self.config["white_saturation_max"]), 255], dtype=np.uint8))
        yellow = cv2.inRange(hsv, np.array([int(self.config["yellow_hue_low"]), 45, 90], dtype=np.uint8), np.array([int(self.config["yellow_hue_high"]), 255, 255], dtype=np.uint8))
        edges = cv2.Canny(gray, 45, 150)
        mask = cv2.bitwise_or(cv2.bitwise_or(white, yellow), edges)
        mask[:int(self.config["bev_height"] * 0.12)] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=28, minLineLength=28, maxLineGap=35)
        candidates = []
        if lines is not None:
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = map(float, line)
                if abs(y2 - y1) < 8:
                    continue
                slope = (x2 - x1) / (y2 - y1)
                if abs(slope) < 0.08 or abs(slope) > 4.0:
                    continue
                points = [[x1, y1], [x2, y2]]
                coefficients = fit_curve(points, 1)
                if coefficients is None:
                    continue
                sample_y = np.linspace(min(y1, y2), max(y1, y2), 12)
                x_values = np.polyval(coefficients, sample_y)
                valid = (x_values >= 0) & (x_values < self.config["bev_width"])
                support = float(valid.mean())
                if support < 0.35:
                    continue
                midpoint = float(np.polyval(coefficients, self.config["bev_height"] * 0.86))
                region = hsv[int(max(0, min(self.config["bev_height"] - 1, np.mean([y1, y2])))), int(max(0, min(self.config["bev_width"] - 1, np.mean([x1, x2]))))]
                color = lane_color(float(region[0]), self.config["yellow_hue_low"], self.config["yellow_hue_high"], float(region[1]))
                detection_conf = min(1.0, (abs(slope) / 1.8) * support)
                candidates.append({
                    "coefficients": coefficients,
                    "polyline": polyline_from_coefficients(coefficients, np.linspace(0, self.config["bev_height"], 20)),
                    "bev_polyline": polyline_from_coefficients(coefficients, np.linspace(0, self.config["bev_height"], 20)),
                    "detection_confidence": detection_conf,
                    "style": lane_style(support, self.config["solid_support_ratio"]),
                    "color": color,
                    "support": support,
                    "bottom_x": midpoint,
                    "visibility": "visible",
                })
        candidates.sort(key=lambda item: item["detection_confidence"], reverse=True)
        deduped = []
        for candidate in candidates:
            if not deduped or all(abs(candidate["bottom_x"] - item["bottom_x"]) > 70 for item in deduped):
                deduped.append(candidate)
            if len(deduped) >= 6:
                break
        deduped.sort(key=lambda item: item["bottom_x"])
        return {"frame": frame_index, "homography": self.homography, "lanes": deduped, "bev": bev}
