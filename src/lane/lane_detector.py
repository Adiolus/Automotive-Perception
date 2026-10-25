"""Fast classical lane-marking detector for forward road video."""

import cv2
import numpy as np

from lane_bev import default_homography, warp
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
        marking_mask = cv2.bitwise_or(white, yellow)
        edges = cv2.Canny(gray, 45, 150)
        color_support = cv2.dilate(marking_mask, np.ones((5, 5), np.uint8))
        mask = cv2.bitwise_and(edges, color_support)
        mask[:int(self.config["bev_height"] * 0.12)] = 0
        lines = cv2.HoughLinesP(
            mask,
            1,
            np.pi / 180,
            threshold=28,
            minLineLength=int(self.config["min_lane_segment_px"]),
            maxLineGap=25,
        )
        candidates = []
        if lines is not None:
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = map(float, line)
                y_min, y_max = sorted((y1, y2))
                span_y = y_max - y_min
                if (
                    span_y < self.config["min_lane_segment_px"]
                    or y_max < self.config["bev_height"] * self.config["min_lane_bottom_y_ratio"]
                ):
                    continue
                slope = (x2 - x1) / (y2 - y1)
                max_slope = self.config["max_lane_dx_per_dy"]
                if abs(slope) > max_slope:
                    continue
                points = [[x1, y1], [x2, y2]]
                coefficients = fit_curve(points, 1)
                if coefficients is None:
                    continue
                span_ratio = span_y / self.config["bev_height"]
                if span_ratio < self.config["min_lane_span_ratio"]:
                    continue
                sample_y = np.linspace(y_min, y_max, 20)
                x_values = np.polyval(coefficients, sample_y)
                if np.any((x_values < 0) | (x_values >= self.config["bev_width"])):
                    continue
                support = self._line_support(mask, x_values, sample_y)
                if support < 0.55:
                    continue
                marking_width = self._marking_width(
                    marking_mask,
                    x_values,
                    sample_y,
                    slope,
                )
                if marking_width > self.config["max_marking_width_px"]:
                    continue
                reference_y = self.config["bev_height"] * 0.86
                bottom_x = float(np.polyval(coefficients, reference_y))
                if not -0.1 * self.config["bev_width"] <= bottom_x <= 1.1 * self.config["bev_width"]:
                    continue
                midpoint_x = float(np.polyval(coefficients, (y_min + y_max) * 0.5))
                region = hsv[
                    int(np.clip((y_min + y_max) * 0.5, 0, self.config["bev_height"] - 1)),
                    int(np.clip(midpoint_x, 0, self.config["bev_width"] - 1)),
                ]
                color = lane_color(float(region[0]), self.config["yellow_hue_low"], self.config["yellow_hue_high"], float(region[1]))
                span_quality = min(1.0, span_ratio / 0.45)
                geometry_quality = max(0.0, 1.0 - abs(slope) / max_slope)
                detection_conf = support * span_quality * geometry_quality
                polyline_y = np.linspace(y_min, y_max, 20)
                candidates.append({
                    "coefficients": coefficients,
                    "polyline": polyline_from_coefficients(coefficients, polyline_y),
                    "bev_polyline": polyline_from_coefficients(coefficients, polyline_y),
                    "detection_confidence": detection_conf,
                    "style": lane_style(support, self.config["solid_support_ratio"]),
                    "color": color,
                    "support": support,
                    "marking_width": marking_width,
                    "span_ratio": span_ratio,
                    "support_y": [y_min, y_max],
                    "geometry_quality": geometry_quality,
                    "bottom_x": bottom_x,
                    "visibility": "visible" if span_ratio >= 0.45 else "partial",
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

    @staticmethod
    def _line_support(mask, x_values, y_values):
        supported = 0
        for x, y in zip(x_values, y_values):
            cx, cy = int(round(x)), int(round(y))
            patch = mask[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
            supported += bool(np.any(patch))
        return supported / max(len(x_values), 1)

    @staticmethod
    def _marking_width(mask, x_values, y_values, slope):
        normal_x = 1.0 / np.sqrt(1.0 + slope * slope)
        normal_y = -slope * normal_x
        offsets = np.arange(-32, 33)
        widths = []
        for x, y in zip(x_values, y_values):
            xs = np.rint(x + offsets * normal_x).astype(int)
            ys = np.rint(y + offsets * normal_y).astype(int)
            valid = (
                (xs >= 0) & (xs < mask.shape[1])
                & (ys >= 0) & (ys < mask.shape[0])
            )
            samples = np.zeros(len(offsets), dtype=bool)
            samples[valid] = mask[ys[valid], xs[valid]] > 0
            longest = current = 0
            for present in samples:
                current = current + 1 if present else 0
                longest = max(longest, current)
            widths.append(longest)
        return float(np.median(widths)) if widths else 0.0
