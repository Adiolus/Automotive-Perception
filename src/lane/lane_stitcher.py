"""Temporal lane identity stitching with missing-frame persistence."""

import math
from lane_confidence import lane_confidence, visibility_quality
from lane_geometry import curve_rmse


class LaneStitcher:
    def __init__(self, config):
        self.config = config
        self.next_id = 1
        self.tracks = {}
        self.last_state = None
        self.switches = 0
        self.created = 0
        self.recovered = 0

    def update(self, observation, timestamp):
        detections = observation["lanes"]
        used = set()
        assigned = []
        y_values = [0.0, self.config["bev_height"] * 0.5, float(self.config["bev_height"])]
        for lane_id, track in list(self.tracks.items()):
            best = None
            best_cost = float("inf")
            for index, candidate in enumerate(detections):
                if index in used:
                    continue
                cost = abs(candidate["bottom_x"] - track["bottom_x"])
                cost += curve_rmse(candidate["coefficients"], track["coefficients"], y_values)
                if cost < best_cost:
                    best_cost, best = cost, (index, candidate)
            if best and best_cost <= self.config["match_x_threshold"] + self.config["match_curve_rmse"]:
                index, candidate = best
                used.add(index)
                temporal = max(0.0, 1.0 - best_cost / (self.config["match_x_threshold"] + self.config["match_curve_rmse"]))
                candidate["lane_id"] = lane_id
                candidate["ego_relation"] = self._relation(candidate["bottom_x"])
                candidate["confidence"] = lane_confidence(candidate["detection_confidence"], temporal, 0.7, 0.7, temporal, visibility_quality(candidate["support"] * 2000))
                track.update(candidate)
                track["missing"] = 0
                assigned.append(candidate)
            else:
                track["missing"] += 1
        for index, candidate in enumerate(detections):
            if index in used or candidate["detection_confidence"] < self.config["new_lane_min_confidence"]:
                continue
            lane_id = self.next_id
            self.next_id += 1
            self.created += 1
            candidate["lane_id"] = lane_id
            candidate["ego_relation"] = self._relation(candidate["bottom_x"])
            candidate["confidence"] = candidate["detection_confidence"]
            self.tracks[lane_id] = dict(candidate, missing=0)
            assigned.append(candidate)
        expired = [lane_id for lane_id, track in self.tracks.items() if track["missing"] > self.config["max_missing_frames"]]
        for lane_id in expired:
            del self.tracks[lane_id]
        assigned.sort(key=lambda item: item["bottom_x"])
        for index, lane in enumerate(assigned):
            lane["ego_relation"] = "ego_lane" if index == len(assigned) // 2 else ("left_lane" if index < len(assigned) // 2 else "right_lane")
        state = {
            "frame": observation["frame"], "timestamp": timestamp, "lanes": assigned,
            "ego_lane_id": assigned[len(assigned) // 2]["lane_id"] if assigned else None,
            "left_lane_id": assigned[len(assigned) // 2 - 1]["lane_id"] if len(assigned) >= 3 else None,
            "right_lane_id": assigned[len(assigned) // 2 + 1]["lane_id"] if len(assigned) >= 3 else None,
            "ego_lane_offset": 0.0 if assigned else None,
            "lane_boundary_distance": None,
            "confidence": sum(lane["confidence"] for lane in assigned) / len(assigned) if assigned else 0.0,
        }
        self.last_state = state
        return state

    def _relation(self, bottom_x):
        return "ego_lane" if abs(bottom_x - self.config["bev_width"] * 0.5) < self.config["bev_width"] * 0.2 else ("left_lane" if bottom_x < self.config["bev_width"] * 0.5 else "right_lane")
