"""Temporal lane identity stitching with conservative ego-corridor selection."""

from copy import deepcopy

from lane_confidence import lane_confidence
from lane_geometry import curve_rmse, curve_x


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
        detections = sorted(
            (dict(item) for item in observation["lanes"]),
            key=lambda item: item["bottom_x"],
        )
        tracks = sorted(
            self.tracks.items(),
            key=lambda item: (item[1]["bottom_x"], item[0]),
        )
        matches = self._ordered_matches(tracks, detections)
        used_detections = set()
        assigned = []
        frame = observation["frame"]

        for track_index, detection_index, match_cost in matches:
            lane_id, track = tracks[track_index]
            candidate = detections[detection_index]
            used_detections.add(detection_index)
            was_missing = track["missing"] > 0
            temporal = max(0.0, 1.0 - match_cost)
            span_quality = min(
                1.0,
                candidate.get("span_ratio", 0.0) / 0.45,
            )
            candidate["lane_id"] = lane_id
            candidate["confidence"] = lane_confidence(
                candidate["detection_confidence"],
                temporal,
                candidate.get("geometry_quality", 0.0),
                span_quality,
            )
            track.update(candidate)
            track["missing"] = 0
            track["last_seen_timestamp"] = timestamp
            track["last_seen_frame"] = frame
            track["hits"] = track.get("hits", 1) + 1
            if was_missing:
                self.recovered += 1
            assigned.append(candidate)

        matched_track_indices = {track_index for track_index, _, _ in matches}
        for track_index, (lane_id, track) in enumerate(tracks):
            if track_index not in matched_track_indices:
                track["missing"] += 1

        for index, candidate in enumerate(detections):
            if index in used_detections:
                continue
            if candidate["detection_confidence"] < self.config["new_lane_min_confidence"]:
                continue
            lane_id = self.next_id
            self.next_id += 1
            self.created += 1
            span_quality = min(1.0, candidate.get("span_ratio", 0.0) / 0.45)
            candidate["lane_id"] = lane_id
            candidate["confidence"] = lane_confidence(
                candidate["detection_confidence"],
                candidate.get("geometry_quality", 0.0),
                span_quality,
            )
            self.tracks[lane_id] = dict(
                candidate,
                missing=0,
                hits=1,
                last_seen_timestamp=timestamp,
                last_seen_frame=frame,
            )
            assigned.append(candidate)

        prediction_decay = self.config["prediction_confidence_decay"]
        max_prediction_updates = self.config["max_prediction_updates"]
        matched_track_ids = {
            tracks[index][0] for index in matched_track_indices
        }
        for lane_id, track in tracks:
            if (
                lane_id in self.tracks
                and lane_id not in matched_track_ids
                and track["missing"] <= max_prediction_updates
            ):
                prediction = deepcopy(track)
                prediction["confidence"] *= prediction_decay ** track["missing"]
                prediction["visibility"] = "predicted"
                prediction["prediction_age_updates"] = track["missing"]
                assigned.append(prediction)

        expired = [
            lane_id
            for lane_id, track in self.tracks.items()
            if track["missing"] > self.config["max_missing_frames"]
        ]
        for lane_id in expired:
            del self.tracks[lane_id]

        assigned.sort(key=lambda item: item["bottom_x"])
        corridor = self._ego_corridor(assigned)
        if corridor:
            left_boundary, right_boundary = corridor
            left_x, right_x = left_boundary["bottom_x"], right_boundary["bottom_x"]
            center_x = self.config["bev_width"] * 0.5
            for lane in assigned:
                if lane["lane_id"] in (left_boundary["lane_id"], right_boundary["lane_id"]):
                    lane["ego_relation"] = "ego_lane"
                    lane["boundary_role"] = (
                        "left" if lane["lane_id"] == left_boundary["lane_id"] else "right"
                    )
                else:
                    lane["ego_relation"] = "left_lane" if lane["bottom_x"] < left_x else "right_lane"
                    lane["boundary_role"] = None
            confidence = min(
                left_boundary["confidence"],
                right_boundary["confidence"],
            )
            boundary_ids = (
                left_boundary["lane_id"],
                right_boundary["lane_id"],
            )
            offset = (left_x + right_x) * 0.5 - center_x
            boundary_distance = min(center_x - left_x, right_x - center_x)
        else:
            for lane in assigned:
                lane["ego_relation"] = "unknown"
                lane["boundary_role"] = None
            confidence = 0.0
            boundary_ids = None
            offset = None
            boundary_distance = None

        state = {
            "frame": observation["frame"],
            "timestamp": timestamp,
            "lanes": assigned,
            "ego_lane_id": None,
            "ego_lane_boundary_ids": boundary_ids,
            "left_lane_id": None,
            "right_lane_id": None,
            "ego_lane_offset": offset,
            "lane_boundary_distance": boundary_distance,
            "confidence": confidence,
        }
        self.last_state = state
        return state

    def hold(self, frame, timestamp):
        if self.last_state is None:
            return {
                "frame": frame,
                "timestamp": timestamp,
                "lanes": [],
                "ego_lane_id": None,
                "ego_lane_boundary_ids": None,
                "left_lane_id": None,
                "right_lane_id": None,
                "ego_lane_offset": None,
                "lane_boundary_distance": None,
                "confidence": 0.0,
            }

        state = deepcopy(self.last_state)
        age = frame - state["frame"]
        state["frame"] = frame
        state["timestamp"] = timestamp
        if age > self.config["max_prediction_frames"]:
            state["lanes"] = []
            state["ego_lane_boundary_ids"] = None
            state["ego_lane_offset"] = None
            state["lane_boundary_distance"] = None
            state["confidence"] = 0.0
        else:
            decay = self.config["prediction_confidence_decay"] ** age
            for lane in state["lanes"]:
                lane["confidence"] *= decay
                lane["visibility"] = "predicted"
            state["confidence"] *= decay
        return state

    def _ordered_matches(self, tracks, detections):
        track_count = len(tracks)
        detection_count = len(detections)
        skip_cost = 1.0
        costs = [[None] * detection_count for _ in range(track_count)]
        for track_index, (_, track) in enumerate(tracks):
            for detection_index, candidate in enumerate(detections):
                bottom_delta = abs(candidate["bottom_x"] - track["bottom_x"])
                curve_distance = curve_rmse(
                    candidate["coefficients"],
                    track["coefficients"],
                    [0.0, self.config["bev_height"] * 0.5, self.config["bev_height"]],
                )
                if (
                    bottom_delta > self.config["max_match_x_delta"]
                    or curve_distance > self.config["max_match_curve_rmse"]
                ):
                    continue
                costs[track_index][detection_index] = 0.5 * (
                    bottom_delta / self.config["max_match_x_delta"]
                    + curve_distance / self.config["max_match_curve_rmse"]
                )

        dp = [
            [float("inf")] * (detection_count + 1)
            for _ in range(track_count + 1)
        ]
        action = [
            [None] * (detection_count + 1)
            for _ in range(track_count + 1)
        ]
        dp[0][0] = 0.0
        for i in range(track_count + 1):
            for j in range(detection_count + 1):
                current = dp[i][j]
                if current == float("inf"):
                    continue
                if i < track_count and current + skip_cost < dp[i + 1][j]:
                    dp[i + 1][j] = current + skip_cost
                    action[i + 1][j] = "track"
                if j < detection_count and current + skip_cost < dp[i][j + 1]:
                    dp[i][j + 1] = current + skip_cost
                    action[i][j + 1] = "detection"
                if (
                    i < track_count
                    and j < detection_count
                    and costs[i][j] is not None
                    and current + costs[i][j] < dp[i + 1][j + 1]
                ):
                    dp[i + 1][j + 1] = current + costs[i][j]
                    action[i + 1][j + 1] = "match"

        matches = []
        i, j = track_count, detection_count
        while i or j:
            step = action[i][j]
            if step == "match":
                matches.append((i - 1, j - 1, costs[i - 1][j - 1]))
                i -= 1
                j -= 1
            elif step == "track":
                i -= 1
            elif step == "detection":
                j -= 1
            else:
                break
        return list(reversed(matches))

    def _ego_corridor(self, lanes):
        if len(lanes) < 2:
            return None
        center_x = self.config["bev_width"] * 0.5
        reference_y = self.config["bev_height"] * 0.86
        candidates = []
        for left_index, left in enumerate(lanes[:-1]):
            if left.get("support_y", [0.0, 0.0])[1] < reference_y - self.config["bev_height"] * 0.08:
                continue
            if left.get("visibility") != "visible":
                continue
            if (
                self.tracks.get(left["lane_id"], {}).get("hits", 0)
                < self.config["min_ego_lane_observations"]
                or left["confidence"] < self.config["min_ego_lane_confidence"]
            ):
                continue
            for right in lanes[left_index + 1:]:
                if right.get("support_y", [0.0, 0.0])[1] < reference_y - self.config["bev_height"] * 0.08:
                    continue
                if right.get("visibility") != "visible":
                    continue
                if (
                    self.tracks.get(right["lane_id"], {}).get("hits", 0)
                    < self.config["min_ego_lane_observations"]
                    or right["confidence"] < self.config["min_ego_lane_confidence"]
                ):
                    continue
                left_x = left["bottom_x"]
                right_x = right["bottom_x"]
                width = right_x - left_x
                widths = [
                    curve_x(right["coefficients"], y) - curve_x(left["coefficients"], y)
                    for y in (
                        self.config["bev_height"] * 0.72,
                        reference_y,
                        self.config["bev_height"] * 0.98,
                    )
                ]
                mean_width = sum(widths) / len(widths)
                width_variation = (
                    (max(widths) - min(widths)) / mean_width
                    if mean_width > 0
                    else float("inf")
                )
                midpoint_error = abs((left_x + right_x) * 0.5 - center_x)
                if (
                    left_x < center_x < right_x
                    and self.config["ego_lane_min_width"] <= width <= self.config["ego_lane_max_width"]
                    and midpoint_error <= self.config["ego_lane_center_tolerance"]
                    and all(
                        self.config["ego_lane_min_width"] <= lane_width <= self.config["ego_lane_max_width"]
                        for lane_width in widths
                    )
                    and width_variation <= self.config["ego_lane_max_width_variation_ratio"]
                ):
                    candidates.append((midpoint_error, width, left, right))
        if not candidates:
            return None
        _, _, left, right = min(candidates, key=lambda item: (item[0], item[1]))
        return left, right
