import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "lane"))

from lane_bev import default_homography
from lane_confidence import lane_confidence
from lane_detector import ClassicalLaneDetector
from lane_stitcher import LaneStitcher
from lane_tracker import LaneTracker
from run_lane import config


CONFIG_PATH = Path("src/lane/lane_config.yaml")


def lane_config():
    return config(CONFIG_PATH)


def candidate(bottom_x, confidence=0.9, y_extent=(80.0, 400.0), slope=0.0):
    coefficients = [float(slope), float(bottom_x) - slope * 412.0]
    y_values = np.linspace(*y_extent, 12)
    span_ratio = (y_extent[1] - y_extent[0]) / 480.0
    return {
        "coefficients": coefficients,
        "polyline": [[float(np.polyval(coefficients, y)), float(y)] for y in y_values],
        "bev_polyline": [[float(np.polyval(coefficients, y)), float(y)] for y in y_values],
        "detection_confidence": confidence,
        "geometry_quality": 0.9,
        "style": "solid",
        "color": "white",
        "support": 0.8,
        "span_ratio": span_ratio,
        "support_y": list(y_extent),
        "bottom_x": float(bottom_x),
        "visibility": "visible" if span_ratio >= 0.45 else "partial",
    }


def source_from_bev(bev, cfg):
    homography = default_homography(1280, 720, cfg)
    inverse = cv2.invert(homography)[1]
    return cv2.warpPerspective(bev, inverse, (1280, 720))


def lane_points(bottom_x):
    y_values = np.linspace(0.45 * 480, 0.98 * 480, 24)
    return np.column_stack((np.full_like(y_values, bottom_x), y_values))


def test_bev_maps_a_trapezoidal_road_roi_to_pixel_bounds():
    cfg = lane_config()
    homography = default_homography(1280, 720, cfg)
    source = np.float32([
        [cfg["roi_top_left"] * 1280, cfg["roi_top_y"] * 720],
        [cfg["roi_top_right"] * 1280, cfg["roi_top_y"] * 720],
        [cfg["roi_bottom_right"] * 1280, cfg["roi_bottom_y"] * 720],
        [cfg["roi_bottom_left"] * 1280, cfg["roi_bottom_y"] * 720],
    ]).reshape(-1, 1, 2)

    mapped = cv2.perspectiveTransform(source, homography).reshape(-1, 2)

    np.testing.assert_allclose(
        mapped,
        [[0, 0], [639, 0], [639, 479], [0, 479]],
        atol=1e-3,
    )


def test_detector_keeps_a_marking_segment_inside_its_observed_vertical_span():
    cfg = lane_config()
    bev = np.full((480, 640, 3), 70, dtype=np.uint8)
    cv2.line(bev, (260, 150), (260, 455), (240, 240, 240), 8)
    frame = source_from_bev(bev, cfg)

    lanes = ClassicalLaneDetector(cfg).detect(frame, 1)["lanes"]

    assert lanes
    lane = min(lanes, key=lambda item: abs(item["bottom_x"] - 260))
    ys = [point[1] for point in lane["polyline"]]
    assert min(ys) >= 140
    assert max(ys) <= 465
    assert all(0 <= point[0] < cfg["bev_width"] for point in lane["polyline"])


def test_detector_does_not_extend_an_upper_occluded_fragment_into_the_road():
    cfg = lane_config()
    bev = np.full((480, 640, 3), 70, dtype=np.uint8)
    cv2.line(bev, (260, 120), (260, 280), (240, 240, 240), 8)
    frame = source_from_bev(bev, cfg)

    lanes = ClassicalLaneDetector(cfg).detect(frame, 1)["lanes"]

    assert lanes == []


def test_detector_rejects_uncolored_road_texture_edges():
    cfg = lane_config()
    frame = np.full((720, 1280, 3), 80, dtype=np.uint8)
    cv2.line(frame, (360, 690), (880, 350), (255, 0, 0), 8)

    lanes = ClassicalLaneDetector(cfg).detect(frame, 1)["lanes"]

    assert lanes == []


def test_detector_rejects_a_steep_marking_shaped_artifact():
    cfg = lane_config()
    bev = np.full((480, 640, 3), 70, dtype=np.uint8)
    cv2.line(bev, (100, 450), (500, 150), (240, 240, 240), 8)
    frame = source_from_bev(bev, cfg)

    lanes = ClassicalLaneDetector(cfg).detect(frame, 1)["lanes"]

    assert lanes == []


def test_detector_rejects_wide_bright_panel_edges_as_lane_markings():
    cfg = lane_config()
    bev = np.full((480, 640, 3), 70, dtype=np.uint8)
    cv2.rectangle(bev, (250, 300), (370, 455), (240, 240, 240), -1)
    frame = source_from_bev(bev, cfg)

    lanes = ClassicalLaneDetector(cfg).detect(frame, 1)["lanes"]

    assert lanes == []


def test_lane_confidence_is_capped_by_weakest_evidence():
    assert lane_confidence(0.95, 0.9, 0.8, 0.7, 0.6, 0.25) == 0.25


def test_stitcher_preserves_left_to_right_ids_when_close_candidates_reorder():
    cfg = lane_config()
    cfg["ego_lane_min_width"] = 70.0
    cfg["ego_lane_max_width"] = 260.0
    stitcher = LaneStitcher(cfg)
    first = stitcher.update(
        {"frame": 1, "lanes": [candidate(270), candidate(370)]},
        0.0,
    )
    original_ids = [lane["lane_id"] for lane in first["lanes"]]

    second = stitcher.update(
        {"frame": 2, "lanes": [candidate(340), candidate(300)]},
        0.033,
    )

    assert [lane["lane_id"] for lane in second["lanes"]] == original_ids
    assert [lane["bottom_x"] for lane in second["lanes"]] == [300.0, 340.0]
    assert second["lanes"][0]["lane_id"] < second["lanes"][1]["lane_id"]


def test_stitcher_does_not_reuse_a_track_for_a_neighbor_after_one_frame_gap():
    cfg = lane_config()
    stitcher = LaneStitcher(cfg)
    first = stitcher.update(
        {"frame": 1, "lanes": [candidate(250), candidate(350)]},
        0.0,
    )
    left_id, right_id = [lane["lane_id"] for lane in first["lanes"]]

    second = stitcher.update(
        {"frame": 2, "lanes": [candidate(340)]},
        0.033,
    )

    assert len(second["lanes"]) == 2
    predicted_left = next(lane for lane in second["lanes"] if lane["lane_id"] == left_id)
    visible_right = next(lane for lane in second["lanes"] if lane["lane_id"] == right_id)
    assert predicted_left["visibility"] == "predicted"
    assert visible_right["visibility"] == "visible"
    assert visible_right["lane_id"] != left_id


def test_stitcher_emits_only_bounded_decaying_predictions_for_missing_lanes():
    cfg = lane_config()
    stitcher = LaneStitcher(cfg)
    first = stitcher.update(
        {"frame": 1, "lanes": [candidate(250), candidate(350)]},
        1 / 30.0,
    )
    left_id, right_id = [lane["lane_id"] for lane in first["lanes"]]
    left_confidence = first["lanes"][0]["confidence"]

    one_missing = stitcher.update(
        {"frame": 4, "lanes": [candidate(350)]},
        4 / 30.0,
    )
    predicted_left = next(lane for lane in one_missing["lanes"] if lane["lane_id"] == left_id)
    visible_right = next(lane for lane in one_missing["lanes"] if lane["lane_id"] == right_id)

    assert predicted_left["visibility"] == "predicted"
    assert predicted_left["prediction_age_updates"] == 1
    assert predicted_left["confidence"] < left_confidence
    assert visible_right["visibility"] == "visible"

    two_missing = stitcher.update(
        {"frame": 7, "lanes": []},
        7 / 30.0,
    )
    assert all(lane["visibility"] == "predicted" for lane in two_missing["lanes"])
    twice_predicted_left = next(lane for lane in two_missing["lanes"] if lane["lane_id"] == left_id)
    assert twice_predicted_left["confidence"] < predicted_left["confidence"]

    expired_prediction = stitcher.update(
        {"frame": 10, "lanes": []},
        10 / 30.0,
    )
    assert all(lane["prediction_age_updates"] <= cfg["max_prediction_updates"] for lane in expired_prediction["lanes"])
    assert left_id not in {lane["lane_id"] for lane in expired_prediction["lanes"]}


def test_lane_tracker_uses_gated_ordered_one_to_one_association():
    tracker = LaneTracker(640, 480, max_lanes=2, match_threshold=0.14)
    tracker.update([lane_points(230), lane_points(400)])
    original = tracker.update([lane_points(230), lane_points(400)])
    original_ids = [lane.lane_id for lane in original]

    reordered_input = tracker.update([lane_points(390), lane_points(240)])

    assert [lane.lane_id for lane in reordered_input] == original_ids
    assert [tracker._bottom_x(lane.curve) for lane in reordered_input] == sorted(
        tracker._bottom_x(lane.curve) for lane in reordered_input
    )
    assert tracker.total_matches == 4

    tracker.update([lane_points(520), lane_points(600)])

    assert tracker.total_matches == 4


def test_ego_lane_requires_a_plausible_pair_bracketing_image_center():
    cfg = lane_config()
    cfg["ego_lane_min_width"] = 70.0
    cfg["ego_lane_max_width"] = 260.0
    stitcher = LaneStitcher(cfg)

    no_pair = stitcher.update(
        {"frame": 1, "lanes": [candidate(80), candidate(170), candidate(520)]},
        0.0,
    )

    assert no_pair["ego_lane_id"] is None
    assert no_pair["ego_lane_boundary_ids"] is None
    assert no_pair["confidence"] == 0.0

    paired = None
    for frame in (2, 3, 4):
        paired = stitcher.update(
            {"frame": frame, "lanes": [candidate(80), candidate(260), candidate(380), candidate(540)]},
            frame / 30.0,
        )

    assert paired is not None
    assert paired["ego_lane_id"] is None
    assert paired["ego_lane_boundary_ids"] == (
        paired["lanes"][1]["lane_id"],
        paired["lanes"][2]["lane_id"],
    )
    assert paired["ego_lane_offset"] == 0.0


def test_ego_corridor_rejects_boundaries_with_unstable_width():
    cfg = lane_config()
    stitcher = LaneStitcher(cfg)

    for frame in range(1, 5):
        state = stitcher.update(
            {
                "frame": frame,
                "lanes": [
                    candidate(260),
                    candidate(380, slope=0.5),
                ],
            },
            frame / 30.0,
        )

    assert state["ego_lane_boundary_ids"] is None


def test_ego_corridor_stays_uncertain_when_a_boundary_is_only_partial():
    cfg = lane_config()
    stitcher = LaneStitcher(cfg)

    for frame in range(1, 5):
        state = stitcher.update(
            {
                "frame": frame,
                "lanes": [
                    candidate(260, y_extent=(200.0, 414.0)),
                    candidate(380),
                ],
            },
            frame / 30.0,
        )

    assert state["ego_lane_boundary_ids"] is None


def test_held_lane_state_is_marked_predicted_and_loses_freshness_confidence():
    cfg = lane_config()
    cfg["ego_lane_min_width"] = 70.0
    cfg["ego_lane_max_width"] = 260.0
    stitcher = LaneStitcher(cfg)
    detected = stitcher.update(
        {"frame": 1, "lanes": [candidate(260), candidate(380)]},
        0.033,
    )

    held = stitcher.hold(frame=2, timestamp=0.066)

    assert held["frame"] == 2
    assert held["lanes"][0]["visibility"] == "predicted"
    assert held["lanes"][0]["confidence"] < detected["lanes"][0]["confidence"]
    assert detected["lanes"][0]["visibility"] == "visible"
