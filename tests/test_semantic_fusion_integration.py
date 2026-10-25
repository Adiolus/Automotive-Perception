import importlib
import queue
import sys
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.fusion.class_mapping import CLASS_IDS, ROAD_CLASSES
from src.fusion.fusion import FusionConfig, FusionEngine
from src.fusion.fusion_types import DepthObservation, TrackingObservation
from src.fusion.semantic_fusion_interface import SceneFusionInterface, SegmentationWorker
from src.fusion.semantic_object_fusion import SemanticObjectFusion
from src.risk.risk_engine import RiskEngine, parse_config
from src.pipeline.realtime_pipeline import VisualizationWorker, overlay


def make_track(track_id="17", bbox=(2, 2, 6, 7), yolo_class="person"):
    return {"track_id": track_id, "bbox": list(bbox), "yolo_class": yolo_class}


def test_idd_labels_do_not_replace_yolo_class_and_road_ids_are_mapped():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = CLASS_IDS["car"]
    result = SemanticObjectFusion().fuse_object(
        make_track(), mask, segmentation_age_frames=0
    )

    assert result["yolo_class"] == "person"
    assert result["semantic_class_id"] == CLASS_IDS["car"]
    assert result["semantic_class_name"] == "car"
    assert CLASS_IDS["sidewalk"] == 3
    assert ROAD_CLASSES == {CLASS_IDS["road"], CLASS_IDS["drivable fallback"]}
    assert result["road_contact"] is True


def test_bbox_is_clipped_and_mask_and_depth_crops_use_image_coordinates():
    mask = np.full((4, 4), CLASS_IDS["sidewalk"], dtype=np.uint8)
    mask[0:3, 0:3] = CLASS_IDS["person"]
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    depth[0, 0] = np.nan

    result = SemanticObjectFusion().fuse_object(
        make_track(bbox=(-2, 0, 3, 3)), mask, depth
    )

    assert result["bbox"] == [0, 0, 3, 3]
    assert result["semantic_distribution"] == {
        CLASS_IDS["person"]: 1.0
    }
    assert result["depth_valid"] is True
    assert result["depth_value"] == 5.5


def test_empty_or_missing_segmentation_is_explicitly_unavailable():
    fusion = SemanticObjectFusion()
    missing = fusion.fuse_object(make_track(), None)
    empty = fusion.fuse_object(
        make_track(bbox=(20, 20, 30, 30)),
        np.zeros((8, 8), dtype=np.uint8),
    )

    assert missing["context_valid"] is False
    assert missing["semantic_class_id"] is None
    assert missing["semantic_confidence"] == 0.0
    assert missing["depth_valid"] is False
    assert empty["context_valid"] is False
    assert empty["semantic_distribution"] == {}


def test_depth_region_ignores_nonfinite_values_and_rejects_bad_shape():
    mask = np.full((8, 8), CLASS_IDS["road"], dtype=np.uint8)
    depth = np.full((8, 8), np.nan, dtype=np.float32)
    fusion = SemanticObjectFusion()
    invalid = fusion.fuse_object(make_track(), mask, depth)

    assert invalid["depth_valid"] is False
    with pytest.raises(ValueError, match="same-size 2D"):
        fusion.fuse_object(make_track(), mask, np.zeros((4, 4), dtype=np.float32))


def test_temporal_history_is_bounded_by_track_count():
    fusion = SemanticObjectFusion(history_frames=3, max_tracks=2)
    mask = np.full((8, 8), CLASS_IDS["car"], dtype=np.uint8)
    for track_id in ("a", "b", "c"):
        fusion.fuse_object(make_track(track_id=track_id), mask)

    assert len(fusion.track_history) == 2
    assert "a" not in fusion.track_history


def test_scene_interface_preserves_frame_identity_and_decays_stale_context():
    interface = SceneFusionInterface(max_segmentation_age_frames=4)
    mask = np.full((8, 8), CLASS_IDS["car"], dtype=np.uint8)
    fresh = interface.process(
        12, 0.4, None, [make_track()],
        {"frame_id": 12, "timestamp": 0.4, "mask": mask, "valid": True},
        None,
    )
    stale = interface.process(
        14, 0.466, None, [make_track()],
        {"frame_id": 12, "timestamp": 0.4, "mask": mask, "valid": True},
        None,
    )
    expired = interface.process(
        20, 0.667, None, [make_track()],
        {"frame_id": 12, "timestamp": 0.4, "mask": mask, "valid": True},
        None,
    )

    assert fresh["object_contexts"][0]["frame_id"] == 12
    assert fresh["object_contexts"][0]["timestamp"] == 0.4
    assert fresh["object_contexts"][0]["semantic_confidence"] == 1.0
    assert stale["object_contexts"][0]["segmentation_age_frames"] == 2
    assert stale["object_contexts"][0]["semantic_confidence"] < 1.0
    assert expired["object_contexts"][0]["context_valid"] is False
    assert interface.metrics["segmentation_stale_records"] == 1
    assert interface.metrics["segmentation_expired_records"] == 1


def test_scene_interface_rejects_future_and_invalid_masks():
    interface = SceneFusionInterface()
    track = [make_track()]
    future = interface.process(
        3, 0.1, None, track,
        {"frame_id": 4, "mask": np.zeros((8, 8), dtype=np.uint8), "valid": True},
        None,
    )
    invalid = interface.process(
        4, 0.133, None, track,
        {"frame_id": 4, "mask": np.zeros((8, 8, 3), dtype=np.uint8), "valid": True},
        None,
    )

    assert future["object_contexts"][0]["context_valid"] is False
    assert future["object_contexts"][0]["segmentation_age_frames"] == -1
    assert invalid["object_contexts"][0]["context_valid"] is False


def test_fusion_preserves_track_fields_and_disables_relative_depth_ttc():
    engine = FusionEngine(FusionConfig())
    engine.add_depth(DepthObservation(
        frame=1,
        timestamp=0.1,
        track_id="17",
        depth=2.0,
        quality=0.8,
        depth_change=0.1,
        trend="approaching",
    ))
    tracking = TrackingObservation(
        frame=2,
        timestamp=0.2,
        track_id="17",
        class_name="car",
        bbox=[1.0, 2.0, 5.0, 8.0],
        confidence=0.9,
        velocity_x=0.0,
        velocity_y=0.0,
    )

    result = engine.fuse(tracking).to_dict()

    assert result["frame"] == 2
    assert result["track_id"] == "17"
    assert result["class_name"] == "car"
    assert result["bbox"] == [1.0, 2.0, 5.0, 8.0]
    assert result["depth_type"] == "relative"
    assert result["ttc"] is None
    assert result["ttc_valid"] is False
    assert result["ttc_reason"] == "non_metric_depth"


def test_fusion_accepts_ttc_only_for_metric_depth_with_per_second_change():
    engine = FusionEngine(FusionConfig())
    engine.add_depth(DepthObservation(
        frame=1,
        timestamp=0.1,
        track_id="17",
        depth=20.0,
        quality=0.8,
        depth_change=-2.0,
        depth_type="metric",
        depth_change_per_second=True,
    ))
    tracking = TrackingObservation(
        frame=2,
        timestamp=0.2,
        track_id="17",
        class_name="car",
        bbox=[1.0, 2.0, 5.0, 8.0],
        confidence=0.9,
        velocity_x=0.0,
        velocity_y=0.0,
    )

    result = engine.fuse(tracking)

    assert result.ttc_valid is True
    assert result.ttc == 10.0


def test_risk_engine_rejects_relative_ttc_but_preserves_semantic_context():
    config = parse_config(
        Path("src/risk/risk_config.yaml")
    )
    engine = RiskEngine(config)
    relative = engine.evaluate({
        "track_id": "1",
        "class": "car",
        "depth": 2.0,
        "depth_type": "relative",
        "depth_change_per_second": False,
        "closing_speed": 0.1,
        "ttc": 1.0,
        "ttc_valid": "True",
        "confidence": 0.8,
        "semantic_context": "car",
        "semantic_confidence": 0.7,
        "trajectory": [],
    })
    metric = engine.evaluate({
        "track_id": "2",
        "class": "car",
        "depth": 20.0,
        "depth_type": "metric",
        "depth_change_per_second": True,
        "closing_speed": 2.0,
        "ttc": 10.0,
        "ttc_valid": True,
        "confidence": 0.8,
        "trajectory": [],
    })

    assert relative["ttc_valid"] is False
    assert relative["ttc"] is None
    assert relative["ttc_reason"] == "requires metric depth and per-second depth change"
    assert relative["semantic_context"] == "car"
    assert relative["semantic_confidence"] == 0.7
    assert metric["ttc_valid"] is True
    assert metric["ttc"] == 10.0


def test_risk_engine_rejects_nonpositive_or_missing_metric_ttc():
    config = parse_config(Path("src/risk/risk_config.yaml"))
    engine = RiskEngine(config)

    for value in (None, 0.0, -1.0):
        result = engine.evaluate({
            "track_id": f"bad-{value}",
            "class": "car",
            "depth": 20.0,
            "depth_type": "metric",
            "depth_change_per_second": True,
            "ttc": value,
            "ttc_valid": True,
            "confidence": 0.8,
            "trajectory": [],
        })
        assert result["ttc_valid"] is False
        assert result["ttc"] is None
        assert result["ttc_reason"] == "invalid or missing TTC value"


def test_semantic_context_does_not_change_risk_score():
    config = parse_config(Path("src/risk/risk_config.yaml"))
    baseline_engine = RiskEngine(config)
    semantic_engine = RiskEngine(config)
    baseline = {
        "track_id": "1",
        "class": "car",
        "depth": 0.8,
        "depth_type": "relative",
        "closing_speed_type": "relative_per_frame",
        "closing_speed": 0.04,
        "ttc_valid": False,
        "confidence": 0.8,
        "trajectory": [],
    }
    with_semantics = {
        **baseline,
        "semantic_context": "road",
        "semantic_confidence": 1.0,
        "road_contact": True,
    }

    base_risk = baseline_engine.evaluate(baseline)
    semantic_risk = semantic_engine.evaluate(with_semantics)

    assert semantic_risk["risk_score"] == base_risk["risk_score"]
    assert semantic_risk["risk_confidence"] == base_risk["risk_confidence"]


def test_segmentation_worker_has_bounded_queue_and_reports_worker_errors():
    entered = threading.Event()
    release = threading.Event()

    def blocking_inference(frame):
        entered.set()
        assert release.wait(2.0)
        return np.full(frame.shape[:2], CLASS_IDS["road"], dtype=np.uint8)

    worker = SegmentationWorker("unused", infer=blocking_inference)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert worker.submit(1, 0.1, frame) is True
    assert entered.wait(1.0)
    assert worker.submit(2, 0.2, frame) is True
    assert worker.submit(3, 0.3, frame) is True
    release.set()
    assert worker.close(timeout=3.0) == []
    assert worker.dropped == 1
    assert worker.inference_count == 2
    assert worker.get_latest()["frame_id"] == 3

    failed = SegmentationWorker(
        "unused",
        infer=lambda _: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    assert failed.submit(9, 0.9, frame) is True
    errors = failed.close(timeout=3.0)
    assert len(errors) == 1
    assert "frame 9" in errors[0]
    assert "synthetic failure" in errors[0]


def test_depth_worker_replaces_a_queued_stale_request(monkeypatch):
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoImageProcessor = object
    transformers_stub.AutoModelForDepthEstimation = object
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    depth_module = importlib.import_module("src.depth.run_depth")
    worker = object.__new__(depth_module.DepthWorker)
    worker.requests = queue.Queue(maxsize=1)
    worker.dropped_requests = 0
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    worker.requests.put_nowait((1, frame))

    assert worker.submit(2, frame) is True
    queued_frame, queued_image = worker.requests.get_nowait()

    assert worker.dropped_requests == 1
    assert queued_frame == 2
    np.testing.assert_array_equal(queued_image, frame)
    worker.stop_requested = False
    worker.thread = threading.Thread(target=worker._run, daemon=True)
    worker.thread.start()
    worker.close()
    assert not worker.thread.is_alive()

    valid_depth, quality, mad = depth_module.robust_depth(
        np.arange(64, dtype=np.float32).reshape(8, 8),
        [0, 0, 8, 8],
    )
    missing_depth, missing_quality, missing_mad = depth_module.robust_depth(
        np.full((8, 8), np.nan, dtype=np.float32),
        [0, 0, 8, 8],
    )
    assert np.isfinite(valid_depth)
    assert quality > 0.0
    assert np.isfinite(mad)
    assert np.isnan(missing_depth)
    assert missing_quality == 0.0
    assert np.isnan(missing_mad)


def test_visualization_submission_does_not_wait_for_encoder(monkeypatch, tmp_path):
    write_started = threading.Event()
    release_writer = threading.Event()

    class BlockingWriter:
        def isOpened(self):
            return True

        def write(self, _frame):
            write_started.set()
            assert release_writer.wait(2.0)

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoWriter", lambda *_args: BlockingWriter())
    worker = VisualizationWorker(tmp_path / "out.mp4", 30.0, (8, 8), queue_size=1)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert worker.submit(1, 0.1, frame) is True
    assert write_started.wait(1.0)

    started = time.perf_counter()
    assert worker.submit(2, 0.2, frame) is True
    assert worker.submit(3, 0.3, frame) is False
    assert time.perf_counter() - started < 0.5

    release_writer.set()
    worker.close()
    assert worker.dropped_frames == 1
    assert worker.written_frames == 2


def test_visualization_overlay_displays_semantic_context(monkeypatch):
    labels = []

    def capture_text(_frame, text, *_args, **_kwargs):
        labels.append(text)
        return _frame

    monkeypatch.setattr(cv2, "putText", capture_text)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    state = {
        "bbox": [1, 1, 10, 10],
        "risk_level": "LOW",
        "track_id": "2",
        "class": "car",
        "semantic_context": "road",
    }

    overlay(frame, [state], 1)

    assert any("Sem:road" in label for label in labels)
