import numpy as np
from collections import OrderedDict, defaultdict, deque

try:
    from .class_mapping import CLASS_NAMES, CLASS_IDS, ROAD_CLASSES
except ImportError:
    from class_mapping import CLASS_NAMES, CLASS_IDS, ROAD_CLASSES


SIDEWALK_IDS = {CLASS_IDS["sidewalk"]}

class SemanticObjectFusion:
    def __init__(self, history_frames=5, max_tracks=512):
        if history_frames < 1 or max_tracks < 1:
            raise ValueError("history_frames and max_tracks must be positive")
        self.history_frames = history_frames
        self.max_tracks = max_tracks
        self.track_history = OrderedDict()
        self.metrics = {"semantic_switches": 0, "processed_objects": 0, "stale_associations": 0}

    def _get_roi_distribution(self, mask, x1, y1, x2, y2):
        """Extracts class distribution strictly from the bbox interior."""
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return {}, None, 0.0

        unique, counts = np.unique(roi, return_counts=True)
        total = roi.size
        distribution = {int(k): float(v)/total for k, v in zip(unique, counts)}

        dominant_class = int(unique[np.argmax(counts)])
        confidence = float(np.max(counts)) / total
        return distribution, dominant_class, confidence

    def _check_road_contact(self, mask, x1, y1, x2, y2):
        """Evaluates the bottom 20% of the bounding box for road contact."""
        h = y2 - y1
        bottom_y = max(y1, int(y2 - 0.2 * h))
        contact_roi = mask[bottom_y:y2, x1:x2]

        if contact_roi.size == 0:
            return False, False

        unique_classes = set(np.unique(contact_roi))
        on_road = bool(unique_classes.intersection(ROAD_CLASSES))
        on_sidewalk = bool(unique_classes.intersection(SIDEWALK_IDS))

        return on_road, on_sidewalk

    def _smooth_temporal_class(self, track_id, current_class, confidence):
        """Applies confidence-aware temporal smoothing to prevent flickering."""
        history = self.track_history.get(track_id)
        if history is None:
            history = deque(maxlen=self.history_frames)
            self.track_history[track_id] = history
        self.track_history.move_to_end(track_id)
        while len(self.track_history) > self.max_tracks:
            self.track_history.popitem(last=False)
        history.append((current_class, confidence))

        if len(history) < 3:
            return current_class

        # Weighted voting based on recent confidence
        votes = defaultdict(float)
        for cls_id, conf in history:
            if cls_id is not None:
                votes[cls_id] += conf

        smoothed_class = max(votes.items(), key=lambda x: x[1])[0]

        if smoothed_class != current_class and current_class is not None:
            self.metrics["semantic_switches"] += 1

        return smoothed_class

    def fuse_object(
        self,
        track,
        segmentation_mask,
        depth_map=None,
        is_stale=False,
        segmentation_age_frames=0,
        max_stale_frames=5,
    ):
        self.metrics["processed_objects"] += 1
        if is_stale or segmentation_age_frames > 0:
            self.metrics["stale_associations"] += 1
        effective_age = max(segmentation_age_frames, 1 if is_stale else 0)

        track_id = track["track_id"]
        bbox = track["bbox"]
        if len(bbox) != 4:
            raise ValueError("track bbox must contain four coordinates")
        x1, y1, x2, y2 = map(int, bbox)

        seg_valid = (
            segmentation_mask is not None
            and segmentation_mask.ndim == 2
            and 0 <= effective_age <= max_stale_frames
        )
        if not seg_valid:
            return {
                "track_id": track_id,
                "yolo_class": track.get("yolo_class", track.get("class_name")),
                "bbox": [x1, y1, x2, y2],
                "semantic_class": None,
                "semantic_class_id": None,
                "semantic_class_name": None,
                "raw_semantic_class": None,
                "semantic_confidence": 0.0,
                "semantic_distribution": {},
                "depth_value": None,
                "depth_type": "relative",
                "depth_valid": False,
                "road_contact": None,
                "sidewalk_contact": None,
                "context_valid": False,
                "segmentation_available": False,
                "segmentation_age_frames": effective_age,
            }

        img_h, img_w = segmentation_mask.shape

        # Clamp coordinates to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            return {
                "track_id": track_id,
                "yolo_class": track.get("yolo_class", track.get("class_name")),
                "bbox": [x1, y1, x2, y2],
                "semantic_class": None,
                "semantic_class_id": None,
                "semantic_class_name": None,
                "raw_semantic_class": None,
                "semantic_confidence": 0.0,
                "semantic_distribution": {},
                "depth_value": None,
                "depth_type": "relative",
                "depth_valid": False,
                "road_contact": None,
                "sidewalk_contact": None,
                "context_valid": False,
                "segmentation_available": True,
                "segmentation_age_frames": effective_age,
            }

        # 1. Semantic Association
        distribution, dominant_class, conf = self._get_roi_distribution(segmentation_mask, x1, y1, x2, y2)
        freshness = max(0.0, 1.0 - effective_age / (max_stale_frames + 1.0))
        confidence = conf * freshness
        smoothed_class = self._smooth_temporal_class(track_id, dominant_class, confidence)

        # 2. Road Contact Context
        on_road, on_sidewalk = self._check_road_contact(segmentation_mask, x1, y1, x2, y2)

        # 3. Depth Association
        depth_valid = False
        depth_median = -1.0
        if depth_map is not None:
            if depth_map.ndim != 2 or depth_map.shape != segmentation_mask.shape:
                raise ValueError("depth map and segmentation mask must be same-size 2D arrays")
            depth_roi = depth_map[y1:y2, x1:x2]
            if depth_roi.size > 0:
                finite_depth = depth_roi[np.isfinite(depth_roi)]
                if finite_depth.size:
                    depth_median = float(np.median(finite_depth))
                    depth_valid = True

        return {
            "track_id": track_id,
            "yolo_class": track.get("yolo_class", track.get("class_name")),
            "bbox": [x1, y1, x2, y2],
            "semantic_class": smoothed_class,
            "semantic_class_id": smoothed_class,
            "semantic_class_name": CLASS_NAMES.get(smoothed_class),
            "raw_semantic_class": dominant_class,
            "semantic_confidence": confidence,
            "raw_semantic_confidence": conf,
            "semantic_distribution": distribution,
            "depth_value": depth_median,
            "depth_type": "relative",
            "depth_valid": depth_valid,
            "road_contact": on_road,
            "sidewalk_contact": on_sidewalk,
            "context_valid": True,
            "segmentation_available": True,
            "segmentation_age_frames": effective_age,
        }