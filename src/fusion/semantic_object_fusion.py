import numpy as np
from collections import defaultdict, deque

# Assuming IDD_LEVEL3_CLASSES from your class_mapping.py
DRIVABLE_IDS = {0, 1}
SIDEWALK_IDS = {2}

class SemanticObjectFusion:
    def __init__(self, history_frames=5):
        self.history_frames = history_frames
        self.track_history = defaultdict(lambda: deque(maxlen=self.history_frames))
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
        on_road = bool(unique_classes.intersection(DRIVABLE_IDS))
        on_sidewalk = bool(unique_classes.intersection(SIDEWALK_IDS))

        return on_road, on_sidewalk

    def _smooth_temporal_class(self, track_id, current_class, confidence):
        """Applies confidence-aware temporal smoothing to prevent flickering."""
        history = self.track_history[track_id]
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
            self.metrics["semantic_switches"] -= 1 # Prevented a switch

        return smoothed_class

    def fuse_object(self, track, segmentation_mask, depth_map, is_stale):
        self.metrics["processed_objects"] += 1
        if is_stale:
            self.metrics["stale_associations"] += 1

        x1, y1, x2, y2 = map(int, track['bbox'])
        img_h, img_w = segmentation_mask.shape

        # Clamp coordinates to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        # 1. Semantic Association
        distribution, dominant_class, conf = self._get_roi_distribution(segmentation_mask, x1, y1, x2, y2)
        smoothed_class = self._smooth_temporal_class(track['track_id'], dominant_class, conf)

        # 2. Road Contact Context
        on_road, on_sidewalk = self._check_road_contact(segmentation_mask, x1, y1, x2, y2)

        # 3. Depth Association
        depth_valid = False
        depth_median = -1.0
        if depth_map is not None:
            depth_roi = depth_map[y1:y2, x1:x2]
            if depth_roi.size > 0:
                depth_median = float(np.median(depth_roi))
                depth_valid = True

        return {
            "track_id": track['track_id'],
            "yolo_class": track['yolo_class'],
            "bbox": [x1, y1, x2, y2],
            "semantic_class": smoothed_class,
            "raw_semantic_class": dominant_class,
            "semantic_confidence": conf,
            "semantic_distribution": distribution,
            "depth_value": depth_median,
            "depth_type": "relative", # Strictly enforced per constraints
            "depth_valid": depth_valid,
            "road_contact": on_road,
            "sidewalk_contact": on_sidewalk,
            "context_valid": True
        }