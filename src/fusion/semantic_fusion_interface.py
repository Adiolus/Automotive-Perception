import time
from semantic_object_fusion import SemanticObjectFusion

class SceneFusionInterface:
    def __init__(self):
        self.fusion_engine = SemanticObjectFusion()

    def process(self, frame_id, timestamp, frame, tracked_objects, seg_result, depth_result):
        start_time = time.perf_counter()

        # Handle async SegFormer state
        seg_mask = seg_result['mask'] if seg_result and seg_result['valid'] else None
        seg_age = (frame_id - seg_result['frame_id']) if seg_result else -1
        is_stale = seg_age > 0

        # Handle async Depth state
        depth_map = depth_result['depth'] if depth_result and depth_result['valid'] else None

        object_contexts = []
        if seg_mask is not None and tracked_objects:
            for track in tracked_objects:
                context = self.fusion_engine.fuse_object(track, seg_mask, depth_map, is_stale)

                # Append required flat metrics for CSV/JSONL
                context.update({
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "segmentation_age": seg_age,
                })
                object_contexts.append(context)

        processing_ms = (time.perf_counter() - start_time) * 1000

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "object_contexts": object_contexts,
            "segmentation_age": seg_age,
            "segmentation_valid": seg_mask is not None,
            "processing_time_ms": processing_ms
        }