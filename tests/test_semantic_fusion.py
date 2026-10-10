import numpy as np
from src.fusion.semantic_object_fusion import SemanticObjectFusion

def test_road_contact_and_depth():
    fusion = SemanticObjectFusion()

    # Setup mock masks
    seg_mask = np.ones((100, 100), dtype=np.uint8) * 255 # Unlabeled
    seg_mask[80:100, :] = 0 # IDD Class 0: Road

    depth_mask = np.ones((100, 100), dtype=np.float32) * 0.5
    depth_mask[40:90, 40:60] = 0.8 # Deeper object

    track = {"track_id": 1, "yolo_class": "person", "bbox": [40, 40, 60, 90]}

    result = fusion.fuse_object(track, seg_mask, depth_mask, is_stale=False)

    # Verifications
    assert result['road_contact'] is True, "Failed to detect road contact in bottom 20% of bbox"
    assert np.isclose(result['depth_value'], 0.8, atol=1e-6), 'Failed to extract median depth from bbox region'
    assert result['depth_type'] == "relative", "Violated constraint: Metric distance fabricated"
    assert result['context_valid'] is True

def test_temporal_smoothing():
    fusion = SemanticObjectFusion(history_frames=5)

    # Frame 1: Clear Car (Class 9)
    cls = fusion._smooth_temporal_class(1, 9, 0.9)
    assert cls == 9

    # Frame 2 & 3: Still Car
    fusion._smooth_temporal_class(1, 9, 0.9)
    fusion._smooth_temporal_class(1, 9, 0.9)

    # Frame 4: Sudden glitch to Sky (Class 25) with low confidence
    cls = fusion._smooth_temporal_class(1, 25, 0.3)
    assert cls == 9, "Temporal smoothing failed to prevent semantic switch glitch"
