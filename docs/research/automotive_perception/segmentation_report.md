# Semantic Segmentation Module Report

1. **Model Architecture**: SegFormer (`nvidia/mit-b0`)
2. **Checkpoint**: `lightning_logs/version_2/checkpoints/epoch=9-step=34970.ckpt`
3. **Number of Classes**: 41 output logits (mapped to 26 active Level 3 IDs)
4. **Class Mapping**: IDD Level 3 taxonomy (`src/segmentation/class_mapping.py`)
5. **Ignore Index**: 255
6. **Validation Metrics**: Ground truth pixel masks are not packaged with raw driving video (`huh.mp4` / `indian_road.mp4`); qualitative inspection shows coherent boundary isolation across road, vehicles, riders, vegetation, and sky.
7. **Input Resolution**: 1280x720 (resized to 512x512 for inference)
8. **Output Mask Resolution**: 1280x720 (`.npy` integer arrays)
9. **Raw Segmentation FPS**: ~22–26 FPS
10. **Selected Asynchronous Schedule**: Sample every 6 frames
11. **Effective Segmentation Update Rate**: 5 FPS (leaves full headroom for YOLO + tracking at 30 FPS)
12. **Frame Alignment Percentage**: 100% (aligned 1:1 via integer indices and timestamps)
13. **Per-Track Semantic Context Availability**: Supported via `get_track_semantic_context()` in `src/segmentation/segmentation_interface.py`
14. **Drivable-Region Extraction Method**: Boolean masking over `DRIVABLE_IDS = {0, 1}` with static 15% bottom hood crop (`y > 0.85 * H` set to ignore index 255)
15. **Major Segmentation Failure Modes**:
    * Heavy cast shadows under overpasses occasionally trigger ambiguous obstacle fallbacks.
    * Dense clusters of two-wheelers can exhibit slight boundary blending at wheel-road contact points.
16. **Exact Output Paths**:
    * Masks: `runs/segmentation_eval/masks/frame_*.npy`
    * Metadata: `runs/segmentation_eval/metadata.csv`
    * Visualization: `runs/segmentation_eval/segmentation_visualization.mp4`
17. **Reproduction Command**:
    `py src/segmentation/run_segmentation.py`