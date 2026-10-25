"""Frame-aligned object context fusion and bounded asynchronous SegFormer inference."""

import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    from .semantic_object_fusion import SemanticObjectFusion
except ImportError:
    from semantic_object_fusion import SemanticObjectFusion


LOGGER = logging.getLogger(__name__)


class SceneFusionInterface:
    def __init__(self, max_segmentation_age_frames=5, history_frames=5, max_tracks=512):
        if max_segmentation_age_frames < 0:
            raise ValueError("max_segmentation_age_frames must be non-negative")
        self.max_segmentation_age_frames = max_segmentation_age_frames
        self.fusion_engine = SemanticObjectFusion(history_frames, max_tracks)
        self.metrics = {
            "segmentation_missing_records": 0,
            "segmentation_stale_records": 0,
            "segmentation_expired_records": 0,
            "segmentation_invalid_records": 0,
        }

    def process(self, frame_id, timestamp, frame, tracked_objects, seg_result, depth_result):
        started = time.perf_counter()
        segmentation = seg_result if isinstance(seg_result, dict) else None
        source_frame = segmentation.get("frame_id") if segmentation else None
        age = frame_id - source_frame if isinstance(source_frame, int) else None
        mask = segmentation.get("mask") if segmentation and segmentation.get("valid") else None

        if age is not None and age < 0:
            self.metrics["segmentation_invalid_records"] += 1
            mask = None
        elif mask is None:
            self.metrics["segmentation_missing_records"] += 1
        elif (
            not isinstance(mask, np.ndarray)
            or mask.ndim != 2
            or not np.issubdtype(mask.dtype, np.integer)
        ):
            self.metrics["segmentation_invalid_records"] += 1
            mask = None
        elif age is None or age > self.max_segmentation_age_frames:
            self.metrics["segmentation_expired_records"] += 1
            mask = None

        if mask is not None and age is not None and age > 0:
            self.metrics["segmentation_stale_records"] += 1

        contexts = []
        for track in tracked_objects:
            context = self.fusion_engine.fuse_object(
                track,
                mask,
                is_stale=bool(age and age > 0),
                segmentation_age_frames=age if age is not None and age >= 0 else 0,
                max_stale_frames=self.max_segmentation_age_frames,
            )
            context.update({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "segmentation_source_frame": source_frame,
                "segmentation_age_frames": age,
                "segmentation_valid": mask is not None,
            })
            contexts.append(context)

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "object_contexts": contexts,
            "segmentation_age": age if age is not None else -1,
            "segmentation_valid": mask is not None,
            "segmentation_source_frame": source_frame,
            "processing_time_ms": (time.perf_counter() - started) * 1000.0,
        }


class SegmentationWorker:
    """Run one bounded, latest-frame-wins segmentation inference worker."""

    def __init__(
        self,
        checkpoint_path: Path,
        device=None,
        queue_size=1,
        infer=None,
    ):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.requests = queue.Queue(maxsize=queue_size)
        self._infer = infer or self._load_inference(checkpoint_path, device)
        self._latest = None
        self._latest_lock = threading.Lock()
        self._stop_requested = False
        self._closed = False
        self.submitted = 0
        self.dropped = 0
        self.inference_count = 0
        self.inference_seconds = 0.0
        self.errors = deque(maxlen=32)
        self.thread = threading.Thread(target=self._run, name="segmentation-worker", daemon=True)
        self.thread.start()

    @staticmethod
    def _load_inference(checkpoint_path, device):
        import cv2
        import torch
        from PIL import Image
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SegFormer checkpoint not found: {checkpoint_path}")

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        torch_device = torch.device(device)
        model_name = "nvidia/mit-b0"
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=41,
            ignore_mismatched_sizes=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        compatible_state = {}
        model_state = model.state_dict()
        for key, value in state_dict.items():
            key = key.removeprefix("model.")
            if key in model_state and model_state[key].shape == value.shape:
                compatible_state[key] = value
        if not compatible_state:
            raise RuntimeError(f"No compatible SegFormer weights found in {checkpoint_path}")
        load_result = model.load_state_dict(compatible_state, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "SegFormer checkpoint is incomplete for the installed model: "
                f"missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )
        model.to(torch_device).eval()

        def infer(frame_bgr):
            height, width = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
            inputs = {key: value.to(torch_device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8, copy=False)

        return infer

    def submit(self, frame_id, timestamp, frame):
        if self._closed:
            raise RuntimeError("cannot submit to a closed segmentation worker")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a three-dimensional image array")
        packet = (int(frame_id), float(timestamp), frame.copy())
        try:
            self.requests.put_nowait(packet)
        except queue.Full:
            try:
                self.requests.get_nowait()
                self.requests.task_done()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.requests.put_nowait(packet)
            except queue.Full:
                self.dropped += 1
                return False
        self.submitted += 1
        return True

    def get_latest(self):
        with self._latest_lock:
            return self._latest

    def close(self, timeout=120.0):
        if not self._closed:
            self._stop_requested = True
            self.thread.join(timeout=timeout)
            self._closed = True
        if self.thread.is_alive():
            raise RuntimeError("segmentation worker did not stop before timeout")
        return list(self.errors)

    def _run(self):
        while not self._stop_requested or not self.requests.empty():
            try:
                frame_id, timestamp, frame = self.requests.get(timeout=0.05)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                mask = np.asarray(self._infer(frame))
                if mask.ndim != 2 or mask.shape != frame.shape[:2]:
                    raise ValueError("SegFormer must return a 2D mask matching the input frame")
                if not np.issubdtype(mask.dtype, np.integer):
                    raise ValueError("SegFormer class mask must have an integer dtype")
                result = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "mask": mask,
                    "valid": True,
                }
                with self._latest_lock:
                    self._latest = result
                self.inference_count += 1
            except Exception as error:
                message = f"frame {frame_id}: {type(error).__name__}: {error}"
                self.errors.append(message)
                LOGGER.exception("Segmentation worker failed for frame %s", frame_id)
            finally:
                self.inference_seconds += time.perf_counter() - started
                self.requests.task_done()
