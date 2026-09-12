"""Stable lane and future sign interfaces for fusion."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LaneState:
    frame: int
    timestamp: float
    lanes: list[dict]
    ego_lane_id: Optional[int]
    left_lane_id: Optional[int]
    right_lane_id: Optional[int]
    ego_lane_offset: Optional[float]
    lane_boundary_distance: Optional[float]
    confidence: float


@dataclass
class TrafficSignObservation:
    sign_id: str
    class_name: str
    bbox: list[float]
    confidence: float
    timestamp: float
