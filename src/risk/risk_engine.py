"""Stateful, class-aware risk engine for fused tracking/depth records."""

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RISK_LEVELS = ("SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass
class RiskConfig:
    depth_ema_alpha: float
    closing_speed_ema_alpha: float
    ttc_ema_alpha: float
    approach_persistence_frames: int
    risk_escalation_frames: int
    risk_deescalation_frames: int
    critical_score: float
    high_score: float
    medium_score: float
    low_score: float
    ttc_critical_seconds: float
    ttc_high_seconds: float
    ttc_medium_seconds: float
    trajectory_jitter_scale: float
    stale_confidence_factor: float
    closing_scales: dict[str, float] = field(default_factory=dict)


@dataclass
class TrackRiskState:
    depth_history: deque = field(default_factory=lambda: deque(maxlen=8))
    closing_history: deque = field(default_factory=lambda: deque(maxlen=8))
    ttc_history: deque = field(default_factory=lambda: deque(maxlen=8))
    approach_frames: int = 0
    current_level: str = "SAFE"
    pending_level: str = "SAFE"
    pending_count: int = 0
    deescalation_count: int = 0
    last_timestamp: float | None = None


class RiskEngine:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.states: dict[str, TrackRiskState] = {}
        self.transition_count = 0
        self.high_events = 0
        self.critical_events = 0

    def evaluate(self, fused: dict[str, Any]) -> dict[str, Any]:
        track_id = str(fused["track_id"])
        state = self.states.setdefault(track_id, TrackRiskState())
        depth = optional_float(fused.get("depth"))
        closing = optional_float(fused.get("closing_speed"))
        raw_ttc = optional_float(fused.get("ttc")) if fused.get("ttc_valid") == "True" else None
        if depth is not None:
            state.depth_history.append(depth)
        if closing is not None:
            state.closing_history.append(closing)
        if raw_ttc is not None:
            state.ttc_history.append(raw_ttc)

        smoothed_depth = ema(state.depth_history, self.config.depth_ema_alpha)
        smoothed_closing = ema(state.closing_history, self.config.closing_speed_ema_alpha)
        filtered_ttc = ema(state.ttc_history, self.config.ttc_ema_alpha)
        approaching = smoothed_closing is not None and smoothed_closing > 0.0
        current_ttc = filtered_ttc if raw_ttc is not None and approaching else None
        state.approach_frames = state.approach_frames + 1 if approaching else 0
        persistence = min(
            1.0,
            state.approach_frames / max(1, self.config.approach_persistence_frames),
        )
        stale = as_object(fused.get("staleness", {}), {})
        depth_stale = stale.get("depth") == "stale"
        base_confidence = optional_float(fused.get("confidence")) or 0.0
        freshness_factor = self.config.stale_confidence_factor if depth_stale else 1.0
        risk_confidence = max(0.0, min(1.0, base_confidence * freshness_factor))
        class_name = str(fused.get("class", "unknown")).lower()
        closing_scale = self.config.closing_scales.get(
            class_name,
            self.config.closing_scales["default"],
        )
        closing_score = (
            max(0.0, min(1.0, smoothed_closing / closing_scale))
            if approaching and smoothed_closing is not None
            else 0.0
        )
        ttc_score = ttc_risk_score(current_ttc, self.config)
        persistence_score = persistence
        trajectory_stability = trajectory_stability_score(fused, self.config)
        score = max(
            0.0,
            min(
                1.0,
                0.48 * ttc_score
                + 0.22 * closing_score
                + 0.18 * persistence_score
                + 0.08 * risk_confidence
                + 0.04 * trajectory_stability,
            ),
        )
        evidence_level = level_from_score(score, self.config)
        level = self._apply_hysteresis(state, evidence_level)
        reasons = risk_reasons(
            current_ttc,
            smoothed_closing,
            approaching,
            persistence,
            risk_confidence,
            depth_stale,
            ttc_score,
            closing_score,
            self.config,
        )
        if level in ("HIGH", "CRITICAL") and state.current_level == level:
            pass
        data_age = as_object(fused.get("data_age", {}), {})
        return {
            **fused,
            "class_name": fused.get("class_name", fused.get("class", "unknown")),
            "smoothed_depth": smoothed_depth,
            "smoothed_closing_speed": smoothed_closing,
            "raw_ttc": raw_ttc,
            "filtered_ttc": filtered_ttc,
            "ttc": current_ttc,
            "approaching": approaching,
            "approach_persistence_frames": state.approach_frames,
            "trajectory_stability": trajectory_stability,
            "risk_score": score,
            "risk_level": level,
            "risk_confidence": risk_confidence,
            "dominant_risk_reason": reasons[0],
            "secondary_risk_reasons": reasons[1:],
            "lane_conflict": fused.get("lane_conflict") or "unavailable",
            "semantic_context": fused.get("semantic_context") or "unavailable",
            "data_age": data_age,
        }

    def _apply_hysteresis(self, state, evidence_level):
        current_index = RISK_LEVELS.index(state.current_level)
        evidence_index = RISK_LEVELS.index(evidence_level)
        if evidence_index > current_index:
            if state.pending_level == evidence_level:
                state.pending_count += 1
            else:
                state.pending_level = evidence_level
                state.pending_count = 1
            state.deescalation_count = 0
            if state.pending_count >= self.config.risk_escalation_frames:
                state.current_level = evidence_level
                state.pending_level = state.current_level
                state.pending_count = 0
                self.transition_count += 1
                if evidence_level == "HIGH":
                    self.high_events += 1
                if evidence_level == "CRITICAL":
                    self.critical_events += 1
        elif evidence_index < current_index:
            state.deescalation_count += 1
            state.pending_count = 0
            if state.deescalation_count >= self.config.risk_deescalation_frames:
                state.current_level = evidence_level
                state.deescalation_count = 0
                self.transition_count += 1
        else:
            state.pending_count = 0
            state.deescalation_count = 0
        return state.current_level

    def summary(self):
        return {
            "risk_state_transitions": self.transition_count,
            "high_events": self.high_events,
            "critical_events": self.critical_events,
            "tracks_with_state": len(self.states),
        }


def parse_config(path: Path) -> RiskConfig:
    values = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        values[key] = float(value)
    classes = (
        "car", "truck", "bus", "motorcycle", "rider", "person",
        "autorickshaw", "bicycle", "default",
    )
    return RiskConfig(
        depth_ema_alpha=values["depth_ema_alpha"],
        closing_speed_ema_alpha=values["closing_speed_ema_alpha"],
        ttc_ema_alpha=values["ttc_ema_alpha"],
        approach_persistence_frames=int(values["approach_persistence_frames"]),
        risk_escalation_frames=int(values["risk_escalation_frames"]),
        risk_deescalation_frames=int(values["risk_deescalation_frames"]),
        critical_score=values["critical_score"],
        high_score=values["high_score"],
        medium_score=values["medium_score"],
        low_score=values["low_score"],
        ttc_critical_seconds=values["ttc_critical_seconds"],
        ttc_high_seconds=values["ttc_high_seconds"],
        ttc_medium_seconds=values["ttc_medium_seconds"],
        trajectory_jitter_scale=values["trajectory_jitter_scale"],
        stale_confidence_factor=values["stale_confidence_factor"],
        closing_scales={
            name: values[f"closing_scale_{name}"]
            for name in classes
        },
    )


def optional_float(value):
    if value in (None, "", "nan", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_object(value, default):
    if isinstance(value, (dict, list)):
        return value
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def ema(values, alpha):
    if not values:
        return None
    result = values[0]
    for value in list(values)[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def ttc_risk_score(ttc, config):
    if ttc is None or ttc <= 0:
        return 0.0
    if ttc <= config.ttc_critical_seconds:
        return 1.0
    if ttc <= config.ttc_high_seconds:
        return 0.75 + 0.25 * (config.ttc_high_seconds - ttc) / max(config.ttc_high_seconds - config.ttc_critical_seconds, 1e-6)
    if ttc <= config.ttc_medium_seconds:
        return 0.35 + 0.40 * (config.ttc_medium_seconds - ttc) / max(config.ttc_medium_seconds - config.ttc_high_seconds, 1e-6)
    return max(0.0, 0.35 * config.ttc_medium_seconds / ttc)


def trajectory_stability_score(fused, config):
    try:
        trajectory = as_object(fused.get("trajectory", []), [])
    except (TypeError, json.JSONDecodeError):
        return 0.0
    if len(trajectory) < 3:
        return 0.5
    steps = [
        (trajectory[index][0] - trajectory[index - 1][0], trajectory[index][1] - trajectory[index - 1][1])
        for index in range(1, len(trajectory))
    ]
    magnitudes = [math.hypot(x, y) for x, y in steps]
    jitter = sum(abs(b - a) for a, b in zip(magnitudes, magnitudes[1:])) / max(len(magnitudes) - 1, 1)
    return max(0.0, min(1.0, 1.0 - jitter / max(config.trajectory_jitter_scale, 1e-6)))


def level_from_score(score, config):
    if score >= config.critical_score:
        return "CRITICAL"
    if score >= config.high_score:
        return "HIGH"
    if score >= config.medium_score:
        return "MEDIUM"
    if score >= config.low_score:
        return "LOW"
    return "SAFE"


def risk_reasons(ttc, closing, approaching, persistence, confidence, stale, ttc_score, closing_score, config):
    reasons = []
    if ttc is not None and ttc <= config.ttc_high_seconds:
        reasons.append("very low TTC")
    if closing_score >= 0.7:
        reasons.append("rapidly decreasing depth")
    if approaching and persistence >= 1.0:
        reasons.append("sustained closing")
    if confidence >= 0.75 and approaching:
        reasons.append("high confidence approaching object")
    if stale:
        reasons.append("stale auxiliary data")
    return reasons or ["no persistent collision evidence"]
