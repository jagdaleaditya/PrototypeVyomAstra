"""
DONUTS - Configuration Loader
==============================
Loads and provides access to config/config.yaml.
Provides a singleton CONFIG object usable across all modules.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional


def _find_project_root() -> Path:
    """Locate the project root by walking up from this file."""
    here = Path(__file__).resolve().parent
    # Walk up until we find config/config.yaml or hit filesystem root
    for candidate in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
        if (candidate / "config" / "config.yaml").exists():
            return candidate
    # Fallback: two levels above app/
    return here.parent


PROJECT_ROOT = _find_project_root()


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load config.yaml. Falls back to defaults if file not found.

    Args:
        config_path: Absolute or relative path to config.yaml.
                     If None, looks in <project_root>/config/config.yaml.
    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"[Config] WARNING: config.yaml not found at {config_path}. Using defaults.")
        return _defaults()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        print(f"[Config] Loaded: {config_path}")
        return data
    except Exception as e:
        print(f"[Config] ERROR loading config: {e}. Using defaults.")
        return _defaults()


def _defaults() -> dict:
    """Minimal safe defaults matching config.yaml structure."""
    return {
        "camera": {"source": 0, "width": 1280, "height": 720, "fps": 30, "buffer_size": 1},
        "detection": {
            "model": "yolov8n.pt",
            "confidence": 0.35,
            "iou": 0.45,
            "target_classes": ["bottle"],
            "detection_confirm_frames": 3,
            "max_missed_frames": 8,
            "smoothing": 0.60,
        },
        "pose": {
            "model": "models/mediapipe/pose_landmarker_lite.task",
            "min_detection_confidence": 0.5,
            "min_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "num_poses": 1,
        },
        "interaction": {
            "approach_distance": 0.20,
            "pickup_distance_pixels": 120,
            "release_distance": 0.18,
            "movement_threshold": 0.035,
            "pickup_confirm_frames": 3,
            "release_confirm_frames": 3,
        },
        "experiment": {
            "config": "config/experiment.json",
            "step_timeout_seconds": 60,
        },
        "calibration": {
            "camera_file": "data/camera_calibration.npz",
            "rack_file": "data/rack_calibration.npz",
            "enable_rack_relative_pose": False,
        },
        "recording": {
            "enabled": False,
            "save_annotated": True,
            "output_dir": "data/recordings",
            "fourcc": "mp4v",
            "fps": 30,
        },
        "alerts": {
            "voice_enabled": True,
            "voice_rate": 160,
            "voice_volume": 1.0,
            "cooldown_seconds": 5,
            "async_speech": True,
        },
        "logging": {
            "log_dir": "data/experiment_logs",
            "log_format": "json",
            "log_performance": True,
        },
        "performance": {
            "target_fps": 30,
            "show_fps_overlay": True,
            "show_inference_times": True,
        },
    }


def get(config: dict, key_path: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.

    Example:
        fps = get(config, "camera.fps", 30)
    """
    keys = key_path.split(".")
    node = config
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def resolve_path(config: dict, key_path: str) -> Path:
    """
    Get a config path value and resolve it relative to PROJECT_ROOT.

    Example:
        model_path = resolve_path(config, "detection.model")
    """
    value = get(config, key_path, "")
    p = Path(str(value))
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------
CONFIG = load_config()
