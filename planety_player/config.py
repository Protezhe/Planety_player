from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    source_name: str = "NDI_OBS"
    websocket_port: int = 8765
    yaw_threshold: float = 30.0
    pitch_threshold: float = 30.0
    min_face_size: int = 10

    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = ""

    landmarker_model_path: str = "face_landmarker.task"
    face_model_path: Optional[str] = None
    face_model_repo_id: str = "arnabdhar/YOLOv8-Face-Detection"
    face_model_filename: str = "model.pt"

    snapshot_interval: float = float("inf")
    overlay_path: str = "overlay.html"

    startup_target_scene: str = "Распознавание"
    startup_preview_scene: str = "Камера"
    mission_scene_name: str = "Состав миссии"
    video_scene_name: str = "Видео"

    cycle_enabled: bool = True
    cycle_interval_sec: float = 30.0
    cycle_photo_delay_sec: float = 6.0
    cycle_hide_delay_sec: float = 10.0
