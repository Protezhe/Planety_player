import argparse
from typing import Optional

from planety_player.config import AppConfig
from planety_player.tracker import NDIFaceTracker


def _optional_float(value: str) -> Optional[float]:
    if value is None:
        return None
    if value.lower() in {"inf", "infinity"}:
        return float("inf")
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NDI face tracker with OBS control")
    parser.add_argument("--source-name", default=None, help="NDI source name substring")
    parser.add_argument("--websocket-port", type=int, default=None)
    parser.add_argument("--yaw-threshold", type=float, default=None)
    parser.add_argument("--pitch-threshold", type=float, default=None)
    parser.add_argument("--min-face-size", type=int, default=None)

    parser.add_argument("--obs-host", default=None)
    parser.add_argument("--obs-port", type=int, default=None)
    parser.add_argument("--obs-password", default=None)

    parser.add_argument("--landmarker-model-path", default=None)
    parser.add_argument("--face-model-path", default=None)
    parser.add_argument("--face-model-repo-id", default=None)
    parser.add_argument("--face-model-filename", default=None)

    parser.add_argument(
        "--snapshot-interval",
        type=_optional_float,
        default=None,
        help="Seconds between automatic snapshots, or 'inf' to disable",
    )
    parser.add_argument("--overlay-path", default=None)

    parser.add_argument("--startup-target-scene", default=None)
    parser.add_argument("--startup-preview-scene", default=None)
    parser.add_argument("--video-scene-name", default=None)
    parser.add_argument("--cycle-interval-sec", type=float, default=None)
    parser.add_argument("--cycle-photo-delay-sec", type=float, default=None)
    parser.add_argument("--cycle-hide-delay-sec", type=float, default=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()

    if args.source_name is not None:
        config.source_name = args.source_name
    if args.websocket_port is not None:
        config.websocket_port = args.websocket_port
    if args.yaw_threshold is not None:
        config.yaw_threshold = args.yaw_threshold
    if args.pitch_threshold is not None:
        config.pitch_threshold = args.pitch_threshold
    if args.min_face_size is not None:
        config.min_face_size = args.min_face_size

    if args.obs_host is not None:
        config.obs_host = args.obs_host
    if args.obs_port is not None:
        config.obs_port = args.obs_port
    if args.obs_password is not None:
        config.obs_password = args.obs_password

    if args.landmarker_model_path is not None:
        config.landmarker_model_path = args.landmarker_model_path
    if args.face_model_path is not None:
        config.face_model_path = args.face_model_path
    if args.face_model_repo_id is not None:
        config.face_model_repo_id = args.face_model_repo_id
    if args.face_model_filename is not None:
        config.face_model_filename = args.face_model_filename

    if args.snapshot_interval is not None:
        config.snapshot_interval = args.snapshot_interval
    if args.overlay_path is not None:
        config.overlay_path = args.overlay_path

    if args.startup_target_scene is not None:
        config.startup_target_scene = args.startup_target_scene
    if args.startup_preview_scene is not None:
        config.startup_preview_scene = args.startup_preview_scene
    if args.video_scene_name is not None:
        config.video_scene_name = args.video_scene_name
    if args.cycle_interval_sec is not None:
        config.cycle_interval_sec = args.cycle_interval_sec
    if args.cycle_photo_delay_sec is not None:
        config.cycle_photo_delay_sec = args.cycle_photo_delay_sec
    if args.cycle_hide_delay_sec is not None:
        config.cycle_hide_delay_sec = args.cycle_hide_delay_sec

    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    tracker = NDIFaceTracker(config)
    tracker.run()


if __name__ == "__main__":
    main()
