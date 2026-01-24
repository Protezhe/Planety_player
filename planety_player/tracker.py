import base64
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync
from cyndilib.wrapper.ndi_recv import RecvColorFormat
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from planety_player.config import AppConfig
from planety_player.head_pose import HeadPoseEstimator
from planety_player.obs_controller import OBSController
from planety_player.websocket_server import WebSocketServer


@dataclass
class TrackedPerson:
    """Tracked person with coordinates and last seen time."""

    track_id: int
    bbox: tuple
    center: tuple
    last_seen: float = field(default_factory=time.time)
    looking_at_camera: bool = False
    head_pose: Optional[Tuple[float, float, float]] = None

    def update(
        self,
        bbox: tuple,
        looking_at_camera: bool = False,
        head_pose: Optional[Tuple[float, float, float]] = None,
    ):
        """Update position, head pose, and last seen time."""
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.last_seen = time.time()
        self.looking_at_camera = looking_at_camera
        self.head_pose = head_pose

    def to_dict(self, frame_width: int, frame_height: int) -> dict:
        """Convert to dict with normalized coordinates (0-1)."""
        x1, y1, x2, y2 = self.bbox
        cx, cy = self.center
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        data = {
            "id": int(self.track_id),
            "bbox": {
                "x1": x1 / frame_width,
                "y1": y1 / frame_height,
                "x2": x2 / frame_width,
                "y2": y2 / frame_height,
                "width": bbox_width / frame_width,
                "height": bbox_height / frame_height,
            },
            "center": {
                "x": cx / frame_width,
                "y": cy / frame_height,
            },
            "looking_at_camera": bool(self.looking_at_camera),
        }
        if self.head_pose is not None:
            yaw, pitch, roll = self.head_pose
            data["head_pose"] = {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
            }
        return data


class NDIFaceTracker:
    """NDI stream receiver with face detection and tracking."""

    TRACK_TIMEOUT = 60.0
    MIN_OVERLAY_HEIGHT_RATIO = 0.10

    def __init__(self, config: AppConfig):
        self.config = config
        self.source_name = config.source_name
        self.finder: Optional[Finder] = None
        self.receiver: Optional[Receiver] = None
        self.video_frame: Optional[VideoFrameSync] = None
        self.frame_sync = None

        face_model_path = self._resolve_face_model_path(config)
        self.model = YOLO(face_model_path)

        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.frame_width = 0
        self.frame_height = 0

        self.head_pose_estimator = HeadPoseEstimator(config.landmarker_model_path)
        self.yaw_threshold = config.yaw_threshold
        self.pitch_threshold = config.pitch_threshold
        self.min_face_size = config.min_face_size

        self.ws_server = WebSocketServer(port=config.websocket_port)
        self.obs = OBSController(
            host=config.obs_host, port=config.obs_port, password=config.obs_password
        )

        self.last_snapshot_time = 0.0
        self.snapshot_interval = config.snapshot_interval

    def _resolve_face_model_path(self, config: AppConfig) -> str:
        if config.face_model_path:
            return config.face_model_path
        return hf_hub_download(
            repo_id=config.face_model_repo_id,
            filename=config.face_model_filename,
        )

    def find_and_connect(self) -> bool:
        """Find NDI source and connect to it."""
        print(f"Looking for NDI source '{self.source_name}'...")

        self.finder = Finder()
        self.finder.open()

        timeout_start = time.time()
        target_source = None

        while time.time() - timeout_start < 10:
            time.sleep(0.5)
            sources = list(self.finder.get_source_names())

            if sources:
                print(f"Found {len(sources)} NDI sources:")
                for i, name in enumerate(sources):
                    print(f"  [{i}] {name}")
                    if self.source_name in name:
                        target_source = name

                if target_source:
                    break

        if target_source is None:
            print(f"ERROR: Source containing '{self.source_name}' not found")
            self.finder.close()
            return False

        print(f"Connecting to '{target_source}'...")

        source = self.finder.get_source(target_source)

        self.receiver = Receiver(color_format=RecvColorFormat.BGRX_BGRA)

        self.video_frame = VideoFrameSync()
        self.frame_sync = self.receiver.frame_sync
        self.frame_sync.set_video_frame(self.video_frame)

        self.receiver.set_source(source)

        time.sleep(0.5)

        print("Connected successfully!")
        return True

    def cleanup_old_tracks(self):
        """Remove tracks not seen for TRACK_TIMEOUT seconds."""
        current_time = time.time()
        expired_ids = [
            track_id
            for track_id, person in self.tracked_persons.items()
            if current_time - person.last_seen > self.TRACK_TIMEOUT
        ]
        for track_id in expired_ids:
            print(f"Track {track_id} expired (not seen for {self.TRACK_TIMEOUT}s)")
            del self.tracked_persons[track_id]

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame: detect persons, check head pose, and update tracking."""
        self.frame_height, self.frame_width = frame.shape[:2]

        results = self.model.track(frame, persist=True, verbose=False)

        self.cleanup_old_tracks()

        looking_at_camera_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for bbox, track_id in zip(boxes, track_ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = bbox

                bbox_width = x2 - x1
                bbox_height = y2 - y1
                if bbox_width < self.min_face_size or bbox_height < self.min_face_size:
                    continue

                head_pose = self.head_pose_estimator.get_head_pose(frame, bbox)
                looking_at_camera = False

                if head_pose is not None:
                    yaw, pitch, _roll = head_pose
                    looking_at_camera = self.head_pose_estimator.is_looking_at_camera(
                        yaw, pitch, self.yaw_threshold, self.pitch_threshold
                    )
                    if looking_at_camera:
                        looking_at_camera_ids.add(track_id)

                if track_id in self.tracked_persons:
                    self.tracked_persons[track_id].update(
                        tuple(bbox), looking_at_camera, head_pose
                    )
                else:
                    self.tracked_persons[track_id] = TrackedPerson(
                        track_id=track_id,
                        bbox=tuple(bbox),
                        center=((x1 + x2) // 2, (y1 + y2) // 2),
                        looking_at_camera=looking_at_camera,
                        head_pose=head_pose,
                    )
                    status = "looking at camera" if looking_at_camera else "NOT looking at camera"
                    print(f"New person detected: Track ID {track_id} - {status}")
                    if head_pose:
                        yaw, pitch, roll = head_pose
                        print(
                            f"  Head pose: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°"
                        )

        visible_ids = {
            track_id
            for track_id in looking_at_camera_ids
            if self._is_large_enough(self.tracked_persons.get(track_id))
        }
        self.send_tracking_data(visible_ids)
        self.capture_face_snapshot(frame, visible_ids)

        annotated_frame = self.draw_annotations(frame, looking_at_camera_ids)
        return annotated_frame

    def calculate_head_angle_deviation(self, yaw: float, pitch: float) -> float:
        """
        Calculate total head angle deviation from direct camera gaze.
        Lower value = looking more directly at camera.
        """
        return float(np.sqrt(yaw**2 + pitch**2))

    def calculate_snapshot_position(self, index: int, total_count: int) -> dict:
        """
        Calculate snapshot position based on index and total count.
        Pattern: center-left (0), center-right (1), down1-left (2), down1-right (3), etc.
        Positions are centered vertically based on total count.
        Returns dict with level and side: {'level': 0.0, 'side': 'left'|'right'}
        """
        side = "left" if index % 2 == 0 else "right"

        left_count = (total_count + 1) // 2
        right_count = total_count // 2

        max_left_level = left_count - 1
        max_right_level = right_count - 1
        max_level = max(max_left_level, max_right_level)

        if side == "left":
            side_index = index // 2
            raw_level = float(side_index)
        else:
            side_index = index // 2
            offset = (left_count - right_count) / 2.0
            raw_level = float(side_index) + offset

        level = raw_level - (max_level / 2.0)

        return {"level": level, "side": side}

    def _is_large_enough(self, person: Optional[TrackedPerson]) -> bool:
        if person is None or self.frame_width <= 0 or self.frame_height <= 0:
            return False
        x1, y1, x2, y2 = person.bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width <= 0 or bbox_height <= 0:
            return False
        return (bbox_height / self.frame_height) >= self.MIN_OVERLAY_HEIGHT_RATIO

    def send_tracking_data(self, visible_ids: set):
        """Send tracking data to WebSocket clients (only visible persons)."""
        if self.frame_width > 0 and self.frame_height > 0:
            persons_data = [
                person.to_dict(self.frame_width, self.frame_height)
                for track_id, person in self.tracked_persons.items()
                if track_id in visible_ids
            ]
            data = {
                "type": "tracking",
                "timestamp": time.time(),
                "frame_size": {"width": self.frame_width, "height": self.frame_height},
                "persons": persons_data,
            }
            self.ws_server.send(data)

    def clear_overlay(self):
        """Clear all snapshots from the overlay."""
        data = {"type": "clear_snapshots", "timestamp": time.time()}
        self.ws_server.send(data)
        print("✓ Sent clear overlay command")

    def hide_snapshots_animated(self):
        """Animate snapshots sliding off screen."""
        data = {"type": "hide_snapshots", "timestamp": time.time()}
        self.ws_server.send(data)
        print("✓ Sent hide snapshots animation command")

    def trigger_snapshot(self, frame: np.ndarray):
        """Manually trigger face snapshot capture for all people looking at camera."""
        looking_at_camera_ids = {
            track_id
            for track_id, person in self.tracked_persons.items()
            if person.looking_at_camera and self._is_large_enough(person)
        }
        if not looking_at_camera_ids:
            print("⚠ No one is looking at camera - cannot take snapshot")
            return False

        self._capture_snapshot_internal(frame, looking_at_camera_ids)
        return True

    def capture_face_snapshot(self, frame: np.ndarray, visible_ids: set):
        """Capture circular face snapshots for all visible persons (automatic timer-based)."""
        current_time = time.time()
        if current_time - self.last_snapshot_time < self.snapshot_interval:
            return

        if not visible_ids:
            return

        self._capture_snapshot_internal(frame, visible_ids)

    def _capture_snapshot_internal(self, frame: np.ndarray, visible_ids: set):
        """Internal method to capture and send snapshots."""
        current_time = time.time()
        snapshots = []

        for track_id in sorted(visible_ids):
            person = self.tracked_persons.get(track_id)
            if not person:
                continue

            x1, y1, x2, y2 = person.bbox
            cx, cy = person.center
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            diameter = np.sqrt(bbox_width**2 + bbox_height**2) * 1.2
            radius = int(diameter / 2)

            crop_x1 = max(0, cx - radius)
            crop_y1 = max(0, cy - radius)
            crop_x2 = min(self.frame_width, cx + radius)
            crop_y2 = min(self.frame_height, cy + radius)

            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            if cropped.size == 0:
                continue

            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_rgb)
            face_results = self.head_pose_estimator.detector.detect(mp_image)

            if not face_results.face_landmarks:
                print(
                    f"Skipping snapshot for track ID {track_id}: no face detected in cropped region"
                )
                continue

            if person.head_pose is None:
                print(f"Skipping snapshot for track ID {track_id}: no head pose data")
                continue

            yaw, pitch, _roll = person.head_pose
            angle_deviation = self.calculate_head_angle_deviation(yaw, pitch)

            h, w = cropped.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            mask_radius = min(w, h) // 2
            cv2.circle(mask, center, mask_radius, 255, -1)

            result = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = mask

            _success, buffer = cv2.imencode(".png", result)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            snapshot = {
                "track_id": int(track_id),
                "image": f"data:image/png;base64,{img_base64}",
                "angle_deviation": angle_deviation,
                "cropped_size": (w, h),
            }
            snapshots.append(snapshot)

            print(
                f"Face snapshot prepared for track ID {track_id}, size: {w}x{h}, "
                f"angle_deviation: {angle_deviation:.2f}°"
            )

        snapshots.sort(key=lambda x: x["angle_deviation"])

        def are_snapshots_duplicate(snap1, snap2, threshold=0.15):
            """Check if two snapshots are likely the same person based on face position."""
            person1 = self.tracked_persons.get(snap1["track_id"])
            person2 = self.tracked_persons.get(snap2["track_id"])
            if not person1 or not person2:
                return False

            cx1, cy1 = person1.center
            cx2, cy2 = person2.center
            distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            normalized_distance = distance / np.sqrt(
                self.frame_width**2 + self.frame_height**2
            )

            return normalized_distance < threshold

        unique_snapshots = []
        for snapshot in snapshots:
            is_duplicate = False
            for unique_snap in unique_snapshots:
                if are_snapshots_duplicate(snapshot, unique_snap, threshold=0.05):
                    is_duplicate = True
                    print(
                        f"Skipping duplicate: Track {snapshot['track_id']} is too close to "
                        f"Track {unique_snap['track_id']}"
                    )
                    break
            if not is_duplicate:
                unique_snapshots.append(snapshot)

        top_snapshots = unique_snapshots[:8]

        planets = [
            "Меркурий",
            "Венера",
            "Земля",
            "Марс",
            "Юпитер",
            "Сатурн",
            "Уран",
            "Нептун",
        ]

        shuffled_planets = planets.copy()
        random.shuffle(shuffled_planets)

        total_count = len(top_snapshots)
        for index, snapshot in enumerate(top_snapshots):
            position = self.calculate_snapshot_position(index, total_count)
            snapshot["position"] = position
            snapshot["position"]["index"] = index
            snapshot["planet"] = (
                shuffled_planets[index]
                if index < len(shuffled_planets)
                else planets[index % len(planets)]
            )
            print(
                f"Track ID {snapshot['track_id']}: position level={position['level']:.1f} "
                f"side={position['side']}, planet={snapshot['planet']}, "
                f"angle_deviation: {snapshot['angle_deviation']:.2f}°"
            )

        if top_snapshots:
            for snapshot in top_snapshots:
                snapshot.pop("angle_deviation", None)
                snapshot.pop("cropped_size", None)

            snapshot_data = {
                "type": "face_snapshots",
                "timestamp": current_time,
                "snapshots": top_snapshots,
            }
            self.ws_server.send(snapshot_data)
            self.last_snapshot_time = current_time
            print(f"Sent {len(top_snapshots)} face snapshots (selected from {len(snapshots)} total)")

    def draw_annotations(self, frame: np.ndarray, visible_ids: set) -> np.ndarray:
        """Return frame without annotations."""
        return frame

    def print_coordinates(self):
        """Print coordinates of all tracked persons."""
        if self.tracked_persons:
            print("\n--- Tracked Persons ---")
            for track_id, person in self.tracked_persons.items():
                x1, y1, x2, y2 = person.bbox
                cx, cy = person.center
                ttl = max(0, self.TRACK_TIMEOUT - (time.time() - person.last_seen))
                looking_status = "👁️ LOOKING" if person.looking_at_camera else "❌ NOT LOOKING"
                print(
                    f"  ID {track_id}: center=({cx}, {cy}), bbox=({x1},{y1})-({x2},{y2}), "
                    f"TTL={ttl:.1f}s, {looking_status}"
                )
                if person.head_pose:
                    yaw, pitch, roll = person.head_pose
                    print(
                        f"    Head pose: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°"
                    )
            print("-----------------------\n")

    def run_startup_sequence(self):
        """Run cycle sequence: preview → program scene → photo → hide → video."""
        print("\n=== Starting Cycle Sequence ===")

        scenes = []
        if self.obs._connected:
            scenes = self.obs.get_scene_list()
            if scenes:
                print(f"  Available scenes: {', '.join(scenes)}")

            studio_mode = self.obs.get_studio_mode_enabled()
            if studio_mode:
                preview_scene = self.config.startup_preview_scene
                print(f"  ℹ Studio Mode detected - fixing preview to '{preview_scene}'")
                if preview_scene in scenes:
                    self.obs.set_preview_scene(preview_scene)
                else:
                    print(f"  ⚠ Warning: Preview scene '{preview_scene}' not found!")

        target_scene = self.config.startup_target_scene
        print(f"Switching program scene to '{target_scene}'...")
        if self.obs._connected:
            current_scene = self.obs.get_current_scene()
            print(f"  Current program scene: {current_scene}")

            if target_scene in scenes:
                self.obs.switch_scene(target_scene)
            else:
                print(f"  ⚠ Warning: Scene '{target_scene}' not found in OBS!")
                print(f"  Available scenes are: {scenes}")
        else:
            print("  ⚠ Warning: Not connected to OBS - cannot switch scene")
        time.sleep(0.5)

        now = time.time()
        self.cycle_photo_time = now + self.config.cycle_photo_delay_sec
        self.cycle_hide_time = (
            self.cycle_photo_time + self.config.cycle_hide_delay_sec
        )
        self.cycle_photo_taken = False
        self.cycle_hide_started = False
        self.cycle_video_switched = False
        self.cycle_hide_complete_time = None

        print("\nSequence:")
        print(f"  → In {self.config.cycle_photo_delay_sec:.0f}s: Take photo")
        print(
            f"  → In {self.config.cycle_photo_delay_sec + self.config.cycle_hide_delay_sec:.0f}s: "
            "Hide snapshots + switch to video"
        )
        print("=== Cycle Sequence Initiated ===\n")

    def _overlay_url(self) -> str:
        overlay_path = Path(self.config.overlay_path)
        if not overlay_path.is_absolute():
            overlay_path = Path.cwd() / overlay_path
        return f"file://{overlay_path}"

    def run(self):
        """Main loop: receive frames, detect persons, display results."""
        self.ws_server.start()
        self.obs.connect()

        if not self.find_and_connect():
            return

        print("Starting face tracking with head pose detection...")
        print(
            "Tracking only faces looking at camera "
            f"(yaw ≤ {self.yaw_threshold}°, pitch ≤ {self.pitch_threshold}°)"
        )
        print(f"Minimum face size: {self.min_face_size}x{self.min_face_size} pixels")
        print("Press 'q' to quit, 'p' to print coordinates")
        print(f"HTML overlay: {self._overlay_url()}")

        self.run_startup_sequence()
        next_cycle_time = time.time() + self.config.cycle_interval_sec

        try:
            while True:
                try:
                    self.frame_sync.capture_video()

                    xres = self.video_frame.xres
                    yres = self.video_frame.yres

                    if xres > 0 and yres > 0:
                        data = self.video_frame.get_array()
                        if data is not None and data.size > 0:
                            frame_bgrx = data.reshape(yres, xres, 4)
                            frame = frame_bgrx[:, :, :3].copy()

                            annotated_frame = self.process_frame(frame)

                            now = time.time()
                            if self.config.cycle_enabled and now >= next_cycle_time:
                                self.run_startup_sequence()
                                next_cycle_time = now + self.config.cycle_interval_sec

                            if (
                                hasattr(self, "cycle_photo_taken")
                                and not self.cycle_photo_taken
                                and now >= self.cycle_photo_time
                            ):
                                print("\n⏰ Cycle photo time reached - triggering photo!")
                                if self.trigger_snapshot(frame):
                                    print("✓ Cycle photo captured successfully")
                                self.cycle_photo_taken = True

                            if (
                                hasattr(self, "cycle_video_switched")
                                and not self.cycle_video_switched
                                and now >= self.cycle_hide_time
                            ):
                                if not self.cycle_hide_started:
                                    print("\n⏰ Cycle hide time reached - animating snapshots hide!")
                                    self.hide_snapshots_animated()
                                    self.cycle_hide_started = True
                                    self.cycle_hide_complete_time = now + 1.0
                                elif (
                                    self.cycle_hide_complete_time is not None
                                    and now >= self.cycle_hide_complete_time
                                ):
                                    print("Switching to 'Видео' scene!")
                                    if self.obs._connected:
                                        self.obs.switch_scene(self.config.video_scene_name)
                                    self.cycle_video_switched = True

                            cv2.imshow("NDI Face Tracker", annotated_frame)

                except Exception as e:
                    print(f"Error: {e}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    self.print_coordinates()

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cv2.destroyAllWindows()
            self.ws_server.stop()
            self.obs.disconnect()
            if self.receiver:
                self.receiver.disconnect()
            if self.finder:
                self.finder.close()
