import cv2
import numpy as np
import time
import json
import asyncio
import threading
import base64
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import websockets
import mediapipe as mp

from cyndilib.wrapper.ndi_recv import RecvColorFormat
from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync


class HeadPoseEstimator:
    """Estimate head pose using MediaPipe Face Landmarker."""

    def __init__(self):
        # Download face landmarker model
        import urllib.request
        import os

        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        # Create FaceLandmarker
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        # 3D model points for head pose estimation
        # Order matches landmark_indices: [1, 152, 33, 263, 57, 287]
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (225.0, 170.0, -135.0),      # Right eye (33)
            (-225.0, 170.0, -135.0),     # Left eye (263)
            (150.0, -150.0, -125.0),     # Right mouth corner (57)
            (-150.0, -150.0, -125.0)     # Left mouth corner (287)
        ], dtype=np.float64)

    def get_head_pose(self, frame: np.ndarray, bbox: tuple) -> Optional[Tuple[float, float, float]]:
        """
        Calculate head pose angles (yaw, pitch, roll) for a face region.
        Returns None if face is not looking at camera (abs(yaw) > 30 or abs(pitch) > 30).
        """
        x1, y1, x2, y2 = bbox

        # Expand bbox slightly for better face mesh detection
        h, w = frame.shape[:2]
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return None

        # Convert to RGB for MediaPipe
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)

        # Detect face landmarks
        results = self.detector.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        # Get first face landmarks
        face_landmarks = results.face_landmarks[0]

        # Get 2D image points from specific landmarks
        img_h, img_w = face_img.shape[:2]

        # Key landmark indices for face mesh
        # 1: Nose tip, 152: Chin, 33: Right eye, 263: Left eye
        # 57: Right mouth corner, 287: Left mouth corner
        # Swap left/right to match 3D model orientation
        landmark_indices = [1, 152, 33, 263, 57, 287]

        image_points = np.array([
            (face_landmarks[idx].x * img_w,
             face_landmarks[idx].y * img_h)
            for idx in landmark_indices
        ], dtype=np.float64)

        # Camera matrix (assuming no lens distortion)
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # No lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP to get rotation vector
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Calculate Euler angles from rotation matrix
        # Extract yaw, pitch, roll
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(-rotation_mat[2, 0], sy)
            yaw = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
            roll = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
        else:
            pitch = np.arctan2(-rotation_mat[2, 0], sy)
            yaw = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            roll = 0

        # Convert to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)

        # Correct for 180° offset due to coordinate system orientation
        yaw_deg += 180
        # Normalize to ±180° range
        if yaw_deg > 180:
            yaw_deg -= 360

        return (yaw_deg, pitch_deg, roll_deg)

    def is_looking_at_camera(self, yaw: float, pitch: float,
                            yaw_threshold: float = 30.0,
                            pitch_threshold: float = 30.0) -> bool:
        """Check if person is looking at camera based on yaw and pitch angles."""
        return abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold


@dataclass
class TrackedPerson:
    """Tracked person with coordinates and last seen time."""
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (cx, cy)
    last_seen: float = field(default_factory=time.time)
    looking_at_camera: bool = False
    head_pose: Optional[Tuple[float, float, float]] = None  # (yaw, pitch, roll)

    def update(self, bbox: tuple, looking_at_camera: bool = False,
               head_pose: Optional[Tuple[float, float, float]] = None):
        """Update position, head pose, and last seen time."""
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.last_seen = time.time()
        self.looking_at_camera = looking_at_camera
        self.head_pose = head_pose

    def to_dict(self, frame_width: int, frame_height: int) -> dict:
        """Convert to dict with normalized coordinates (0-1)."""
        import math
        x1, y1, x2, y2 = self.bbox
        cx, cy = self.center
        # Calculate bbox dimensions
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


class WebSocketServer:
    """WebSocket server to broadcast tracking data."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.server = None
        self._thread: Optional[threading.Thread] = None

    async def handler(self, websocket):
        """Handle new WebSocket connection."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )

    def send(self, data: dict):
        """Thread-safe method to send data."""
        if self.loop and self.clients:
            message = json.dumps(data)
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    async def _run_server(self):
        """Run the WebSocket server."""
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    def _thread_target(self):
        """Thread target for running asyncio loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_server())

    def start(self):
        """Start the WebSocket server in a background thread."""
        self._thread = threading.Thread(target=self._thread_target, daemon=True)
        self._thread.start()
        time.sleep(0.5)  # Wait for server to start

    def stop(self):
        """Stop the WebSocket server."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


class NDIFaceTracker:
    """NDI stream receiver with face detection and tracking."""

    TRACK_TIMEOUT = 60.0  # Forget track after 1 minute

    def __init__(self, source_name: str = "NDI_OBS", websocket_port: int = 8765,
                 yaw_threshold: float = 30.0, pitch_threshold: float = 30.0,
                 min_face_size: int = 10 ):
        self.source_name = source_name
        self.finder: Optional[Finder] = None
        self.receiver: Optional[Receiver] = None
        self.video_frame: Optional[VideoFrameSync] = None
        self.frame_sync = None
        # Load face detection model from HuggingFace
        face_model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt"
        )
        self.model = YOLO(face_model_path)
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.frame_width = 0
        self.frame_height = 0

        # Head pose estimation
        self.head_pose_estimator = HeadPoseEstimator()
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.min_face_size = min_face_size  # Minimum face width/height to process

        # WebSocket server
        self.ws_server = WebSocketServer(port=websocket_port)

        # Face snapshot settings
        self.last_snapshot_time = -10.0  # Start immediately
        self.snapshot_interval = 10.0  # seconds

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

        self.receiver = Receiver(
            color_format=RecvColorFormat.BGRX_BGRA,
        )

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
            track_id for track_id, person in self.tracked_persons.items()
            if current_time - person.last_seen > self.TRACK_TIMEOUT
        ]
        for track_id in expired_ids:
            print(f"Track {track_id} expired (not seen for {self.TRACK_TIMEOUT}s)")
            del self.tracked_persons[track_id]

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame: detect persons, check head pose, and update tracking."""
        self.frame_height, self.frame_width = frame.shape[:2]

        results = self.model.track(
            frame,
            persist=True,
            verbose=False
        )

        self.cleanup_old_tracks()

        # Track which IDs are visible AND looking at camera in current frame
        current_frame_ids = set()
        looking_at_camera_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for bbox, track_id in zip(boxes, track_ids):
                track_id = int(track_id)  # Convert to Python int
                x1, y1, x2, y2 = bbox

                # Skip very small detections (likely false positives)
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                if bbox_width < self.min_face_size or bbox_height < self.min_face_size:
                    continue

                current_frame_ids.add(track_id)

                # Estimate head pose
                head_pose = self.head_pose_estimator.get_head_pose(frame, bbox)
                looking_at_camera = False

                if head_pose is not None:
                    yaw, pitch, roll = head_pose
                    looking_at_camera = self.head_pose_estimator.is_looking_at_camera(
                        yaw, pitch, self.yaw_threshold, self.pitch_threshold
                    )
                    if looking_at_camera:
                        looking_at_camera_ids.add(track_id)

                if track_id in self.tracked_persons:
                    self.tracked_persons[track_id].update(tuple(bbox), looking_at_camera, head_pose)
                else:
                    self.tracked_persons[track_id] = TrackedPerson(
                        track_id=track_id,
                        bbox=tuple(bbox),
                        center=((x1 + x2) // 2, (y1 + y2) // 2),
                        looking_at_camera=looking_at_camera,
                        head_pose=head_pose
                    )
                    status = "looking at camera" if looking_at_camera else "NOT looking at camera"
                    print(f"New person detected: Track ID {track_id} - {status}")
                    if head_pose:
                        yaw, pitch, roll = head_pose
                        print(f"  Head pose: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")

        # Send only persons looking at camera via WebSocket
        self.send_tracking_data(looking_at_camera_ids)

        # Capture face snapshot every 10 seconds (only if looking at camera)
        self.capture_face_snapshot(frame, looking_at_camera_ids)

        annotated_frame = self.draw_annotations(frame, looking_at_camera_ids)
        return annotated_frame

    def calculate_snapshot_position(self, index: int) -> dict:
        """
        Calculate snapshot position based on index.
        Pattern: center-left (0), center-right (1), down1-left (2), down1-right (3), down2-left (4), down2-right (5), etc.
        Returns dict with level and side: {'level': 0, 'side': 'left'|'right'}
        """
        level = index // 2  # 0,0,1,1,2,2,3,3...
        side = "left" if index % 2 == 0 else "right"
        return {"level": level, "side": side}

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
                "persons": persons_data
            }
            self.ws_server.send(data)

    def capture_face_snapshot(self, frame: np.ndarray, visible_ids: set):
        """Capture circular face snapshots for all visible persons every 10 seconds and send via WebSocket."""
        current_time = time.time()
        if current_time - self.last_snapshot_time < self.snapshot_interval:
            return

        if not visible_ids:
            return

        import math
        snapshots = []

        # Process all visible persons and verify face presence
        for track_id in sorted(visible_ids):
            person = self.tracked_persons.get(track_id)
            if not person:
                continue

            x1, y1, x2, y2 = person.bbox
            cx, cy = person.center
            # Calculate radius as half of circumscribed circle diameter + 20% padding
            # Same formula as in overlay.html
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            diameter = math.sqrt(bbox_width ** 2 + bbox_height ** 2) * 1.2
            radius = int(diameter / 2)

            # Calculate crop bounds with boundary checks
            crop_x1 = max(0, cx - radius)
            crop_y1 = max(0, cy - radius)
            crop_x2 = min(self.frame_width, cx + radius)
            crop_y2 = min(self.frame_height, cy + radius)

            # Crop the region
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            if cropped.size == 0:
                continue

            # Verify that face is actually present in cropped region
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_rgb)
            face_results = self.head_pose_estimator.detector.detect(mp_image)

            if not face_results.face_landmarks or len(face_results.face_landmarks) == 0:
                print(f"Skipping snapshot for track ID {track_id}: no face detected in cropped region")
                continue

            # Create circular mask
            h, w = cropped.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            mask_radius = min(w, h) // 2
            cv2.circle(mask, center, mask_radius, 255, -1)

            # Apply mask - create RGBA image with transparency
            result = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = mask

            # Encode to PNG (supports transparency)
            _, buffer = cv2.imencode('.png', result)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Calculate position for this snapshot (will be recalculated after limiting to 8)
            snapshot = {
                "track_id": int(track_id),
                "image": f"data:image/png;base64,{img_base64}",
                "cropped_size": (w, h)
            }
            snapshots.append(snapshot)

            print(f"Face snapshot prepared for track ID {track_id}, size: {w}x{h}")

            # Stop if we already have 8 valid snapshots
            if len(snapshots) >= 8:
                break

        # Assign positions to snapshots
        for index, snapshot in enumerate(snapshots):
            position = self.calculate_snapshot_position(index)
            snapshot["position"] = position
            print(f"Track ID {snapshot['track_id']}: position level={position['level']} side={position['side']}")

        # Send all snapshots via WebSocket
        if snapshots:
            # Remove temporary cropped_size field before sending
            for snapshot in snapshots:
                snapshot.pop("cropped_size", None)

            snapshot_data = {
                "type": "face_snapshots",
                "timestamp": current_time,
                "snapshots": snapshots
            }
            self.ws_server.send(snapshot_data)
            self.last_snapshot_time = current_time
            print(f"Sent {len(snapshots)} face snapshots")

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
                print(f"  ID {track_id}: center=({cx}, {cy}), bbox=({x1},{y1})-({x2},{y2}), TTL={ttl:.1f}s, {looking_status}")
                if person.head_pose:
                    yaw, pitch, roll = person.head_pose
                    print(f"    Head pose: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")
            print("-----------------------\n")

    def run(self):
        """Main loop: receive frames, detect persons, display results."""
        # Start WebSocket server
        self.ws_server.start()

        if not self.find_and_connect():
            return

        print("Starting face tracking with head pose detection...")
        print(f"Tracking only faces looking at camera (yaw ≤ {self.yaw_threshold}°, pitch ≤ {self.pitch_threshold}°)")
        print(f"Minimum face size: {self.min_face_size}x{self.min_face_size} pixels")
        print("Press 'q' to quit, 'p' to print coordinates")
        print(f"HTML overlay: file://{__file__.replace('ndi_face_tracker.py', 'overlay.html')}")

        last_print_time = time.time()

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

                            if time.time() - last_print_time > 2:
                                self.print_coordinates()
                                last_print_time = time.time()

                            cv2.imshow("NDI Face Tracker", annotated_frame)

                except Exception as e:
                    print(f"Error: {e}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.print_coordinates()

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cv2.destroyAllWindows()
            self.ws_server.stop()
            if self.receiver:
                self.receiver.disconnect()
            if self.finder:
                self.finder.close()


if __name__ == "__main__":
    tracker = NDIFaceTracker(source_name="NDI_OBS")
    tracker.run()
