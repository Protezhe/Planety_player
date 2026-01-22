import cv2
import numpy as np
import time
import json
import asyncio
import threading
import base64
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import websockets

from cyndilib.wrapper.ndi_recv import RecvColorFormat
from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync


@dataclass
class TrackedPerson:
    """Tracked person with coordinates and last seen time."""
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (cx, cy)
    last_seen: float = field(default_factory=time.time)

    def update(self, bbox: tuple):
        """Update position and last seen time."""
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.last_seen = time.time()

    def to_dict(self, frame_width: int, frame_height: int) -> dict:
        """Convert to dict with normalized coordinates (0-1)."""
        import math
        x1, y1, x2, y2 = self.bbox
        cx, cy = self.center
        # Calculate bbox dimensions
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        return {
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
        }


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

    def __init__(self, source_name: str = "NDI_OBS", websocket_port: int = 8765):
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
        """Process frame: detect persons and update tracking."""
        self.frame_height, self.frame_width = frame.shape[:2]

        results = self.model.track(
            frame,
            persist=True,
            verbose=False
        )

        self.cleanup_old_tracks()

        # Track which IDs are visible in current frame
        current_frame_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for bbox, track_id in zip(boxes, track_ids):
                track_id = int(track_id)  # Convert to Python int
                x1, y1, x2, y2 = bbox
                current_frame_ids.add(track_id)

                if track_id in self.tracked_persons:
                    self.tracked_persons[track_id].update(tuple(bbox))
                else:
                    self.tracked_persons[track_id] = TrackedPerson(
                        track_id=track_id,
                        bbox=tuple(bbox),
                        center=((x1 + x2) // 2, (y1 + y2) // 2)
                    )
                    print(f"New person detected: Track ID {track_id}")

        # Send only currently visible persons via WebSocket
        self.send_tracking_data(current_frame_ids)

        # Capture face snapshot every 10 seconds
        self.capture_face_snapshot(frame, current_frame_ids)

        annotated_frame = self.draw_annotations(frame, current_frame_ids)
        return annotated_frame

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
        """Capture circular face snapshot every 10 seconds and send via WebSocket."""
        current_time = time.time()
        if current_time - self.last_snapshot_time < self.snapshot_interval:
            return

        if not visible_ids:
            return

        # Find first visible person
        for track_id in visible_ids:
            person = self.tracked_persons.get(track_id)
            if person:
                x1, y1, x2, y2 = person.bbox
                cx, cy = person.center
                # Calculate radius as half of circumscribed circle diameter + 20% padding
                # Same formula as in overlay.html
                import math
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

                # Send snapshot via WebSocket
                snapshot_data = {
                    "type": "face_snapshot",
                    "timestamp": current_time,
                    "track_id": int(track_id),
                    "image": f"data:image/png;base64,{img_base64}"
                }
                self.ws_server.send(snapshot_data)

                self.last_snapshot_time = current_time
                print(f"Face snapshot captured for track ID {track_id}, size: {w}x{h}, base64 len: {len(img_base64)}")
                break

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
                print(f"  ID {track_id}: center=({cx}, {cy}), bbox=({x1},{y1})-({x2},{y2}), TTL={ttl:.1f}s")
            print("-----------------------\n")

    def run(self):
        """Main loop: receive frames, detect persons, display results."""
        # Start WebSocket server
        self.ws_server.start()

        if not self.find_and_connect():
            return

        print("Starting face tracking...")
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
