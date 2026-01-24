from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class HeadPoseEstimator:
    """Estimate head pose using MediaPipe Face Landmarker."""

    def __init__(self, model_path: str = "face_landmarker.task"):
        import os
        import urllib.request

        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
            url = (
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )

    def get_head_pose(self, frame: np.ndarray, bbox: tuple) -> Optional[Tuple[float, float, float]]:
        """
        Calculate head pose angles (yaw, pitch, roll) for a face region.
        Returns None if face is not looking at camera (abs(yaw) > 30 or abs(pitch) > 30).
        """
        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]
        img_h, img_w = face_img.shape[:2]

        landmark_indices = [1, 152, 33, 263, 57, 287]

        image_points = np.array(
            [
                (face_landmarks[idx].x * img_w, face_landmarks[idx].y * img_h)
                for idx in landmark_indices
            ],
            dtype=np.float64,
        )

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
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

        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)

        yaw_deg += 180
        if yaw_deg > 180:
            yaw_deg -= 360

        pitch_deg = -pitch_deg

        if abs(roll_deg) > 90 and abs(yaw_deg) > 90:
            if yaw_deg > 0:
                yaw_deg -= 180
            else:
                yaw_deg += 180

        return (yaw_deg, pitch_deg, roll_deg)

    def is_looking_at_camera(
        self,
        yaw: float,
        pitch: float,
        yaw_threshold: float = 30.0,
        pitch_threshold: float = 30.0,
    ) -> bool:
        """Check if person is looking at camera based on yaw and pitch angles."""
        return abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold
