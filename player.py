"""
Video/Camera Player with Fade Transition
Press SPACE to toggle between looped video and camera with fade effect
Press ESC or Q to exit
"""

import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer


def main():
    VIDEO_PATH = "2025-07-03 18-01-20.mp4"
    FADE_DURATION = 30
    WINDOW_NAME = "Player"

    player = MediaPlayer(VIDEO_PATH, loop=0)

    # Wait for video to initialize and get size
    frame = None
    while frame is None:
        frame, val = player.get_frame()
        if val == 'eof':
            print(f"Error: Cannot open video file '{VIDEO_PATH}'")
            return

    img, t = frame
    video_width, video_height = img.get_size()

    camera = cv2.VideoCapture(0)
    camera_available = camera.isOpened()

    show_camera = False
    transitioning = False
    transition_frame = 0
    last_video_frame = None
    last_camera_frame = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # Read video frame
        if not show_camera or transitioning:
            frame, val = player.get_frame()
            if frame is not None:
                img, t = frame
                data = img.to_bytearray()[0]
                video_frame = np.frombuffer(data, dtype=np.uint8).reshape((video_height, video_width, 3))
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
                last_video_frame = video_frame

        # Read camera
        if (show_camera or transitioning) and camera_available:
            ret, camera_frame = camera.read()
            if ret:
                camera_frame = cv2.resize(camera_frame, (video_width, video_height))
                last_camera_frame = camera_frame

        # Prepare frames
        video_frame = last_video_frame if last_video_frame is not None else np.zeros((video_height, video_width, 3), dtype=np.uint8)
        camera_frame = last_camera_frame if last_camera_frame is not None else np.zeros((video_height, video_width, 3), dtype=np.uint8)

        # Handle transition
        if transitioning:
            transition_frame += 1
            alpha = min(transition_frame / FADE_DURATION, 1.0)

            if alpha >= 1.0:
                transitioning = False
                transition_frame = 0

            if show_camera:
                display_frame = cv2.addWeighted(video_frame, 1.0 - alpha, camera_frame, alpha, 0)
            else:
                display_frame = cv2.addWeighted(camera_frame, 1.0 - alpha, video_frame, alpha, 0)
        else:
            display_frame = camera_frame if show_camera else video_frame

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' ') and not transitioning:
            if not camera_available:
                camera.release()
                camera = cv2.VideoCapture(0)
                camera_available = camera.isOpened()
                if not camera_available:
                    continue
            show_camera = not show_camera
            transitioning = True
            transition_frame = 0

    player.close_player()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
