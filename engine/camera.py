import cv2
import time
import threading
from typing import Optional, Tuple

try:
    import config
except ImportError:
    import project_aura.config as config

class CameraManager:
    def __init__(self, camera_id: int = config.CAMERA_ID):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)

        #correcting the camera opening check
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open camera {self.camera_id}")

    def get_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Captures a single frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        
        # Flip frame horizontally for a natural mirror effect
        frame = cv2.flip(frame, 1)
        return True, frame

    def release(self):
        """Releases the camera resource."""
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    # Test capture
    cam = CameraManager()
    while True:
        ret, frame = cam.get_frame()
        if ret:
            cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
