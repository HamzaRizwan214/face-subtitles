import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request

try:
    import config
except ImportError:
    import project_aura.config as config

class FaceTracker:
    def __init__(self, alpha=0.25):
        # 1. Download the modern Face Landmarker model if missing
        self.model_path = "face_landmarker.task"
        if not os.path.exists(self.model_path):
            print("[Tracker] Downloading MediaPipe Face Landmarker model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                self.model_path
            )
            print("[Tracker] Download complete.")

        # 2. Setup Modern MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Smoothing (EMA)
        self.alpha = alpha
        self.smooth_anchor = None
        
        # Fallback Logic
        self.last_seen_time = 0
        self.last_known_pos = None
        self.fallback_duration = 1.0 # seconds

    def get_anchor(self, frame: np.ndarray):
        h, w, _ = frame.shape
        # Convert to RGB and then to MediaPipe Image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run inference
        results = self.detector.detect(mp_image)
        
        target_pos = None

        if results.face_landmarks:
            # Get landmark 152 (Chin Bottom)
            face_landmarks = results.face_landmarks[0]
            landmark = face_landmarks[152]
            
            # Convert normalized to pixel coordinates
            curr_x = int(landmark.x * w)
            curr_y = int(landmark.y * h) + 40 # Add 40px offset below chin
            target_pos = (curr_x, curr_y)
            
            self.last_seen_time = time.time()
            self.last_known_pos = target_pos
        else:
            # Fallback Logic
            if self.last_known_pos and (time.time() - self.last_seen_time < self.fallback_duration):
                target_pos = self.last_known_pos
            else:
                # Default: Bottom Center
                target_pos = (w // 2, h - 100)

        # Apply Smoothing (EMA)
        if self.smooth_anchor is None:
            self.smooth_anchor = target_pos
        else:
            self.smooth_anchor = (
                int(self.alpha * target_pos[0] + (1 - self.alpha) * self.smooth_anchor[0]),
                int(self.alpha * target_pos[1] + (1 - self.alpha) * self.smooth_anchor[1])
            )
            
        return self.smooth_anchor

    def release(self):
        self.detector.close()

if __name__ == "__main__":
    from engine.camera import CameraManager
    
    cam = CameraManager()
    tracker = FaceTracker(alpha=0.20)
    
    print("Testing Face Tracker. Look for the BLUE dot below your chin.")
    
    try:
        while True:
            ret, frame = cam.get_frame()
            if not ret: continue
            
            start_time = time.time()
            anchor = tracker.get_anchor(frame)
            inference_time = (time.time() - start_time) * 1000
            
            # Draw tracking visualization
            cv2.circle(frame, anchor, 8, (255, 200, 0), -1)
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Aura Tracker Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        tracker.release()
        cv2.destroyAllWindows()
