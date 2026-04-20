import cv2
import time
import signal
import sys
import os

# Ensure the current directory is in the path for engine imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.camera import CameraManager
from engine.tracker import FaceTracker
from engine.transcriber import Transcriber
from engine.renderer import OverlayRenderer
import config

class ProjectAuraApp:
    def __init__(self):
        print("[Aura] Initializing core modules...")
        
        print("[Aura] 1/4: Starting Camera...")
        self.camera = CameraManager()
        
        print("[Aura] 2/4: Loading Face Tracker Model...")
        self.tracker = FaceTracker()
        
        print("[Aura] 3/4: Loading Whisper AI Model (This may take 1-2 minutes on first run)...")
        self.transcriber = Transcriber()
        
        print("[Aura] 4/4: Initializing Renderer...")
        self.renderer = OverlayRenderer()
        
        print("[Aura] Initialization complete! Opening window...")
        
        self.current_text = ""
        self.last_text_time = 0
        self.text_persistence = 3.0  # Seconds to keep text on screen after speaking

    def run(self):
        print("[Aura] Starting audio stream...")
        self.transcriber.start()
        
        print("[Aura] Entering main loop. Press 'q' to quit.")
        
        try:
            while True:
                start_time = time.time()
                
                # 1. Capture Frame
                ret, frame = self.camera.get_frame()
                if not ret:
                    break
                
                # 2. Track Face & Get Anchor
                anchor = self.tracker.get_anchor(frame)
                
                # 3. Get Latest Transcription
                self.current_text, _ = self.transcriber.get_current_text()
                
                # 4. Render Overlay
                frame = self.renderer.render_subtitles(frame, self.current_text, anchor)
                
                # 5. Display Preview
                cv2.imshow("Project Aura - Preview", frame)
                
                # 6. Maintain 30 FPS (approximate)
                elapsed = time.time() - start_time
                wait_time = max(1, int((1/config.FPS - elapsed) * 1000))
                
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n[Aura] Cleaning up resources...")
        self.transcriber.stop()
        self.camera.release()
        self.tracker.release()
        cv2.destroyAllWindows()
        print("[Aura] Shutdown complete.")

if __name__ == "__main__":
    app = ProjectAuraApp()
    app.run()
