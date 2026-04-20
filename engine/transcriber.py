import threading
import queue
import numpy as np
import sounddevice as sd
import time
from faster_whisper import WhisperModel

try:
    import config
except ImportError:
    import project_aura.config as config

class Transcriber:
    def __init__(self):
        # Load the model (pinned to base as requested)
        # Using faster-whisper for significantly better CPU performance
        self.model = WhisperModel(
            "base.en", 
            device="cpu", 
            compute_type="int8" # Optimized for CPU
        )
        
        self.audio_queue = queue.Queue()
        self.current_text = ""
        self.last_update_time = time.time()
        self.is_running = False
        
        # Audio Settings
        self.sample_rate = 16000
        self.buffer_size = self.sample_rate * 3  # 3 seconds of context
        self.overlap = self.sample_rate * 1      # 1 second overlap
        
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)

    def audio_callback(self, indata, frames, time_info, status):
        """Continuous background audio capture."""
        self.audio_queue.put(indata.copy())

    def _get_new_text(self, new_transcription: str):
        """Logic to prevent repetition by finding the new suffix."""
        new_transcription = new_transcription.strip()
        if not self.current_text:
            return new_transcription
        
        # Simple suffix match: if the new text starts with the old text, trim it
        if new_transcription.startswith(self.current_text):
            return new_transcription[len(self.current_text):].strip()
        
        # If the context has shifted completely, return the new text
        return new_transcription

    def transcription_worker(self):
        """The heavy lifter: runs in a separate thread."""
        while self.is_running:
            try:
                # Pull all available audio from the queue
                while not self.audio_queue.empty():
                    data = self.audio_queue.get()
                    # Shift buffer and add new data
                    shift = len(data)
                    self.audio_buffer[:-shift] = self.audio_buffer[shift:]
                    self.audio_buffer[-shift:] = data.flatten()

                # Run inference on the 3-second buffer
                segments, _ = self.model.transcribe(
                    self.audio_buffer,
                    beam_size=1, # Fast beam size for low latency
                    vad_filter=True # Ignore silence to save CPU
                )
                
                text_result = " ".join([s.text for s in segments]).strip()
                
                if text_result:
                    # Update text if it's actually new
                    if text_result != self.current_text:
                        self.current_text = text_result
                        self.last_update_time = time.time()
                
                # Small sleep to prevent CPU hammering
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[Transcriber] Error: {e}")

    def get_current_text(self):
        """Returns (text, opacity) for the overlay renderer."""
        elapsed = time.time() - self.last_update_time
        
        # Text Decay System
        if elapsed > 4.0:
            # Linear fade out over 1 second
            opacity = max(0, 1.0 - (elapsed - 4.0))
        else:
            opacity = 1.0
            
        return self.current_text if opacity > 0 else "", opacity

    def start(self):
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.5) # 500ms chunks
        )
        self.stream.start()
        self.worker_thread = threading.Thread(target=self.transcription_worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

if __name__ == "__main__":
    # --- Standalone Terminal Test ---
    print("[Aura] Initializing Whisper base... (First run may take a minute)")
    ts = Transcriber()
    ts.start()
    
    print("[Aura] Listening... Speak now. (Press Ctrl+C to stop)")
    try:
        last_printed = ""
        while True:
            text, opacity = ts.get_current_text()
            if text and text != last_printed:
                print(f"[{opacity:.1f}] {text}")
                last_printed = text
            time.sleep(0.2)
    except KeyboardInterrupt:
        ts.stop()
        print("\n[Aura] Stopped.")
