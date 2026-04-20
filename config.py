import os

# Camera Settings
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Whisper Settings
WHISPER_MODEL = "tiny.en"  # Options: tiny.en, base.en, small.en
WHISPER_COMPUTE_TYPE = "float32"  # Change to "int8" for CPU optimization on mid-range laptops
WHISPER_DEVICE = "cpu"  # "cpu", "cuda", or "mps" for Mac M1/M2/M3

# Audio Settings
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_CHUNK_DURATION = 0.5  # Seconds of audio to process at a time

# Visual Settings
FONT_PATH = None  # Will default to system font if None
FONT_SIZE = 36
TEXT_COLOR = (255, 255, 255)  # White (RGB)
HIGHLIGHT_COLOR = (0, 255, 255)  # Cyan
BACKGROUND_ALPHA = 120  # For subtitle background

# Keywords to highlight (Auto-bold/color)
KEYWORDS = ["Aura", "AI", "project", "camera", "real-time", "future"]

# Face Tracking
STABLE_ANCHOR_OFFSET = (0, -100)  # (x_offset, y_offset) relative to forehead
