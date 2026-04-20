from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

try:
    import config
except ImportError:
    import project_aura.config as config

class OverlayRenderer:
    def __init__(self):
        try:
            # Try to load a font, fallback to default if not found
            if config.FONT_PATH:
                self.font = ImageFont.truetype(config.FONT_PATH, config.FONT_SIZE)
            else:
                self.font = ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()

    def render_subtitles(self, frame: np.ndarray, text: str, anchor: tuple) -> np.ndarray:
        """Renders text near the anchor point with keyword highlighting."""
        if not text or not anchor:
            return frame

        # Convert OpenCV frame to PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, "RGBA")
        
        # Calculate text size using font.getbbox (modern Pillow)
        try:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older Pillow
            text_w, text_h = draw.textsize(text, font=self.font)

        # Padding for background
        padding = 10
        bg_x = anchor[0] - text_w // 2 - padding
        bg_y = anchor[1] - text_h - padding
        bg_w = text_w + 2 * padding
        bg_h = text_h + 2 * padding

        # Draw semi-transparent background box
        draw.rectangle(
            [bg_x, bg_y, bg_x + bg_w, bg_y + bg_h],
            fill=(0, 0, 0, config.BACKGROUND_ALPHA)
        )

        # Split text into words to check for keywords
        words = text.split()
        current_x = bg_x + padding
        
        for word in words:
            clean_word = word.strip(".,!?\"'").lower()
            color = config.TEXT_COLOR
            
            # Check if word is a keyword
            if any(k.lower() == clean_word for k in config.KEYWORDS):
                color = config.HIGHLIGHT_COLOR
            
            # Draw word
            draw.text((current_x, bg_y + padding), word + " ", font=self.font, fill=color)
            
            # Update X position for next word
            try:
                word_bbox = draw.textbbox((0, 0), word + " ", font=self.font)
                current_x += word_bbox[2] - word_bbox[0]
            except AttributeError:
                current_x += draw.textsize(word + " ", font=self.font)[0]

        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
