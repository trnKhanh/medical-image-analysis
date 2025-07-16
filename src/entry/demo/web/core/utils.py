from typing import Dict

import numpy as np
from PIL import Image, ImageDraw


def draw_mask(image: Image.Image, mask: np.ndarray, class_colors: Dict[int, str] = None) -> Image.Image:
    """Draw mask overlay on the image for visualization."""
    if class_colors is None:
        class_colors = {
            1: "#ff0000",  # Red
            2: "#00ff00",  # Green
        }

    # Convert image to RGBA
    image_rgba = image.convert("RGBA")

    # Create overlay
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw mask regions
    for class_id, color in class_colors.items():
        if class_id in mask:
            # Create a binary mask for this class
            class_mask = (mask == class_id)

            # Convert to coordinates (simplified - you'd want a more efficient method)
            y_coords, x_coords = np.where(class_mask)
            for y, x in zip(y_coords, x_coords):
                # Draw pixel with transparency
                hex_color = color.lstrip('#')
                rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
                rgba = rgb + (128,)  # 50% transparency
                draw.point((x, y), fill=rgba)

    # Composite images
    result = Image.alpha_composite(image_rgba, overlay)
    return result.convert("RGB")


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
