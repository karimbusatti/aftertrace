
# =============================================================================
# SIGNAL BLOOM (Lava-red distortion) - REFINED v3
# =============================================================================

def draw_signal_bloom(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Signal Bloom: Lava-red distortion on black background.
    Matches the "fried" high-contrast thermal aesthetic.
    REFINED: Posterized/solarized look with discrete color steps and jagged edges.
    """
    h, w = frame.shape[:2]
    
    # 1. Preprocessing: Grayscale + Extreme Contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Strong local contrast to define regions
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Quantization / Posterization effect
    # Divide into discrete bands to match the "contour" look
    # 5 levels: 0, 1, 2, 3, 4
    n_levels = 5
    # Normalize to 0-255 first
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    # Quantize: (pixel // step) * step
    step = 256 // n_levels
    posterized = (enhanced // step) * step
    
    # 3. Create Stepped Color Map (Discrete Palette)
    # The reference has distinct colors, not smooth gradients.
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    
    # Define palette based on reference:
    # Level 0 (Background/Dark): Dark Red / Black
    # Level 1: Strong Red
    # Level 2: Orange
    # Level 3: Yellow
    # Level 4: White/Bright Yellow
    
    colors_step = [
        (0, 0, 20),      # Deepest Black/Red
        (0, 0, 120),     # Dark Red
        (0, 50, 220),    # Bright Red
        (0, 165, 255),   # Orange
        (180, 255, 255), # White/Yellow
    ]
    
    for i in range(256):
        idx = min(i // step, len(colors_step) - 1)
        lut[i, 0] = colors_step[idx]
        
    # Apply LUT
    # Convert single channel to BGR first for LUT
    posterized_bgr = cv2.cvtColor(posterized, cv2.COLOR_GRAY2BGR)
    output = cv2.LUT(posterized_bgr, lut)
    
    # 4. Digital "Fried" Edges
    # Instead of Canny, use Sobel to find gradients, then threshold strictly
    grad_x = cv2.Sobel(posterized, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(posterized, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # Threshold gradients to get the "lines" between color bands
    _, lines = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
    
    # Make lines jagged/blocky
    # Dilate with a cross kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    lines = cv2.dilate(lines, kernel, iterations=1)
    
    # Overlay lines as bright yellow/white artifacts
    output[lines > 0] = (50, 255, 255)
    
    return output
