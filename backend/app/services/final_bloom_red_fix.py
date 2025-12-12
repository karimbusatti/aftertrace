
# =============================================================================
# SIGNAL BLOOM (Lava-red distortion) - FINAL COLOR FIX
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
    FINAL COLORS: Pure Deep Red (0,0,255) and Blinding Yellow.
    """
    h, w = frame.shape[:2]
    
    # 1. Preprocessing: Grayscale + Extreme Contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Strong local contrast to define regions
    # Increase to 6.0 for more texture
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Level adjustment (Crush blacks deeply)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    # Threshold out background noise to get deep black
    _, enhanced = cv2.threshold(enhanced, 50, 255, cv2.THRESH_TOZERO)
    
    # 3. Create Custom Color Map (Strict Reference Palette)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    
    # The reference is NOT a smooth gradient. It is specific bands.
    # Base: Black
    # Mid: Pure Saturated Red (0, 0, 255)
    # High: Pure Yellow (0, 255, 255)
    # Peak: White (255, 255, 255)
    
    # We construct the LUT to ramp sharply between these points
    for i in range(256):
        if i < 40:
            # Deep Black / Dark Red fade
            t = i / 40
            # From (0,0,0) to (0,0,100)
            lut[i, 0] = (0, 0, int(100 * t))
            
        elif i < 120:
            # The "Main Red" Body
            # From (0,0,100) to (0,0,255) - Pure Red
            t = (i - 40) / 80
            lut[i, 0] = (0, 0, 100 + int(155 * t))
            
        elif i < 180:
            # Red to Orange transition
            # From (0,0,255) to (0,128,255)
            t = (i - 120) / 60
            lut[i, 0] = (0, int(128 * t), 255)
            
        elif i < 230:
            # Orange to Pure Yellow
            # From (0,128,255) to (0,255,255)
            t = (i - 180) / 50
            lut[i, 0] = (0, 128 + int(127 * t), 255)
            
        else:
            # Yellow to White (The "Fried" Highlights)
            # From (0,255,255) to (255,255,255)
            t = (i - 230) / 25
            lut[i, 0] = (int(255 * t), 255, 255)
            
    # Apply LUT
    # Convert single channel to BGR first for LUT
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    output = cv2.LUT(enhanced_bgr, lut)
    
    # 4. Digital Artifacts / Edge Glow
    # Use Sobel for "outline" look derived from brightness
    grad_x = cv2.Sobel(enhanced, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(enhanced, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # Make edges strictly Yellow/White
    # Threshold gradient
    _, edge_mask = cv2.threshold(grad, 100, 255, cv2.THRESH_BINARY)
    
    # Dilate edges slightly to give them "weight"
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edge_mask = cv2.dilate(edge_mask, kernel_cross, iterations=1)
    
    # Apply edges
    # Where edges are strong, set color to Yellow (0, 255, 255)
    output[edge_mask > 0] = (0, 255, 255)
    
    # 5. Stability: No random noise flicker
    # If noise is desired for texture, use a STATIC pattern
    # h_n, w_n = h // 2, w // 2
    # static_noise = np.indices((h_n, w_n)).sum(axis=0) % 2 * 20
    # static_noise = cv2.resize(static_noise.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    # output = cv2.add(output, cv2.cvtColor(static_noise, cv2.COLOR_GRAY2BGR))
    
    return output
