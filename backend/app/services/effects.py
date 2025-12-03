"""
Visual effects rendering module.

Draws points, trails, connections, and effects based on preset configuration.
All drawing functions read from the preset dict - no hardcoded values.
"""

import cv2
import numpy as np
import random
from typing import Any

from .presets import get_preset_colors, COLOR_PALETTES
from .types import TrackedPoint


# Overlay blend intensity (0.0 - 1.0)
OVERLAY_BLEND_ALPHA = 0.4


# =============================================================================
# MAIN DRAWING ENTRY POINT
# =============================================================================

def draw_frame(
    frame: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    frame_idx: int,
    overlay_mode: bool = False,
    face_data: dict | None = None,
) -> np.ndarray:
    """
    Draw all visual elements on a frame according to the preset.
    
    Args:
        frame: Original video frame (BGR)
        points: List of tracked points (alive and fading)
        preset: Preset configuration dict
        frame_idx: Current frame index (for animations)
        overlay_mode: If True, blend effects at 40% over original frame
                      If False, replace background with darkened/tinted version
        face_data: Optional face detection results
    
    Returns:
        Rendered frame with effects applied
    """
    colors = get_preset_colors(preset)
    
    if overlay_mode:
        # OVERLAY MODE: Keep original visible, blend effects on top
        output = _draw_frame_overlay(frame, points, preset, colors, frame_idx)
    else:
        # NORMAL MODE: Replace background with effect
        output = _draw_frame_replace(frame, points, preset, colors, frame_idx)
    
    # Apply face detection overlays
    if face_data:
        output = _apply_face_overlays(output, face_data, preset, colors, frame_idx)
    
    # Apply CCTV overlay if enabled
    if preset.get("cctv_overlay", False):
        from .face_detection import draw_cctv_overlay
        draw_cctv_overlay(output, frame_idx, fps=30.0)
    
    return output


def _apply_face_overlays(
    frame: np.ndarray,
    face_data: dict,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int,
) -> np.ndarray:
    """Apply face detection visual overlays based on preset settings."""
    from .face_detection import (
        draw_face_boxes, draw_face_mesh, draw_face_glow, draw_biometric_data
    )
    
    output = frame.copy()
    faces = face_data.get("faces", [])
    mesh_points = face_data.get("mesh_points", [])
    
    # Get colors for face overlays - use white for clean style
    biometric_style = preset.get("biometric_style", "cctv")
    if biometric_style == "clean":
        face_color = (255, 255, 255)
    else:
        face_color = colors.get("point", (0, 255, 0))
    
    # Draw face glow first (goes under everything)
    if preset.get("face_glow", False) and faces:
        draw_face_glow(output, faces, face_color, intensity=0.3)
    
    # Draw face mesh with appropriate color
    if preset.get("detect_mesh", False) and mesh_points:
        mesh_color = (255, 255, 255) if biometric_style == "clean" else face_color
        draw_face_mesh(output, mesh_points, mesh_color, draw_contours=True, glow=biometric_style != "clean")
    
    # Draw face boxes - skip if biometric_data handles it
    if preset.get("face_boxes", False) and faces and not preset.get("biometric_data", False):
        box_style = "cctv" if preset.get("cctv_overlay", False) else "minimal"
        draw_face_boxes(
            output, faces, (255, 255, 255), thickness=2,
            show_confidence=True, frame_idx=frame_idx, style=box_style
        )
    
    # Draw biometric data panels (includes face boxes for clean style)
    if preset.get("biometric_data", False) and faces:
        draw_biometric_data(output, faces, mesh_points, frame_idx, face_color, style=biometric_style)
    
    return output


def _draw_frame_replace(
    frame: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int,
) -> np.ndarray:
    """Normal mode: darken background and draw effects on top."""
    
    # Check for text-based effects first (they replace the entire pipeline)
    text_result = apply_text_effect(frame, preset, colors)
    if text_result is not None:
        output = text_result
        
        # Still draw minimal point overlay for text modes
        overlay = np.zeros_like(output)
        _draw_all_elements(overlay, points, preset, colors)
        
        # Apply glow to points
        glow_intensity = preset.get("glow_intensity", 0)
        if glow_intensity > 0:
            glow = cv2.GaussianBlur(overlay, (15, 15), 0)
            overlay = cv2.addWeighted(overlay, 1.0, glow, glow_intensity, 0)
        
        output = cv2.add(output, overlay)
        
        # Scanlines still apply
        if preset.get("scanlines", False):
            draw_scanlines(output, frame_idx)
        
        return output
    
    # Check for cube/depth effects
    cube_result = apply_cube_effect(frame, preset, colors, frame_idx, points)
    if cube_result is not None:
        output = cube_result
        
        # Scanlines for CRT vibe
        if preset.get("scanlines", False):
            draw_scanlines(output, frame_idx)
        
        return output
    
    # Standard point-based pipeline
    output = frame.copy()
    
    # Apply high contrast B&W if enabled
    if preset.get("high_contrast_bw", False):
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Darken background for contrast
    darken = preset.get("darken_factor", 0.9)
    output = (output * darken).astype(np.uint8)
    
    # Apply background tint
    tint = colors.get("background_tint")
    if tint:
        tint_overlay = np.full_like(output, tint, dtype=np.uint8)
        output = cv2.add(output, tint_overlay)
    
    # Create overlay for additive drawing
    overlay = np.zeros_like(output)
    
    # Draw all elements onto overlay
    _draw_all_elements(overlay, points, preset, colors)
    
    # Apply blur (for soft blobs effect)
    blur_radius = preset.get("blur_radius", 0)
    if blur_radius > 0:
        kernel = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        overlay = cv2.GaussianBlur(overlay, (kernel, kernel), 0)
    
    # Apply glow effect
    glow_intensity = preset.get("glow_intensity", 0)
    if glow_intensity > 0:
        glow = cv2.GaussianBlur(overlay, (21, 21), 0)
        overlay = cv2.addWeighted(overlay, 1.0, glow, glow_intensity, 0)
    
    # Apply colormap for heat map effect
    if preset.get("use_colormap", False):
        gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        if gray_overlay.max() > 0:
            colored = cv2.applyColorMap(gray_overlay, cv2.COLORMAP_HOT)
            mask = gray_overlay > 10
            output[mask] = cv2.addWeighted(
                output[mask], 0.3,
                colored[mask], 0.7,
                0
            )
            overlay = np.zeros_like(output)
    
    # Composite overlay onto output
    output = cv2.add(output, overlay)
    
    # Draw scanlines (surveillance mode)
    if preset.get("scanlines", False):
        draw_scanlines(output, frame_idx)
    
    return output


def _draw_frame_overlay(
    frame: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int,
) -> np.ndarray:
    """
    Overlay mode: blend effects at ~40% over the original frame.
    Shows "what the algorithm sees" on top of reality.
    """
    # Keep original frame intact
    original = frame.copy()
    
    # Check for text-based effects
    text_result = apply_text_effect(frame, preset, colors)
    if text_result is not None:
        # For text effects in overlay mode, blend text layer over original
        effect_layer = text_result
        
        # Add minimal point overlay
        point_layer = np.zeros_like(frame)
        _draw_all_elements(point_layer, points, preset, colors)
        
        glow_intensity = preset.get("glow_intensity", 0)
        if glow_intensity > 0:
            glow = cv2.GaussianBlur(point_layer, (15, 15), 0)
            point_layer = cv2.addWeighted(point_layer, 1.0, glow, glow_intensity, 0)
        
        effect_layer = cv2.add(effect_layer, point_layer)
        
        # Blend text effect over original at higher alpha (text needs to be visible)
        effect_mask = cv2.cvtColor(effect_layer, cv2.COLOR_BGR2GRAY) > 10
        
        output = original.copy()
        if effect_mask.any():
            blended = cv2.addWeighted(
                original, 0.4,  # More effect, less original for text
                effect_layer, 0.6,
                0
            )
            output[effect_mask] = blended[effect_mask]
        
        if preset.get("scanlines", False):
            draw_scanlines(output, frame_idx)
        
        return output
    
    # Check for cube/depth effects
    cube_result = apply_cube_effect(frame, preset, colors, frame_idx, points)
    if cube_result is not None:
        effect_layer = cube_result
        
        # Blend cube effect over original (cube effect looks better with more effect)
        effect_mask = cv2.cvtColor(effect_layer, cv2.COLOR_BGR2GRAY) > 8
        
        output = original.copy()
        if effect_mask.any():
            blended = cv2.addWeighted(
                original, 0.35,
                effect_layer, 0.65,
                0
            )
            output[effect_mask] = blended[effect_mask]
        
        if preset.get("scanlines", False):
            draw_scanlines(output, frame_idx)
        
        return output
    
    # Standard point-based pipeline
    # Create effect layer (black background)
    effect_layer = np.zeros_like(frame)
    
    # Draw all elements onto effect layer
    _draw_all_elements(effect_layer, points, preset, colors)
    
    # Apply blur (for soft blobs effect)
    blur_radius = preset.get("blur_radius", 0)
    if blur_radius > 0:
        kernel = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        effect_layer = cv2.GaussianBlur(effect_layer, (kernel, kernel), 0)
    
    # Apply glow effect
    glow_intensity = preset.get("glow_intensity", 0)
    if glow_intensity > 0:
        glow = cv2.GaussianBlur(effect_layer, (21, 21), 0)
        effect_layer = cv2.addWeighted(effect_layer, 1.0, glow, glow_intensity, 0)
    
    # Apply colormap for heat map effect
    if preset.get("use_colormap", False):
        gray_overlay = cv2.cvtColor(effect_layer, cv2.COLOR_BGR2GRAY)
        if gray_overlay.max() > 0:
            effect_layer = cv2.applyColorMap(gray_overlay, cv2.COLORMAP_HOT)
    
    # Blend effect layer over original at OVERLAY_BLEND_ALPHA
    # Only blend where there are actual effects (non-black pixels)
    effect_mask = cv2.cvtColor(effect_layer, cv2.COLOR_BGR2GRAY) > 5
    
    output = original.copy()
    if effect_mask.any():
        # Blend: output = original * (1 - alpha) + effect * alpha
        blended = cv2.addWeighted(
            original, 1.0 - OVERLAY_BLEND_ALPHA,
            effect_layer, OVERLAY_BLEND_ALPHA,
            0
        )
        # Apply blend only where effects exist, keep original elsewhere
        output[effect_mask] = blended[effect_mask]
    
    # Optionally add a subtle scanline effect for surveillance vibe
    if preset.get("scanlines", False):
        draw_scanlines(output, frame_idx)
    
    return output


def _draw_all_elements(
    overlay: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    colors: dict,
):
    """Draw trails, connections, and points onto an overlay."""
    # Draw trails
    draw_trails(overlay, points, preset, colors)
    
    # Draw grid connections
    if preset.get("connect_points", False):
        draw_connections(overlay, points, preset, colors)
    
    # Draw points/shapes
    draw_points(overlay, points, preset, colors)


# =============================================================================
# TRAIL DRAWING
# =============================================================================

def draw_trails(
    overlay: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    colors: dict,
):
    """Draw point trails with optional fading."""
    base_color = np.array(colors["trail"])
    thickness = preset.get("trace_thickness", 1)
    fade = preset.get("trail_fade", True)
    max_trail = preset.get("trail_length", 25)
    
    for point in points:
        trail = point.trail[-max_trail:] + [point.position]
        if len(trail) < 2:
            continue
        
        if fade:
            # Draw each segment with decreasing opacity
            for i in range(len(trail) - 1):
                alpha = (i + 1) / len(trail)
                color = (base_color * alpha).astype(int).tolist()
                pt1 = tuple(trail[i].astype(int))
                pt2 = tuple(trail[i + 1].astype(int))
                cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
        else:
            # Draw as single polyline
            pts = np.array(trail, dtype=np.int32)
            cv2.polylines(
                overlay, [pts], False,
                colors["trail"], thickness, cv2.LINE_AA
            )


# =============================================================================
# CONNECTION DRAWING
# =============================================================================

def draw_connections(
    overlay: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    colors: dict,
):
    """Draw lines connecting nearby points (grid effect)."""
    alive_points = [p for p in points if p.alive]
    if len(alive_points) < 2:
        return
    
    max_dist = preset.get("max_connect_distance", 100)
    thickness = preset.get("connection_thickness", 1)
    base_color = np.array(colors["line"])
    
    positions = np.array([p.position for p in alive_points])
    
    # O(n²) distance check - fine for <300 points
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < max_dist:
                # Fade line based on distance
                alpha = 1.0 - (dist / max_dist)
                color = (base_color * alpha).astype(int).tolist()
                pt1 = tuple(positions[i].astype(int))
                pt2 = tuple(positions[j].astype(int))
                cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)


# =============================================================================
# POINT/SHAPE DRAWING
# =============================================================================

def draw_points(
    overlay: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    colors: dict,
):
    """Draw points as various shapes."""
    shape = preset.get("shape", "circle")
    size = preset.get("point_size", 5)
    color = colors["point"]
    
    for point in points:
        if not point.alive:
            continue
        
        center = tuple(point.position.astype(int))
        x, y = center
        
        if shape == "circle":
            cv2.circle(overlay, center, size, color, -1, cv2.LINE_AA)
            
        elif shape == "square":
            half = size // 2
            cv2.rectangle(
                overlay,
                (x - half, y - half),
                (x + half, y + half),
                color, -1
            )
            
        elif shape == "diamond":
            pts = np.array([
                [x, y - size],
                [x + size, y],
                [x, y + size],
                [x - size, y],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color, cv2.LINE_AA)
            
        elif shape == "cross":
            # Crosshairs
            arm = size
            cv2.line(overlay, (x - arm, y), (x + arm, y), color, 1, cv2.LINE_AA)
            cv2.line(overlay, (x, y - arm), (x, y + arm), color, 1, cv2.LINE_AA)
            # Small center dot
            cv2.circle(overlay, center, 2, color, -1, cv2.LINE_AA)
            
        else:
            # Fallback to circle
            cv2.circle(overlay, center, size, color, -1, cv2.LINE_AA)


# =============================================================================
# EFFECTS
# =============================================================================

def draw_scanlines(frame: np.ndarray, frame_idx: int):
    """Add CRT-style scanlines effect (modifies frame in-place)."""
    h, w = frame.shape[:2]
    
    # Static horizontal scanlines
    for y in range(0, h, 3):
        frame[y, :] = (frame[y, :] * 0.7).astype(np.uint8)
    
    # Moving bright scanline (top to bottom sweep)
    scan_y = (frame_idx * 4) % h
    if scan_y + 2 < h:
        frame[scan_y:scan_y+2, :] = np.clip(
            frame[scan_y:scan_y+2, :].astype(np.int32) + 25,
            0, 255
        ).astype(np.uint8)


# =============================================================================
# TEXT-BASED EFFECTS
# =============================================================================

def draw_data_body(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Data Body effect: render the subject as a cloud of alphanumeric glyphs.
    
    Samples brightness on a grid and places characters where brightness
    exceeds a threshold, forming the silhouette from text.
    """
    h, w = frame.shape[:2]
    
    # Get preset params with defaults
    glyph_chars = preset.get("glyph_chars", "ABCDEF0123456789")
    cell_size = preset.get("glyph_cell_size", 10)
    jitter = preset.get("glyph_jitter", 2)
    min_brightness = preset.get("min_brightness", 40)
    font_scale = preset.get("glyph_font_scale", 0.35)
    invert_bg = preset.get("invert_background", False)
    
    # Convert to grayscale for brightness sampling
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create output: dark background (or white if inverted)
    if invert_bg:
        output = np.full((h, w, 3), 240, dtype=np.uint8)
        text_color = (40, 40, 40)  # Dark text on light bg
    else:
        output = np.zeros((h, w, 3), dtype=np.uint8)
        text_color = colors.get("point", (80, 255, 80))
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    # Sample grid and place glyphs
    glyph_list = list(glyph_chars)
    
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            # Sample brightness at cell center
            cy = min(y + cell_size // 2, h - 1)
            cx = min(x + cell_size // 2, w - 1)
            brightness = gray[cy, cx]
            
            # Only draw if brightness exceeds threshold
            if brightness < min_brightness:
                continue
            
            # Pick random glyph
            glyph = random.choice(glyph_list)
            
            # Apply jitter
            jx = x + random.randint(-jitter, jitter)
            jy = y + random.randint(-jitter, jitter)
            
            # Clamp to frame bounds
            jx = max(0, min(jx, w - 1))
            jy = max(0, min(jy, h - 1))
            
            # Scale color brightness with pixel brightness
            brightness_factor = brightness / 255.0
            scaled_color = tuple(int(c * brightness_factor) for c in text_color)
            
            cv2.putText(
                output, glyph, (jx, jy),
                font, font_scale, scaled_color,
                thickness, cv2.LINE_AA
            )
    
    return output


def draw_numeric_aura(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Numeric Aura effect: glowing 0s and 1s clustered around edges/motion.
    
    Uses Canny edge detection to find interesting regions, then places
    binary glyphs with a soft glow/halo effect.
    """
    h, w = frame.shape[:2]
    
    # Get preset params with defaults
    glyph_chars = preset.get("glyph_chars", "01")
    edge_threshold = preset.get("edge_threshold", 50)
    density = preset.get("glyph_density", 0.4)
    font_scale = preset.get("glyph_font_scale", 0.4)
    glow_radius = preset.get("text_glow_radius", 11)
    glow_intensity = preset.get("text_glow_intensity", 0.6)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
    
    # Get edge pixel coordinates
    edge_points = np.column_stack(np.where(edges > 0))  # (row, col) format
    
    # Create glyph layer (black background)
    glyph_layer = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    text_color = colors.get("point", (50, 200, 255))  # Amber/gold
    
    glyph_list = list(glyph_chars)
    
    # Downsample edge points for performance (max ~2000 glyphs)
    max_glyphs = 2000
    if len(edge_points) > max_glyphs / density:
        step = int(len(edge_points) * density / max_glyphs)
        step = max(1, step)
        edge_points = edge_points[::step]
    
    # Place glyphs at edge locations
    for (row, col) in edge_points:
        glyph = random.choice(glyph_list)
        
        # Small random offset for organic feel
        ox = random.randint(-3, 3)
        oy = random.randint(-3, 3)
        
        px = max(0, min(col + ox, w - 1))
        py = max(8, min(row + oy, h - 1))  # Offset for text baseline
        
        cv2.putText(
            glyph_layer, glyph, (px, py),
            font, font_scale, text_color,
            thickness, cv2.LINE_AA
        )
    
    # Create glow/halo effect
    if glow_radius > 0 and glow_intensity > 0:
        # Blur the glyph layer
        kernel = glow_radius if glow_radius % 2 == 1 else glow_radius + 1
        glow_layer = cv2.GaussianBlur(glyph_layer, (kernel, kernel), 0)
        
        # Add glow back onto glyphs (additive blend)
        glyph_layer = cv2.addWeighted(
            glyph_layer, 1.0,
            glow_layer, glow_intensity,
            0
        )
    
    # Create dark background with subtle original frame hint
    output = (frame * 0.15).astype(np.uint8)
    
    # Composite glyphs onto output
    output = cv2.add(output, glyph_layer)
    
    return output


# =============================================================================
# THERMAL SCAN EFFECT (Skepta "Ignorance is Bliss" style)
# =============================================================================

def draw_thermal_scan(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Thermal Scan effect: Skepta album cover style thermal imaging.
    
    Hot areas (skin, bodies) = orange/yellow/red
    Cold areas (background) = cyan/blue/teal
    Creates that iconic thermal camera look.
    """
    h, w = frame.shape[:2]
    
    # Convert to grayscale (intensity = "temperature")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply slight blur for smoother thermal look
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast for more dramatic thermal effect
    gray = cv2.equalizeHist(gray)
    
    # Create thermal colormap
    # We'll map grayscale to: cold (cyan/blue) -> warm (yellow/orange/red)
    output = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize gray to 0-1
    normalized = gray.astype(np.float32) / 255.0
    
    # Custom thermal colormap (Skepta style):
    # Low values (0-0.3): Cyan/Teal (cold)
    # Mid values (0.3-0.6): Green/Yellow transition  
    # High values (0.6-1.0): Orange/Red (hot)
    
    for y in range(h):
        for x in range(w):
            t = normalized[y, x]
            
            if t < 0.25:
                # Cold: Deep cyan/teal
                r = int(20 + t * 80)
                g = int(120 + t * 200)
                b = int(180 + t * 75)
            elif t < 0.45:
                # Cool: Teal to green
                blend = (t - 0.25) / 0.2
                r = int(40 + blend * 60)
                g = int(170 + blend * 50)
                b = int(200 - blend * 100)
            elif t < 0.6:
                # Warm: Green-yellow
                blend = (t - 0.45) / 0.15
                r = int(100 + blend * 100)
                g = int(220 - blend * 30)
                b = int(100 - blend * 80)
            elif t < 0.75:
                # Hot: Yellow-orange
                blend = (t - 0.6) / 0.15
                r = int(200 + blend * 55)
                g = int(190 - blend * 80)
                b = int(20 - blend * 20)
            else:
                # Very hot: Orange-red/white
                blend = (t - 0.75) / 0.25
                r = int(255)
                g = int(110 + blend * 100)
                b = int(0 + blend * 50)
            
            output[y, x] = [b, g, r]  # BGR format
    
    # Optional: Add subtle glow to hot areas
    hot_mask = (normalized > 0.6).astype(np.uint8) * 255
    if np.any(hot_mask):
        glow = cv2.GaussianBlur(output, (21, 21), 0)
        hot_mask_3d = np.stack([hot_mask] * 3, axis=-1) / 255.0
        output = cv2.addWeighted(output, 1.0, (glow * hot_mask_3d * 0.3).astype(np.uint8), 1.0, 0)
    
    return output


# Optimized thermal using numpy vectorization
def draw_thermal_scan_fast(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Thermal Scan effect - vectorized version for speed.
    """
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    
    # Use OpenCV's built-in colormap and modify it
    # COLORMAP_JET is close but we want more cyan in cold areas
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Shift colors to be more cyan-to-orange (Skepta style)
    # Swap blue channel with enhanced cyan
    b, g, r = cv2.split(thermal)
    
    # Boost cyan in cold areas (where original gray is low)
    cold_mask = gray < 100
    g[cold_mask] = np.minimum(255, g[cold_mask].astype(np.int32) + 40).astype(np.uint8)
    b[cold_mask] = np.minimum(255, b[cold_mask].astype(np.int32) + 20).astype(np.uint8)
    
    # Boost orange/yellow in hot areas
    hot_mask = gray > 150
    r[hot_mask] = np.minimum(255, r[hot_mask].astype(np.int32) + 30).astype(np.uint8)
    
    output = cv2.merge([b, g, r])
    
    # Add glow to hottest areas
    very_hot = gray > 200
    if np.any(very_hot):
        glow = cv2.GaussianBlur(output, (15, 15), 0)
        mask = very_hot.astype(np.float32)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask_3d = np.stack([mask] * 3, axis=-1)
        output = cv2.addWeighted(output, 1.0, (glow * mask_3d * 0.4).astype(np.uint8), 1.0, 0)
    
    return output


# =============================================================================
# MATRIX MODE EFFECT (Green data rain)
# =============================================================================

def draw_matrix_mode(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Matrix Mode effect: Green digital rain over subject.
    
    Creates that iconic Matrix movie look with falling green characters
    concentrated on the subject.
    """
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create dark green-tinted background
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[:, :, 1] = (gray * 0.15).astype(np.uint8)  # Subtle green hint
    
    # Matrix characters
    matrix_chars = "01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    char_height = 14
    char_width = 10
    
    # Create columns of falling characters
    num_cols = w // char_width
    
    # Use frame_idx to animate the rain
    np.random.seed(42)  # Consistent random for each frame position
    
    for col in range(num_cols):
        x = col * char_width
        
        # Each column has a "head" position that moves down
        col_seed = col * 1000
        head_y = ((frame_idx * 3 + col_seed) % (h + 200)) - 100
        
        # Draw trail of characters above the head
        trail_length = random.randint(10, 25)
        
        for i in range(trail_length):
            y = head_y - i * char_height
            if 0 <= y < h:
                # Brightness fades as we go up the trail
                brightness = 1.0 - (i / trail_length) * 0.8
                
                # Check if this position is on the subject (brighter original)
                if 0 <= y < h and 0 <= x < w:
                    subject_brightness = gray[int(y), int(x)] / 255.0
                    brightness *= (0.5 + subject_brightness * 0.5)
                
                # Green color with varying intensity
                green = int(255 * brightness)
                color = (0, green, int(green * 0.3))  # Slight cyan tint
                
                # Random character
                char = random.choice(matrix_chars)
                
                # Head is brightest (white-green)
                if i == 0:
                    color = (200, 255, 200)
                
                cv2.putText(output, char, (x, int(y)), font, font_scale, 
                           color, 1, cv2.LINE_AA)
    
    # Blend with original to show subject
    subject_blend = 0.2
    output = cv2.addWeighted(output, 1.0, frame, subject_blend, 0)
    
    # Add scanlines for CRT feel
    for y in range(0, h, 3):
        output[y, :] = (output[y, :] * 0.85).astype(np.uint8)
    
    return output


def apply_text_effect(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray | None:
    """
    Apply text-based effect if preset has text_mode set.
    
    Returns the processed frame, or None if no text effect applies.
    """
    text_mode = preset.get("text_mode")
    
    if text_mode == "data_body":
        return draw_data_body(frame, preset, colors)
    elif text_mode == "numeric_aura" or text_mode == "number_cloud":
        return draw_number_cloud(frame, preset, colors)
    elif text_mode == "blob_track":
        return draw_blob_track(frame, preset, colors)
    elif text_mode == "particle_silhouette":
        return draw_particle_silhouette(frame, preset, colors)
    elif text_mode == "thermal_scan":
        return draw_thermal_scan_fast(frame, preset, colors)
    elif text_mode == "matrix_mode":
        return draw_matrix_mode(frame, preset, colors, frame_idx=0)
    elif text_mode == "contour_trace":
        return draw_contour_trace(frame, preset, colors)
    
    return None


# =============================================================================
# BLOB TRACKING EFFECT (TouchDesigner style)
# =============================================================================

def draw_blob_track(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Blob Track effect: Clean minimal tracking - TouchDesigner style.
    
    Simple thin white rectangles with:
    - Clean thin box outlines (no corner brackets)
    - Crosshairs on only ~30% of boxes
    - White connection lines between nearby blobs
    - Minimal ID labels
    """
    h, w = frame.shape[:2]
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get parameters
    blur_size = preset.get("blob_blur", 11)
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Use edge detection + Otsu for robust detection
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, None, iterations=2)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_or(edges, otsu)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Show original video with slight darkening
    bg_alpha = preset.get("bg_alpha", 0.75)
    output = (frame * bg_alpha).astype(np.uint8)
    
    # Filter contours
    min_area = preset.get("min_blob_area", 150)
    max_blobs = preset.get("max_blobs", 100)
    
    valid_contours = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) > min_area]
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    valid_contours = valid_contours[:max_blobs]
    
    if not valid_contours:
        return output
    
    # Colors - clean white
    box_color = (255, 255, 255)
    line_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    
    # Collect blob data
    blob_centers = []
    blob_boxes = []
    
    for idx, (area, contour) in enumerate(valid_contours):
        x, y, bw, bh = cv2.boundingRect(contour)
        center_x = x + bw // 2
        center_y = y + bh // 2
        blob_centers.append((center_x, center_y))
        blob_boxes.append((x, y, bw, bh, idx, area))
    
    # Draw WHITE connection lines first
    max_connection_dist = preset.get("max_connection_dist", 180)
    for i in range(len(blob_centers)):
        for j in range(i + 1, len(blob_centers)):
            p1 = blob_centers[i]
            p2 = blob_centers[j]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist < max_connection_dist:
                alpha = 0.6 * (1.0 - (dist / max_connection_dist))
                color = tuple(int(255 * alpha) for _ in range(3))
                cv2.line(output, p1, p2, color, 1, cv2.LINE_AA)
    
    # Draw each blob - SIMPLE THIN RECTANGLES
    start_id = preset.get("start_id", 100)
    
    for (x, y, bw, bh, idx, area) in blob_boxes:
        blob_id = start_id + idx
        center_x = x + bw // 2
        center_y = y + bh // 2
        
        # Simple thin rectangle outline (NO corner brackets)
        cv2.rectangle(output, (x, y), (x + bw, y + bh), box_color, 1, cv2.LINE_AA)
        
        # Crosshair on only ~30% of boxes (randomly selected by idx)
        if idx % 3 == 0:
            cross_size = 5
            cv2.line(output, (center_x - cross_size, center_y), 
                    (center_x + cross_size, center_y), box_color, 1, cv2.LINE_AA)
            cv2.line(output, (center_x, center_y - cross_size), 
                    (center_x, center_y + cross_size), box_color, 1, cv2.LINE_AA)
        
        # Small ID label - only on larger blobs
        if area > 500:
            id_label = f"{blob_id}"
            cv2.putText(output, id_label, (x + 2, y - 4), font, font_scale, 
                       (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output, id_label, (x + 2, y - 4), font, font_scale, 
                       box_color, 1, cv2.LINE_AA)
    
    return output


# =============================================================================
# PARTICLE SILHOUETTE EFFECT (bb.dere style)
# =============================================================================

def draw_particle_silhouette(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Particle Silhouette effect: Dense ethereal point cloud - TouchDesigner/AE quality.
    
    Creates thousands of particles forming a glowing silhouette with:
    - Multiple depth layers (foreground bright, background dim)
    - Soft ethereal glow with multi-pass blur
    - Dynamic particle sizing for depth perception
    """
    h, w = frame.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get preset params
    particle_density = preset.get("particle_density", 0.06)
    brightness_threshold = preset.get("brightness_threshold", 20)
    glow_intensity = preset.get("particle_glow", 1.0)
    
    # Create layers for depth effect
    layer_back = np.zeros((h, w, 3), dtype=np.uint8)
    layer_mid = np.zeros((h, w, 3), dtype=np.uint8)
    layer_front = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Particle colors - warm whites with slight color variation
    color_back = (180, 190, 200)   # Cooler, dimmer - background
    color_mid = (220, 225, 235)    # Neutral - midground
    color_front = (255, 250, 245)  # Warm, bright - foreground
    
    # Find subject pixels using multiple methods
    # 1. Brightness-based
    bright_mask = gray > brightness_threshold
    
    # 2. Edge detection for crisp boundaries
    edges = cv2.Canny(gray, 20, 70)
    edge_mask = edges > 0
    
    # 3. Gradient magnitude for texture detail
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    gradient_mask = gradient > 15
    
    # Combine all masks
    combined_mask = np.logical_or(bright_mask, np.logical_or(edge_mask, gradient_mask))
    all_coords = np.column_stack(np.where(combined_mask))
    
    if len(all_coords) == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    
    # Sample particles - very high density
    num_particles = int(len(all_coords) * particle_density)
    num_particles = min(num_particles, 50000)  # Higher cap for ultra-dense
    num_particles = max(num_particles, 3000)
    
    if len(all_coords) > num_particles:
        indices = np.random.choice(len(all_coords), size=num_particles, replace=False)
        sampled = all_coords[indices]
    else:
        sampled = all_coords
    
    # Draw particles in THREE layers for parallax depth
    for (row, col) in sampled:
        base_brightness = gray[row, col] / 255.0
        
        # Assign to layer based on brightness + randomness
        layer_chance = random.random()
        
        if layer_chance < 0.3:
            # Background layer - more scatter, dimmer
            scatter = random.randint(-4, 4)
            px = max(0, min(col + scatter, w - 1))
            py = max(0, min(row + scatter, h - 1))
            brightness = base_brightness * (0.3 + random.random() * 0.3)
            color = tuple(int(c * brightness) for c in color_back)
            layer_back[py, px] = color
            
        elif layer_chance < 0.7:
            # Midground layer - medium scatter
            scatter = random.randint(-2, 2)
            px = max(0, min(col + scatter, w - 1))
            py = max(0, min(row + scatter, h - 1))
            brightness = base_brightness * (0.5 + random.random() * 0.4)
            color = tuple(int(c * brightness) for c in color_mid)
            layer_mid[py, px] = color
            
        else:
            # Foreground layer - minimal scatter, brightest
            scatter = random.randint(-1, 1)
            px = max(0, min(col + scatter, w - 1))
            py = max(0, min(row + scatter, h - 1))
            brightness = base_brightness * (0.7 + random.random() * 0.3)
            color = tuple(int(c * brightness) for c in color_front)
            layer_front[py, px] = color
    
    # Apply different blur levels per layer for depth of field
    layer_back_glow = cv2.GaussianBlur(layer_back, (31, 31), 0)
    layer_mid_glow = cv2.GaussianBlur(layer_mid, (15, 15), 0)
    layer_front_glow = cv2.GaussianBlur(layer_front, (7, 7), 0)
    
    # Composite layers: back → mid → front
    output = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add back layer with heavy glow
    output = cv2.addWeighted(output, 1.0, layer_back_glow, 0.4, 0)
    output = cv2.add(output, layer_back)
    
    # Add mid layer
    output = cv2.addWeighted(output, 1.0, layer_mid_glow, 0.5, 0)
    output = cv2.add(output, layer_mid)
    
    # Add front layer with subtle glow
    output = cv2.addWeighted(output, 1.0, layer_front_glow, 0.3, 0)
    output = cv2.add(output, layer_front)
    
    # Final ethereal glow pass
    if glow_intensity > 0:
        final_glow = cv2.GaussianBlur(output, (25, 25), 0)
        output = cv2.addWeighted(output, 1.0, final_glow, glow_intensity * 0.5, 0)
    
    return output


# =============================================================================
# NUMERIC AURA EFFECT (Subject isolation - numbers on subject, video background)
# =============================================================================

def draw_number_cloud(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Numeric Aura effect: Subject becomes numbers, background stays as video.
    
    Isolates the main subject and overlays numbers ONLY on the subject,
    while keeping the background as pristine original video.
    """
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get params
    font_scale = preset.get("number_font_scale", 0.28)
    start_number = preset.get("start_number", 19000)
    
    # === SUBJECT ISOLATION ===
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Edge detection for subject boundaries  
    edges = cv2.Canny(blurred, 25, 70)
    
    # Dilate to create connected regions
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=5)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    # Find contours - ONLY THE LARGEST ONE
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    subject_mask = np.zeros((h, w), dtype=np.uint8)
    
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        min_area = h * w * 0.03  # At least 3% of frame
        
        for contour in sorted_contours[:1]:
            if cv2.contourArea(contour) > min_area:
                hull = cv2.convexHull(contour)
                cv2.fillPoly(subject_mask, [hull], 255)
                break
    
    # Smooth mask edges
    subject_mask = cv2.GaussianBlur(subject_mask, (15, 15), 0)
    _, subject_mask = cv2.threshold(subject_mask, 80, 255, cv2.THRESH_BINARY)
    
    # === OUTPUT: Background = original video, Subject = dark + numbers ===
    output = frame.copy()
    
    # Darken ONLY the subject area (for number contrast)
    subject_darken = 0.15  # Subject becomes quite dark
    for c in range(3):
        output[:, :, c] = np.where(
            subject_mask > 0,
            (frame[:, :, c] * subject_darken).astype(np.uint8),
            frame[:, :, c]  # Background stays FULL brightness
        )
    
    # Find points within subject
    subject_points = np.column_stack(np.where(subject_mask > 0))
    
    if len(subject_points) == 0:
        return frame  # No subject, return original
    
    # Sample for numbers
    max_numbers = preset.get("max_numbers", 4000)
    density = preset.get("number_density", 0.05)
    num_to_sample = min(max_numbers, int(len(subject_points) * density))
    num_to_sample = max(500, num_to_sample)
    
    if len(subject_points) > num_to_sample:
        indices = np.random.choice(len(subject_points), size=num_to_sample, replace=False)
        sampled = subject_points[indices]
    else:
        sampled = subject_points
    
    # Draw numbers ONLY on subject
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    
    for idx, (row, col) in enumerate(sampled):
        number = start_number + idx
        text = str(number)
        
        px = max(0, min(col, w - 35))
        py = max(10, min(row, h - 2))
        
        # Brightness varies with original image
        orig_brightness = gray[row, col] / 255.0
        brightness = 0.5 + 0.5 * orig_brightness
        color = tuple(int(c * brightness) for c in text_color)
        
        cv2.putText(output, text, (px, py), font, font_scale, color, 1, cv2.LINE_AA)
    
    return output


# =============================================================================
# MOTION TRACE EFFECT (Clean optical flow lines)
# =============================================================================

def draw_motion_trace(
    frame: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Motion Trace effect: Clean flowing lines following optical flow.
    
    Creates elegant curved traces that visualize motion vectors,
    similar to TouchDesigner's trail-based particle systems.
    """
    h, w = frame.shape[:2]
    
    # Create output - dark background with subtle original
    bg_alpha = preset.get("bg_alpha", 0.1)
    output = (frame * bg_alpha).astype(np.uint8)
    
    # Get trace params
    trace_color = colors.get("trail", (255, 255, 255))
    point_color = colors.get("point", (255, 255, 255))
    line_thickness = preset.get("trace_thickness", 1)
    point_size = preset.get("point_size", 2)
    trail_length = preset.get("trail_length", 20)
    
    # Draw motion traces
    for point in points:
        if len(point.trace) < 2:
            continue
        
        # Get visible portion of trail
        visible_trace = point.trace[-trail_length:]
        
        # Draw trail with fading opacity
        for i in range(len(visible_trace) - 1):
            alpha = (i + 1) / len(visible_trace)
            color = tuple(int(c * alpha * 0.8) for c in trace_color)
            
            pt1 = (int(visible_trace[i][0]), int(visible_trace[i][1]))
            pt2 = (int(visible_trace[i + 1][0]), int(visible_trace[i + 1][1]))
            
            cv2.line(output, pt1, pt2, color, line_thickness, cv2.LINE_AA)
        
        # Draw endpoint
        if visible_trace:
            endpoint = visible_trace[-1]
            cv2.circle(
                output, 
                (int(endpoint[0]), int(endpoint[1])), 
                point_size, 
                point_color, 
                -1, 
                cv2.LINE_AA
            )
    
    # Add subtle glow
    glow_intensity = preset.get("glow_intensity", 0.2)
    if glow_intensity > 0:
        glow = cv2.GaussianBlur(output, (11, 11), 0)
        output = cv2.addWeighted(output, 1.0, glow, glow_intensity, 0)
    
    return output


# =============================================================================
# CONTOUR TRACE EFFECT (Edge-based visualization)
# =============================================================================

def draw_contour_trace(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Contour Trace: Pure minimalist edge visualization.
    
    Creates clean white edges on black background with subtle glow.
    """
    h, w = frame.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Multi-scale edge detection for cleaner lines
    edges1 = cv2.Canny(filtered, 20, 60)
    edges2 = cv2.Canny(filtered, 40, 120)
    
    # Combine edges
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Optional: thin edges using morphological operations
    if not preset.get("thick_edges", False):
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Create output - pure black background
    output = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Pure white edges
    line_color = (255, 255, 255)
    output[edges > 0] = line_color
    
    # Add glow for ethereal effect
    glow_intensity = preset.get("glow_intensity", 0.4)
    if glow_intensity > 0:
        # Multi-layer glow for depth
        glow1 = cv2.GaussianBlur(output, (5, 5), 0)
        glow2 = cv2.GaussianBlur(output, (15, 15), 0)
        output = cv2.addWeighted(output, 1.0, glow1, glow_intensity * 0.6, 0)
        output = cv2.addWeighted(output, 1.0, glow2, glow_intensity * 0.3, 0)
    
    return output


# =============================================================================
# CATODIC CUBE / DEPTH EFFECTS
# =============================================================================

def draw_catodic_cube(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int,
    points: list[TrackedPoint],
) -> np.ndarray:
    """
    Catodic Cube effect: screen breaking into 3D wireframe depth.
    
    Creates the illusion of looking "into" the display with:
    - Perspective wireframe grid receding to vanishing point
    - RGB channel split (chromatic aberration)
    - Motion-triggered glitch effects
    """
    h, w = frame.shape[:2]
    
    # Get preset params
    depth_amount = preset.get("depth_amount", 0.4)
    wireframe_layers = preset.get("wireframe_layers", 5)
    wireframe_intensity = preset.get("wireframe_intensity", 0.7)
    rgb_offset = preset.get("rgb_offset_px", 4)
    glitch_freq = preset.get("glitch_frequency", 8)
    glitch_strength = preset.get("glitch_strength", 0.3)
    motion_amplify = preset.get("motion_amplify", 1.5)
    
    # Calculate motion intensity from alive points
    alive_count = sum(1 for p in points if p.alive)
    motion_factor = min(1.0, alive_count / 50.0) * motion_amplify
    
    # Start with darkened original
    output = (frame * 0.3).astype(np.uint8)
    
    # Create wireframe layer
    wireframe = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Vanishing point (center of frame, slightly offset for dynamism)
    vp_x = w // 2 + int(np.sin(frame_idx * 0.05) * 20 * motion_factor)
    vp_y = h // 2 + int(np.cos(frame_idx * 0.07) * 15 * motion_factor)
    
    # Draw nested perspective rectangles
    line_color = colors.get("line", (255, 200, 100))
    
    for i in range(wireframe_layers):
        # Progress from outer (0) to inner (1)
        t = i / max(1, wireframe_layers - 1)
        
        # Interpolate corners from frame edges to vanishing point
        margin = int(20 * (1 - t))  # Small margin from edges
        
        # Outer rectangle corners
        outer_tl = (margin, margin)
        outer_tr = (w - margin, margin)
        outer_br = (w - margin, h - margin)
        outer_bl = (margin, h - margin)
        
        # Lerp towards vanishing point based on depth
        depth_t = t * depth_amount
        
        def lerp_point(outer, vp, t):
            return (
                int(outer[0] + (vp[0] - outer[0]) * t),
                int(outer[1] + (vp[1] - outer[1]) * t),
            )
        
        tl = lerp_point(outer_tl, (vp_x, vp_y), depth_t)
        tr = lerp_point(outer_tr, (vp_x, vp_y), depth_t)
        br = lerp_point(outer_br, (vp_x, vp_y), depth_t)
        bl = lerp_point(outer_bl, (vp_x, vp_y), depth_t)
        
        # Fade inner layers
        alpha = wireframe_intensity * (1.0 - t * 0.5)
        color = tuple(int(c * alpha) for c in line_color)
        thickness = 2 if i == 0 else 1
        
        # Draw rectangle
        cv2.line(wireframe, tl, tr, color, thickness, cv2.LINE_AA)
        cv2.line(wireframe, tr, br, color, thickness, cv2.LINE_AA)
        cv2.line(wireframe, br, bl, color, thickness, cv2.LINE_AA)
        cv2.line(wireframe, bl, tl, color, thickness, cv2.LINE_AA)
        
        # Draw depth lines from corners to vanishing point (sparse)
        if i == 0:
            depth_color = tuple(int(c * 0.3) for c in line_color)
            cv2.line(wireframe, outer_tl, (vp_x, vp_y), depth_color, 1, cv2.LINE_AA)
            cv2.line(wireframe, outer_tr, (vp_x, vp_y), depth_color, 1, cv2.LINE_AA)
            cv2.line(wireframe, outer_br, (vp_x, vp_y), depth_color, 1, cv2.LINE_AA)
            cv2.line(wireframe, outer_bl, (vp_x, vp_y), depth_color, 1, cv2.LINE_AA)
    
    # Draw tracked points as depth lines
    point_color = colors.get("point", (255, 255, 255))
    for point in points:
        if not point.alive:
            continue
        px, py = point.position.astype(int)
        if 0 <= px < w and 0 <= py < h:
            # Draw line from point towards vanishing point
            line_len = int(30 * depth_amount * motion_factor)
            dx = vp_x - px
            dy = vp_y - py
            dist = max(1, np.sqrt(dx*dx + dy*dy))
            end_x = int(px + (dx / dist) * line_len)
            end_y = int(py + (dy / dist) * line_len)
            cv2.line(wireframe, (px, py), (end_x, end_y), point_color, 1, cv2.LINE_AA)
            cv2.circle(wireframe, (px, py), 2, point_color, -1, cv2.LINE_AA)
    
    # Add glow to wireframe
    glow = cv2.GaussianBlur(wireframe, (15, 15), 0)
    wireframe = cv2.addWeighted(wireframe, 1.0, glow, 0.5, 0)
    
    # Composite wireframe onto output
    output = cv2.add(output, wireframe)
    
    # Apply RGB split (chromatic aberration)
    if rgb_offset > 0:
        output = apply_rgb_split(output, rgb_offset, motion_factor)
    
    # Apply glitch effect on certain frames
    if glitch_freq > 0 and frame_idx % glitch_freq == 0:
        output = apply_glitch(output, glitch_strength, frame_idx)
    
    return output


def apply_rgb_split(
    frame: np.ndarray,
    offset: int,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Apply chromatic aberration / RGB channel split.
    
    Shifts R channel left and B channel right for a glitchy CRT look.
    """
    h, w = frame.shape[:2]
    actual_offset = int(offset * intensity)
    
    if actual_offset <= 0:
        return frame
    
    # Split channels (BGR format)
    b, g, r = cv2.split(frame)
    
    # Shift red channel left
    r_shifted = np.zeros_like(r)
    if actual_offset < w:
        r_shifted[:, :w-actual_offset] = r[:, actual_offset:]
    
    # Shift blue channel right  
    b_shifted = np.zeros_like(b)
    if actual_offset < w:
        b_shifted[:, actual_offset:] = b[:, :w-actual_offset]
    
    # Merge with original green channel
    result = cv2.merge([b_shifted, g, r_shifted])
    
    # Blend with original to control intensity
    return cv2.addWeighted(frame, 0.3, result, 0.7, 0)


def apply_glitch(
    frame: np.ndarray,
    strength: float,
    seed: int,
) -> np.ndarray:
    """
    Apply horizontal slice displacement glitch effect.
    
    Randomly shifts horizontal bands of the image left/right.
    """
    h, w = frame.shape[:2]
    output = frame.copy()
    
    # Use seed for reproducible randomness per frame
    rng = np.random.default_rng(seed)
    
    # Number of glitch slices
    num_slices = rng.integers(3, 8)
    
    for _ in range(num_slices):
        # Random slice position and height
        # Ensure enough room for at least a 5px slice
        if h <= 14:
            continue
        y_start = rng.integers(0, h - 10)
        remaining = h - y_start
        if remaining < 5:
            continue
        slice_height = rng.integers(5, min(40, remaining))
        y_end = y_start + slice_height
        
        # Random horizontal shift
        max_shift = int(w * strength * 0.1)
        if max_shift > 0:
            shift = rng.integers(-max_shift, max_shift + 1)
            
            if shift != 0:
                # Shift the slice horizontally
                shifted_slice = np.zeros_like(output[y_start:y_end])
                if shift > 0:
                    shifted_slice[:, shift:] = output[y_start:y_end, :w-shift]
                else:
                    shifted_slice[:, :w+shift] = output[y_start:y_end, -shift:]
                output[y_start:y_end] = shifted_slice
    
    return output


def apply_cube_effect(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int,
    points: list[TrackedPoint],
) -> np.ndarray | None:
    """
    Apply cube/depth effect if preset has cube_mode set.
    
    Returns the processed frame, or None if no cube effect applies.
    """
    if not preset.get("cube_mode", False):
        return None
    
    return draw_catodic_cube(frame, preset, colors, frame_idx, points)