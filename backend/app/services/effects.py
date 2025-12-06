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
    text_result = apply_text_effect(frame, preset, colors, frame_idx=frame_idx, points=points)
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
    text_result = apply_text_effect(frame, preset, colors, frame_idx=frame_idx, points=points)
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
    Thermal Scan effect: EXACT Skepta "Ignorance is Bliss" style.
    Uses the fast vectorized version.
    """
    return draw_thermal_scan_fast(frame, preset, colors)


def draw_thermal_scan_slow(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Thermal Scan effect (slow pixel-by-pixel version - not used).
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
    Thermal Scan effect - EXACT Skepta "Ignorance is Bliss" colors.
    
    Precise color match from the album cover:
    - Background: Dark saturated teal #006B6B (deep cyan-teal)
    - Skin: Vivid orange #FF6600 to #FF8800
    - Hot: Yellow-orange #FFAA00 to #FFCC44
    """
    h, w = frame.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Strong CLAHE for dramatic contrast like the album
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Smooth for thermal look
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Create output
    output = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize
    norm = gray.astype(np.float32) / 255.0
    
    # === SKEPTA EXACT COLORS (BGR format) ===
    # Deep teal: RGB(0, 107, 107) = BGR(107, 107, 0) - the dark cyan from album
    # Orange: RGB(255, 102, 0) = BGR(0, 102, 255) - vivid orange skin
    # Yellow-orange: RGB(255, 170, 0) = BGR(0, 170, 255) - hot spots
    
    # COLD: Deep teal background - bottom 45%
    cold_mask = norm < 0.45
    t = norm[cold_mask] / 0.45
    # Start very dark teal, get slightly brighter
    output[cold_mask, 0] = (100 + t * 20).astype(np.uint8)    # B: 100->120
    output[cold_mask, 1] = (90 + t * 25).astype(np.uint8)     # G: 90->115
    output[cold_mask, 2] = (0 + t * 15).astype(np.uint8)      # R: 0->15
    
    # TRANSITION: Teal to orange - 45% to 55%
    trans_mask = (norm >= 0.45) & (norm < 0.55)
    t = (norm[trans_mask] - 0.45) / 0.10
    output[trans_mask, 0] = (120 - t * 118).astype(np.uint8)  # B: 120->2
    output[trans_mask, 1] = (115 - t * 15).astype(np.uint8)   # G: 115->100
    output[trans_mask, 2] = (15 + t * 240).astype(np.uint8)   # R: 15->255
    
    # WARM: Vivid orange - 55% to 70%
    warm_mask = (norm >= 0.55) & (norm < 0.70)
    t = (norm[warm_mask] - 0.55) / 0.15
    output[warm_mask, 0] = 2                                   # B: very low
    output[warm_mask, 1] = (100 + t * 70).astype(np.uint8)    # G: 100->170
    output[warm_mask, 2] = 255                                 # R: max orange
    
    # HOT: Yellow-orange - 70% to 85%
    hot_mask = (norm >= 0.70) & (norm < 0.85)
    t = (norm[hot_mask] - 0.70) / 0.15
    output[hot_mask, 0] = (2 + t * 20).astype(np.uint8)       # B: 2->22
    output[hot_mask, 1] = (170 + t * 50).astype(np.uint8)     # G: 170->220
    output[hot_mask, 2] = 255                                  # R: max
    
    # VERY HOT: Bright yellow - top 15%
    very_hot_mask = norm >= 0.85
    t = (norm[very_hot_mask] - 0.85) / 0.15
    t = np.clip(t, 0, 1)
    output[very_hot_mask, 0] = (22 + t * 50).astype(np.uint8)   # B: 22->72
    output[very_hot_mask, 1] = (220 + t * 35).astype(np.uint8)  # G: 220->255
    output[very_hot_mask, 2] = 255                               # R: max
    
    # Subtle glow on hot areas
    hot_areas = norm > 0.55
    if np.any(hot_areas):
        glow = cv2.GaussianBlur(output, (25, 25), 0)
        mask = hot_areas.astype(np.float32)
        mask = cv2.GaussianBlur(mask, (35, 35), 0)
        mask_3d = np.stack([mask] * 3, axis=-1)
        output = cv2.addWeighted(output, 1.0, (glow * mask_3d * 0.3).astype(np.uint8), 1.0, 0)
    
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
    frame_idx: int = 0,
    points: list[TrackedPoint] | None = None,
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
        return draw_matrix_mode(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "contour_trace":
        return draw_contour_trace(frame, preset, colors)
    elif text_mode == "motion_flow":
        # Ensure points are available
        return draw_motion_trace(frame, points or [], preset, colors)
    elif text_mode == "signal_map":
        return draw_signal_map(frame, preset, colors, frame_idx=frame_idx)
    # === NEW EFFECTS v2 ===
    elif text_mode == "codenet_overlay":
        return draw_codenet_overlay(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "code_shadow":
        return draw_code_shadow(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "binary_bloom":
        return draw_binary_bloom(frame, preset, colors, frame_idx=frame_idx)
    
    return None


# =============================================================================
# SIGNAL MAP EFFECT (Data visualization / bit mapping style)
# =============================================================================

def draw_signal_map(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Signal Map effect: Data visualization overlay inspired by surveillance/art.
    
    Features:
    - Blue thin rectangle outlines on detected motion/objects
    - Green/red filled boxes with scanline patterns
    - Random hex/code text labels
    - Small marker squares at tracked points
    """
    import random
    h, w = frame.shape[:2]
    
    # Color scheme (BGR)
    blue_outline = (255, 150, 50)   # Blue for outlines
    green_fill = (80, 200, 80)      # Green boxes
    red_fill = (60, 60, 200)        # Red/maroon boxes
    cyan_marker = (255, 255, 100)   # Cyan small squares
    white_text = (255, 255, 255)    # Text
    
    # Code prefixes
    code_prefixes = ["REP", "@E", "ID:", "+EP", "@M", "RE@", "REPR", "@PROC", "E/", "@X", "///"]
    
    # Keep original visible
    output = frame.copy()
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find objects
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Dilate edges to form regions
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort by area
    min_area = h * w * 0.005  # At least 0.5% of frame
    max_area = h * w * 0.6    # At most 60% of frame
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append((contour, area))
    
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    valid_contours = valid_contours[:15]  # Max 15 tracked objects
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    random.seed(frame_idx // 3)  # Consistent randomness
    
    for idx, (contour, area) in enumerate(valid_contours):
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # === BLUE OUTLINE ===
        cv2.rectangle(output, (x, y), (x + bw, y + bh), blue_outline, 1, cv2.LINE_AA)
        
        # === CORNER MARKERS (cyan squares) ===
        sq = 3
        cv2.rectangle(output, (x-sq, y-sq), (x+sq, y+sq), cyan_marker, -1)
        cv2.rectangle(output, (x+bw-sq, y-sq), (x+bw+sq, y+sq), cyan_marker, -1)
        cv2.rectangle(output, (x-sq, y+bh-sq), (x+sq, y+bh+sq), cyan_marker, -1)
        cv2.rectangle(output, (x+bw-sq, y+bh-sq), (x+bw+sq, y+bh+sq), cyan_marker, -1)
        
        # === DATA CODE LABEL ===
        prefix = code_prefixes[(idx + frame_idx // 8) % len(code_prefixes)]
        suffix = chr(65 + (idx + frame_idx // 15) % 26)
        num = (idx * 17 + frame_idx) % 100
        code = f"{prefix}{suffix}{num:02d}" if random.random() > 0.5 else f"{prefix}{suffix}"
        cv2.putText(output, code, (x, y - 4), font, 0.32, white_text, 1, cv2.LINE_AA)
        
        # === SCANLINE DATA BOXES ===
        box_w = min(max(bw // 4, 20), 50)
        box_h = min(max(bh // 5, 12), 30)
        
        # Green or red box (alternating + random)
        fill_color = green_fill if (idx + frame_idx // 20) % 3 != 0 else red_fill
        
        # Position inside bounding box
        bx = x + 3 + (idx * 7) % max(1, bw - box_w - 6)
        by = y + 3 + (idx * 11) % max(1, bh - box_h - 6)
        
        if bx + box_w < x + bw and by + box_h < y + bh:
            # Draw filled box with scanlines
            overlay = output.copy()
            cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), fill_color, -1)
            output[by:by+box_h, bx:bx+box_w] = cv2.addWeighted(
                output[by:by+box_h, bx:bx+box_w], 0.4,
                overlay[by:by+box_h, bx:bx+box_w], 0.6, 0
            )
            # Scanlines
            for sy in range(by, by + box_h, 2):
                cv2.line(output, (bx, sy), (bx + box_w, sy), (30, 30, 30), 1)
            # Blue outline on box
            cv2.rectangle(output, (bx, by), (bx + box_w, by + box_h), blue_outline, 1)
        
        # === SECONDARY BOX (sometimes) ===
        if idx % 2 == 0 and bw > 60 and bh > 60:
            bx2 = x + bw - box_w - 5
            by2 = y + bh - box_h - 5
            fill_color2 = red_fill if fill_color == green_fill else green_fill
            
            if bx2 > x + box_w:
                overlay2 = output.copy()
                cv2.rectangle(overlay2, (bx2, by2), (bx2 + box_w, by2 + box_h), fill_color2, -1)
                output[by2:by2+box_h, bx2:bx2+box_w] = cv2.addWeighted(
                    output[by2:by2+box_h, bx2:bx2+box_w], 0.4,
                    overlay2[by2:by2+box_h, bx2:bx2+box_w], 0.6, 0
                )
                for sy in range(by2, by2 + box_h, 2):
                    cv2.line(output, (bx2, sy), (bx2 + box_w, sy), (30, 30, 30), 1)
                cv2.rectangle(output, (bx2, by2), (bx2 + box_w, by2 + box_h), blue_outline, 1)
        
        # === SMALL TRACKING DOTS scattered in box ===
        num_dots = min(5, max(2, int(area / 5000)))
        for _ in range(num_dots):
            dx = x + random.randint(5, max(6, bw - 5))
            dy = y + random.randint(5, max(6, bh - 5))
            cv2.rectangle(output, (dx-1, dy-1), (dx+1, dy+1), cyan_marker, -1)
    
    return output


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
    - Clean thin box outlines (no crosshairs)
    - White connection lines between nearby blobs
    - NO boxes touching frame edges (removes corner artifacts)
    """
    h, w = frame.shape[:2]
    
    # Edge margin - ignore detections touching frame borders
    edge_margin = 10
    
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
    
    # Filter contours - exclude ones touching frame edges
    min_area = preset.get("min_blob_area", 200)
    max_blobs = preset.get("max_blobs", 80)
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        # Skip if touching any edge of the frame
        if x <= edge_margin or y <= edge_margin:
            continue
        if x + bw >= w - edge_margin or y + bh >= h - edge_margin:
            continue
        valid_contours.append((area, contour))
    
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    valid_contours = valid_contours[:max_blobs]
    
    if not valid_contours:
        return output
    
    # Colors - clean white
    box_color = (255, 255, 255)
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
                cv2.line(output, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw each blob - SIMPLE THIN RECTANGLES with coordinates
    for (x, y, bw, bh, idx, area) in blob_boxes:
        # Simple thin rectangle outline
        cv2.rectangle(output, (x, y), (x + bw, y + bh), box_color, 1, cv2.LINE_AA)
        
        # Font size proportional to box size (smaller box = smaller text)
        box_size = min(bw, bh)
        dynamic_font = max(0.15, min(0.3, box_size / 200.0))  # Scale between 0.15-0.3
        
        # Coordinate label: x:123;y:456 (ABOVE the box, top-left corner)
        coord_label = f"x:{x};y:{y}"
        label_x = x
        label_y = y - 3  # Above the box
        
        # Make sure label doesn't go off screen
        if label_y < 8:
            label_y = y + int(box_size * 0.15) + 5  # Put inside if too close to top
        
        # Draw with shadow for readability
        cv2.putText(output, coord_label, (label_x + 1, label_y + 1), font, dynamic_font, 
                   (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(output, coord_label, (label_x, label_y), font, dynamic_font, 
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
    Numeric Aura effect - IMPRESSIVE data visualization.
    
    Multi-layer number cloud with:
    - Glowing blue numbers in varying sizes
    - Hex codes mixed with decimals
    - Depth layers (foreground/background)
    - Animated glow effect
    - Subject isolation with clean outline
    """
    import random
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start_number = preset.get("start_number", 19000)
    
    # Seed for consistent randomness
    random.seed(42)
    
    # === SUBJECT DETECTION ===
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 30, 100)
    
    kernel = np.ones((30, 30), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    subject_mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    
    if contours:
        best_contour = None
        best_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < h * w * 0.03:
                continue
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                centrality = 1.0 - (dist / max_dist)
                score = area * (0.4 + 0.6 * centrality)
                if score > best_score:
                    best_score = score
                    best_contour = contour
        if best_contour is not None:
            hull = cv2.convexHull(best_contour)
            cv2.fillPoly(subject_mask, [hull], 255)
    
    if np.sum(subject_mask) < h * w * 0.03 * 255:
        cv2.ellipse(subject_mask, (center_x, center_y), (w//3, h//2), 0, 0, 360, 255, -1)
    
    # === CREATE DARK BASE ===
    output = np.zeros_like(frame)
    
    # Keep background visible but dimmed
    bg_visible = (frame * 0.15).astype(np.uint8)
    output = np.where(subject_mask[:, :, np.newaxis] == 0, bg_visible, output)
    
    # === COLORS ===
    # BSOD Blue variations
    blue_bright = (255, 120, 0)    # Bright blue
    blue_mid = (200, 80, 0)        # Medium blue
    blue_dim = (140, 50, 0)        # Dim blue
    blue_glow = (255, 150, 50)     # Glowing blue
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # === LAYER 1: Background dim numbers (small, dense) ===
    for gy in range(8, h - 8, 12):
        for gx in range(4, w - 30, 38):
            if subject_mask[gy, gx] > 0:
                # Mix of hex and decimal
                if random.random() > 0.7:
                    text = f"0x{random.randint(0, 0xFFFF):04X}"
                else:
                    text = str(start_number + random.randint(0, 9999))
                
                cv2.putText(output, text, (gx, gy), font, 0.28, blue_dim, 1, cv2.LINE_AA)
    
    # === LAYER 2: Mid-layer numbers (medium size) ===
    for gy in range(15, h - 15, 22):
        for gx in range(10, w - 45, 55):
            if subject_mask[gy, gx] > 0:
                if random.random() > 0.6:
                    text = f"{random.randint(10000, 99999)}"
                else:
                    text = f"0x{random.randint(0, 0xFFFFFF):06X}"
                
                # Black shadow
                cv2.putText(output, text, (gx+1, gy+1), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output, text, (gx, gy), font, 0.4, blue_mid, 1, cv2.LINE_AA)
    
    # === LAYER 3: Foreground bright numbers (large, sparse, glowing) ===
    glow_layer = np.zeros_like(frame)
    for gy in range(25, h - 25, 40):
        for gx in range(15, w - 60, 80):
            if subject_mask[gy, gx] > 0:
                # Prominent numbers
                text = str(start_number + random.randint(0, 50000))
                
                # Draw glow (thicker, blurred later)
                cv2.putText(glow_layer, text, (gx, gy), font, 0.6, blue_glow, 3, cv2.LINE_AA)
                
                # Draw crisp text
                cv2.putText(output, text, (gx+1, gy+1), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(output, text, (gx, gy), font, 0.6, blue_bright, 1, cv2.LINE_AA)
    
    # Apply glow
    glow_layer = cv2.GaussianBlur(glow_layer, (21, 21), 0)
    output = cv2.addWeighted(output, 1.0, glow_layer, 0.5, 0)
    
    # === BRIGHT OUTLINE around subject ===
    contours_outline, _ = cv2.findContours(subject_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_outline:
        # Glow outline
        outline_glow = np.zeros_like(frame)
        cv2.drawContours(outline_glow, contours_outline, -1, blue_glow, 8, cv2.LINE_AA)
        outline_glow = cv2.GaussianBlur(outline_glow, (15, 15), 0)
        output = cv2.addWeighted(output, 1.0, outline_glow, 0.6, 0)
        
        # Crisp outline
        cv2.drawContours(output, contours_outline, -1, blue_bright, 2, cv2.LINE_AA)
    
    # === SCANLINE EFFECT ===
    for y in range(0, h, 4):
        output[y, :] = (output[y, :] * 0.85).astype(np.uint8)
    
    return output


# =============================================================================
# MOTION TRACE EFFECT (Dense optical flow visualization)
# =============================================================================

# Frame cache for optical flow computation
_motion_trace_prev_frame: np.ndarray | None = None

def draw_motion_trace(
    frame: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Motion Flow: Curved flowing lines that trace actual movement.
    
    Uses dense optical flow (Farneback) to detect motion and draws
    smooth curved lines only in regions with actual movement.
    Very different from Grid Trace - no grid, just organic flow lines.
    """
    global _motion_trace_prev_frame
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Parameters
    bg_alpha = preset.get("bg_alpha", 0.15)
    flow_color = preset.get("flow_color", colors.get("trail", (255, 200, 100)))
    thickness = preset.get("line_thickness", preset.get("trace_thickness", 2))
    min_flow_mag = preset.get("min_flow_mag", 1.5)  # Minimum flow magnitude to draw
    line_length_scale = preset.get("line_length_scale", 8)  # How long to draw lines
    sample_step = preset.get("sample_step", 12)  # Grid sampling step
    
    # Keep original video visible underneath
    output = (frame * bg_alpha).astype(np.uint8)
    
    # Need previous frame for optical flow
    if _motion_trace_prev_frame is None or _motion_trace_prev_frame.shape != gray.shape:
        _motion_trace_prev_frame = gray.copy()
        return output
    
    # Compute dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        _motion_trace_prev_frame, gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    # Update previous frame
    _motion_trace_prev_frame = gray.copy()
    
    # Compute flow magnitude and angle
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    
    # Color palette for variety (BGR)
    flow_colors = [
        flow_color,                    # Primary color from preset
        (255, 180, 80),               # Cyan
        (180, 255, 120),              # Light green-cyan
        (255, 220, 150),              # Light cyan
    ]
    
    # Sample points from motion regions and draw flow lines
    color_idx = 0
    for y in range(sample_step, h - sample_step, sample_step):
        for x in range(sample_step, w - sample_step, sample_step):
            mag = magnitude[y, x]
            
            # Only draw in regions with significant motion
            if mag < min_flow_mag:
                continue
            
            # Get flow vector
            dx = flow_x[y, x]
            dy = flow_y[y, x]
            
            # Scale line length by magnitude (capped)
            line_len = min(mag * line_length_scale, 60)
            
            # Calculate end point (following flow direction)
            x2 = int(x + dx * line_length_scale)
            y2 = int(y + dy * line_length_scale)
            
            # Create curved line using 3 points (start, mid-offset, end)
            # Add slight perpendicular curve for organic feel
            mid_x = (x + x2) // 2 + int(dy * 0.3)
            mid_y = (y + y2) // 2 - int(dx * 0.3)
            
            # Draw curved polyline
            pts = np.array([[x, y], [mid_x, mid_y], [x2, y2]], dtype=np.int32)
            
            # Alpha based on magnitude (stronger motion = brighter)
            alpha = min(1.0, mag / 8.0)
            color = flow_colors[color_idx % len(flow_colors)]
            draw_color = tuple(int(c * alpha) for c in color)
            
            # Draw smooth polyline
            cv2.polylines(output, [pts], False, draw_color, thickness, cv2.LINE_AA)
            
            # Draw small glowing head at end point
            if mag > min_flow_mag * 2:
                cv2.circle(output, (x2, y2), 2, (255, 255, 255), -1, cv2.LINE_AA)
            
            color_idx += 1
    
    # Add global glow for polish
    glow_intensity = preset.get("glow_intensity", 0.5)
    if glow_intensity > 0:
        blur = cv2.GaussianBlur(output, (9, 9), 0)
        output = cv2.addWeighted(output, 1.0, blur, glow_intensity, 0)
    
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


# =============================================================================
# CODENET OVERLAY (Feature network with Delaunay mesh + labels)
# =============================================================================

def draw_codenet_overlay(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    CodeNet Overlay: Feature-point network with labels.
    
    Inspired by the leaf-twirl reel with connected nodes and "codecore N" labels.
    - Detects Shi-Tomasi corners
    - Creates Delaunay triangulation for organic mesh
    - Gradient lines: short=red, medium=orange/yellow, long=white
    - Glowing cyan/white nodes
    - "codecore N" labels above each point
    """
    h, w = frame.shape[:2]
    
    # Parameters
    max_points = preset.get("max_points", 80)
    max_connect_dist = preset.get("connection_max_dist", 150)
    node_radius = preset.get("node_radius", 4)
    label_scale = preset.get("label_font_scale", 0.28)
    blend_alpha = preset.get("blend_alpha", 0.85)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast for better feature detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Detect Shi-Tomasi corners
    corners = cv2.goodFeaturesToTrack(
        enhanced,
        maxCorners=max_points,
        qualityLevel=0.02,
        minDistance=20,
        blockSize=7,
    )
    
    if corners is None or len(corners) < 3:
        return frame.copy()
    
    points = corners.reshape(-1, 2).astype(np.float32)
    
    # Create overlay layer
    overlay = np.zeros_like(frame)
    
    # Build Delaunay triangulation for organic mesh
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    
    valid_points = []
    for pt in points:
        x, y = pt
        if 0 < x < w - 1 and 0 < y < h - 1:
            subdiv.insert((float(x), float(y)))
            valid_points.append((int(x), int(y)))
    
    # Get edges from triangulation
    edge_list = subdiv.getEdgeList()
    
    # Draw connections with gradient colors based on distance
    for edge in edge_list:
        x1, y1, x2, y2 = edge
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        
        # Check bounds
        if not (0 <= p1[0] < w and 0 <= p1[1] < h):
            continue
        if not (0 <= p2[0] < w and 0 <= p2[1] < h):
            continue
        
        # Calculate distance
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if dist > max_connect_dist:
            continue
        
        # Color gradient based on distance: short=red, medium=orange/yellow, long=white
        t = min(dist / max_connect_dist, 1.0)
        
        if t < 0.33:
            # Red to orange
            r = 255
            g = int(100 * (t / 0.33))
            b = int(50 * (t / 0.33))
        elif t < 0.66:
            # Orange to yellow/white
            tt = (t - 0.33) / 0.33
            r = 255
            g = int(100 + 100 * tt)
            b = int(50 + 100 * tt)
        else:
            # Yellow to white
            tt = (t - 0.66) / 0.34
            r = 255
            g = int(200 + 55 * tt)
            b = int(150 + 105 * tt)
        
        line_color = (b, g, r)  # BGR
        thickness = max(1, 2 - int(t * 1.5))
        
        cv2.line(overlay, p1, p2, line_color, thickness, cv2.LINE_AA)
    
    # Draw nodes with glow
    glow_layer = np.zeros_like(frame)
    for idx, (px, py) in enumerate(valid_points):
        # Glow (larger, blurred)
        cv2.circle(glow_layer, (px, py), node_radius * 3, (255, 255, 200), -1)
        
        # Node point (cyan/white)
        cv2.circle(overlay, (px, py), node_radius, (255, 255, 255), -1)
        cv2.circle(overlay, (px, py), node_radius - 1, (255, 200, 100), -1)  # Cyan center
    
    # Blur glow layer
    glow_layer = cv2.GaussianBlur(glow_layer, (21, 21), 0)
    overlay = cv2.addWeighted(overlay, 1.0, glow_layer, 0.3, 0)
    
    # Draw labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (px, py) in enumerate(valid_points):
        label = f"codecore {idx + 1}"
        label_y = max(py - 8, 12)
        
        # Shadow
        cv2.putText(overlay, label, (px - 10 + 1, label_y + 1), font, label_scale, (0, 0, 0), 1, cv2.LINE_AA)
        # Text
        cv2.putText(overlay, label, (px - 10, label_y), font, label_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Blend with original
    output = cv2.addWeighted(frame, 1.0 - blend_alpha * 0.3, overlay, blend_alpha, 0)
    
    # Also add overlay on top
    mask = (overlay.sum(axis=2) > 0).astype(np.float32)
    mask = np.stack([mask] * 3, axis=-1)
    output = (output * (1 - mask * 0.6) + overlay * mask * 0.9).astype(np.uint8)
    
    return output


# =============================================================================
# CODESHADOW (ASCII/Matrix density effect)
# =============================================================================

def draw_code_shadow(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    CodeShadow: Dense ASCII characters forming the image.
    
    - Maps brightness to character density
    - Red for dark/background, green for bright/subject
    - Black background with CRT feel
    """
    h, w = frame.shape[:2]
    
    # Parameters
    cell_size = preset.get("cell_size", 8)
    char_palette = preset.get("char_palette", " .·:;=+*#@")
    color_dark = preset.get("color_dark", (0, 0, 140))      # Deep red BGR
    color_bright = preset.get("color_bright", (0, 200, 0))   # Green BGR
    threshold_split = preset.get("threshold_split", 0.45)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Create output canvas (black background)
    output = np.zeros((h, w, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell_size / 18.0
    
    # Grid dimensions
    rows = h // cell_size
    cols = w // cell_size
    
    # Sample random characters for variety
    random.seed(frame_idx + 42)
    
    # Map brightness to characters
    for row in range(rows):
        for col in range(cols):
            # Sample center of cell
            cy = row * cell_size + cell_size // 2
            cx = col * cell_size + cell_size // 2
            
            if cy >= h or cx >= w:
                continue
            
            # Get brightness (average of cell area)
            y1, y2 = row * cell_size, min((row + 1) * cell_size, h)
            x1, x2 = col * cell_size, min((col + 1) * cell_size, w)
            cell_brightness = np.mean(gray[y1:y2, x1:x2]) / 255.0
            
            # Skip very dark areas (sparse)
            if cell_brightness < 0.08:
                continue
            
            # Choose character based on brightness
            char_idx = int(cell_brightness * (len(char_palette) - 1))
            char = char_palette[min(char_idx, len(char_palette) - 1)]
            
            # Add some randomness to characters for texture
            if random.random() < 0.3:
                char = random.choice("(){}[]<>/\\|!?@#$%&*+-=~")
            
            # Color: below threshold = red (background), above = green (subject)
            if cell_brightness < threshold_split:
                # Red with intensity variation
                intensity = 0.4 + cell_brightness * 1.2
                color = tuple(int(c * intensity) for c in color_dark)
            else:
                # Green with intensity variation
                intensity = 0.5 + (cell_brightness - threshold_split) * 1.0
                color = tuple(int(min(c * intensity, 255)) for c in color_bright)
            
            # Draw character
            pos = (col * cell_size, (row + 1) * cell_size - 2)
            cv2.putText(output, char, pos, font, font_scale, color, 1, cv2.LINE_AA)
    
    # Add subtle scanlines
    for y in range(0, h, 3):
        output[y, :] = (output[y, :] * 0.7).astype(np.uint8)
    
    return output


# =============================================================================
# BINARY BLOOM (0/1 digits on solid color background)
# =============================================================================

def draw_binary_bloom(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Binary Bloom: 0/1 digits inside subject silhouette on solid background.
    
    Pipeline: grayscale → blur → Canny → morph close → largest contour → fill
    Edge emphasis: brighter + denser digits along silhouette edges.
    """
    h, w = frame.shape[:2]
    
    # Parameters
    bg_color = preset.get("bg_color", (160, 40, 0))         # Deep blue BGR
    grid_step = preset.get("grid_step", 14)                  # Sparser grid
    edge_grid_step = preset.get("edge_grid_step", 10)        # Denser at edges
    font_scale = preset.get("binary_font_scale", 0.4)
    
    # Colors
    interior_color = (180, 180, 180)   # Dimmer grey for interior
    edge_color = (255, 255, 255)       # Bright white for edges
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # =========================================================================
    # SUBJECT MASK: blur → Canny → morph close → largest contour
    # =========================================================================
    # 1. Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. Morphological close to connect edges into regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Find contours and keep the largest one
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    subject_mask = np.zeros((h, w), dtype=np.uint8)
    
    if contours:
        # Find largest contour by area
        best_contour = max(contours, key=lambda c: cv2.contourArea(c))
        area = cv2.contourArea(best_contour)
        
        # Only use if reasonable size (3-90% of frame)
        if h * w * 0.03 < area < h * w * 0.9:
            cv2.drawContours(subject_mask, [best_contour], -1, 255, -1)
    
    # Fallback: center ellipse if no good contour found
    mask_coverage = np.sum(subject_mask > 0) / (h * w)
    if mask_coverage < 0.02:
        cv2.ellipse(subject_mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 255, -1)
    
    # =========================================================================
    # EDGE MASK: detect edges of the silhouette for emphasis
    # =========================================================================
    edge_mask = cv2.Canny(subject_mask, 50, 150)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_mask = cv2.dilate(edge_mask, kernel_small, iterations=2)
    
    # =========================================================================
    # DRAW OUTPUT - solid blue background
    # =========================================================================
    output = np.full((h, w, 3), bg_color, dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Stable random seed (updates every ~100ms / 3 frames) for gentle flicker
    random.seed(frame_idx // 3 + 42)
    
    # -------------------------------------------------------------------------
    # PASS 1: Interior digits (dimmer, sparser)
    # -------------------------------------------------------------------------
    for row in range(0, h, grid_step):
        for col in range(0, w, grid_step):
            cy = min(row + grid_step // 2, h - 1)
            cx = min(col + grid_step // 2, w - 1)
            
            # Only inside subject, skip edges (drawn separately)
            if subject_mask[cy, cx] == 0:
                continue
            if edge_mask[cy, cx] > 0:
                continue
            
            digit = "0" if random.random() < 0.5 else "1"
            pos = (col, row + grid_step - 2)
            cv2.putText(output, digit, pos, font, font_scale, interior_color, 1, cv2.LINE_AA)
    
    # -------------------------------------------------------------------------
    # PASS 2: Edge digits (brighter, denser)
    # -------------------------------------------------------------------------
    for row in range(0, h, edge_grid_step):
        for col in range(0, w, edge_grid_step):
            cy = min(row + edge_grid_step // 2, h - 1)
            cx = min(col + edge_grid_step // 2, w - 1)
            
            # Only on edges
            if edge_mask[cy, cx] == 0:
                continue
            
            digit = "0" if random.random() < 0.5 else "1"
            pos = (col, row + edge_grid_step - 2)
            cv2.putText(output, digit, pos, font, font_scale * 1.1, edge_color, 1, cv2.LINE_AA)
    
    return output