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
from .segmentation import get_person_mask


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
        output = _draw_frame_overlay(frame, points, preset, colors, frame_idx, face_data)
    else:
        # NORMAL MODE: Replace background with effect
        output = _draw_frame_replace(frame, points, preset, colors, frame_idx, face_data)
    
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
    if preset.get("detect_mesh", False) and preset.get("draw_mesh", True) and mesh_points:
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
    face_data: dict | None = None,
) -> np.ndarray:
    """Normal mode: darken background and draw effects on top."""
    
    # Check for text-based effects first (they replace the entire pipeline)
    text_result = apply_text_effect(frame, preset, colors, frame_idx=frame_idx, points=points, face_data=face_data)
    if text_result is not None:
        output = text_result

        # Minimal point overlay for text modes - skipped entirely when the
        # effect tracks no points (saves full-frame allocs/blurs every frame).
        if points:
            overlay = np.zeros_like(output)
            _draw_all_elements(overlay, points, preset, colors)

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
    face_data: dict | None = None,
) -> np.ndarray:
    """
    Overlay mode: blend effects at ~40% over the original frame.
    Shows "what the algorithm sees" on top of reality.
    """
    # Keep original frame intact
    original = frame.copy()
    
    # Check for text-based effects
    text_result = apply_text_effect(frame, preset, colors, frame_idx=frame_idx, points=points, face_data=face_data)
    if text_result is not None:
        # For text effects in overlay mode, blend text layer over original
        effect_layer = text_result

        # Minimal point overlay - only when the effect actually tracks points.
        if points:
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

    # Vectorized pairwise distances; only iterate the pairs that connect.
    diff = positions[:, None, :] - positions[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    ii, jj = np.where(np.triu(dists < max_dist, k=1))

    pos_int = positions.astype(int)
    for i, j in zip(ii, jj):
        alpha = 1.0 - (dists[i, j] / max_dist)
        color = (base_color * alpha).astype(int).tolist()
        cv2.line(overlay, tuple(pos_int[i]), tuple(pos_int[j]), color, thickness, cv2.LINE_AA)


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

_data_body_cache: dict = {}


def draw_data_body(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Data Body effect: render the subject as a cloud of alphanumeric glyphs.

    Vectorized glyph-tile renderer: brightness is sampled per cell, each cell
    shows a stable pseudo-random glyph (no strobing) tinted by local
    brightness, forming the silhouette from text.
    """
    h, w = frame.shape[:2]

    glyph_chars = preset.get("glyph_chars", "ABCDEF0123456789")
    cell_size = max(5, int(preset.get("glyph_cell_size", 10)))
    min_brightness = int(preset.get("min_brightness", 40))
    invert_bg = preset.get("invert_background", False)
    n = len(glyph_chars)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grid_h, grid_w = max(1, h // cell_size), max(1, w // cell_size)
    cell_lum = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    if invert_bg:
        bg_value, text_color = 240, np.array((40, 40, 40), dtype=np.float32)
    else:
        bg_value = 0
        text_color = np.array(colors.get("point", (80, 255, 80)), dtype=np.float32)

    # Pre-render glyphs as white tiles (+ one blank tile at index n) - cached.
    key = (cell_size, glyph_chars)
    tiles = _data_body_cache.get(key)
    if tiles is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = cell_size / 16.0
        tiles = np.zeros((n + 1, cell_size, cell_size, 3), dtype=np.uint8)
        for i, ch in enumerate(glyph_chars):
            cv2.putText(tiles[i], ch, (0, cell_size - 2), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        _data_body_cache[key] = tiles

    # Stable per-cell glyph choice (coordinate hash) so glyphs do not strobe.
    gy, gx = np.indices((grid_h, grid_w))
    idx = ((gx * 7 + gy * 13) % n).astype(np.int32)
    idx[cell_lum * 255.0 < min_brightness] = n  # blank tile below threshold

    mapped = tiles[idx]  # (grid_h, grid_w, cell, cell, 3)
    text_img = mapped.transpose(0, 2, 1, 3, 4).reshape(grid_h * cell_size, grid_w * cell_size, 3)

    # Tint white glyphs by per-cell brightness and the preset color.
    inten = cv2.resize(cell_lum, (text_img.shape[1], text_img.shape[0]),
                       interpolation=cv2.INTER_NEAREST)[:, :, None]
    alpha = text_img[:, :, 0:1].astype(np.float32) / 255.0
    tinted = (text_color[None, None, :] * inten * alpha).astype(np.uint8)

    output = np.full((h, w, 3), bg_value, dtype=np.uint8)
    if invert_bg:
        # Dark glyphs on light paper: subtract the glyph alpha from the page.
        region = output[:tinted.shape[0], :tinted.shape[1]].astype(np.float32)
        region = region * (1.0 - alpha * inten) + text_color[None, None, :] * alpha * inten
        output[:tinted.shape[0], :tinted.shape[1]] = region.astype(np.uint8)
    else:
        output[:tinted.shape[0], :tinted.shape[1]] = tinted

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
def draw_ocular_overload(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
    face_data: dict | None = None,
) -> np.ndarray:
    """
    Ocular Overload effect - Retro computer glitch.
    - High-contrast red horizontal scanlines
    - Tracks eye pupils to render blocky squares that cycle Red -> Blue -> Green
    """
    h, w = frame.shape[:2]
    
    # 1. Base Effect: High Contrast Red scanlines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast dramatically (edge/detail emphasis)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    gray_high = clahe.apply(gray)
    
    # Threshold to create an intense black & white binary look
    _, binary = cv2.threshold(gray_high, 80, 255, cv2.THRESH_BINARY)
    
    Y, X = np.ogrid[:h, :w]
    # Wavy Y displacement
    wave = np.sin(X * 0.05 + frame_idx * 0.2) * 2
    Y_wave = Y + wave
    
    # Horizontal organic scanlines (mask every other line)
    scan_mask = np.where((Y_wave.astype(int) % 2) == 0, 255, 0).astype(np.uint8)
    
    # Only keep bright pixels that fall on the scanlines
    binary_scan = cv2.bitwise_and(binary, scan_mask)
    
    # Color mapping: Black background, Red foreground
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[binary_scan == 255] = (0, 0, 200) # Deep visceral red (BGR format)
    
    # Subtle dark red background for non-scanline bright areas to give it depth
    bg_red = (0, 0, 50)
    output[(binary == 255) & (scan_mask == 0)] = bg_red
    
    # 2. Eye Tracking & Color Cycling
    if face_data and "mesh_points" in face_data:
        # Cycle colors: Red, Blue, Green every 0.25 seconds
        # 30 fps * 0.25s = 7.5 frames. We'll use 7.5 frames for a fast snappy glitch feel.
        color_cycle_idx = int(frame_idx / 7.5) % 3
        # BGR: Red=(0,0,200) to match background perfectly, Blue=(255,0,0), Green=(0,255,0)
        cycle_colors = [(0, 0, 200), (255, 0, 0), (0, 255, 0)]
        iris_color = cycle_colors[color_cycle_idx]
        
        # Create an off-screen layer initialized as a COPY of the output
        # This ensures the sclera (white of eye) maintains the red wavy scanlines
        eyes_layer = output.copy()
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        
        for face_pts in face_data["mesh_points"]:
            if len(face_pts) >= 468:
                # Full eye contours for masking
                # Screen Left Eye (Right eye of the person)
                screen_left_contour = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
                # Screen Right Eye (Left eye of the person)
                screen_right_contour = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
                
                # Create mask polygons
                left_poly = np.array([(int(face_pts[i][0]), int(face_pts[i][1])) for i in screen_left_contour], dtype=np.int32)
                right_poly = np.array([(int(face_pts[i][0]), int(face_pts[i][1])) for i in screen_right_contour], dtype=np.int32)
                
                cv2.fillPoly(eye_mask, [left_poly], 255)
                cv2.fillPoly(eye_mask, [right_poly], 255)
                
                # Approximate Eye Centers (using corner points for stability)
                lx = int((face_pts[33][0] + face_pts[133][0]) / 2)
                ly = int((face_pts[33][1] + face_pts[133][1]) / 2)
                
                rx = int((face_pts[362][0] + face_pts[263][0]) / 2)
                ry = int((face_pts[362][1] + face_pts[263][1]) / 2)
                
                # Sizes based on eye width
                eye_w_left = abs(face_pts[133][0] - face_pts[33][0])
                eye_w_right = abs(face_pts[263][0] - face_pts[362][0])
                
                # Hardcoded retro 8-bit eye sprite: 9 rows, 8 columns
                # 0: transparent, 1: iris color, 2: pupil (black)
                eye_sprite = [
                    "  1111  ",
                    " 111111 ",
                    "11111111",
                    "11122111",
                    "11122111",
                    "11122111",
                    "11111111",
                    " 111111 ",
                    "  1111  "
                ]
                sprite_h = len(eye_sprite)
                sprite_w = len(eye_sprite[0])
                
                # Draw blocky scanline iris and rectangular pupil for both eyes
                for cx, cy, eye_w in [(lx, ly, eye_w_left), (rx, ry, eye_w_right)]:
                    # The sprite width is 8 blocks. The iris diameter in the ref is quite small,
                    # roughly 35% of the total eye width, not 70%.
                    # Block size = (0.35 * eye_w) / 8.
                    block_size = max(2, int((eye_w * 0.35) / sprite_w))
                    
                    # Align the top-left of the sprite grid so it's centered
                    start_x = cx - (sprite_w * block_size) // 2
                    start_y = cy - (sprite_h * block_size) // 2
                    
                    for row_idx, row_str in enumerate(eye_sprite):
                        for col_idx, char in enumerate(row_str):
                            if char == ' ':
                                continue
                                
                            bx = start_x + col_idx * block_size
                            by = start_y + row_idx * block_size
                            
                            x1, x2 = max(0, bx), min(w, bx + block_size)
                            y1, y2 = max(0, by), min(h, by + block_size)
                            
                            if x2 > x1 and y2 > y1:
                                block = eyes_layer[y1:y2, x1:x2]
                                s_mask = scan_mask[y1:y2, x1:x2]
                                
                                if char == '1': # Iris - drawn only on bright scanlines
                                    block[s_mask == 255] = iris_color
                                    # Override dim areas with pure black to hollow out scanlines
                                    block[s_mask == 0] = (0, 0, 0)
                                elif char == '2': # Pupil - solid black rectangle
                                    block[:] = (0, 0, 0)
                
        # Blend the eyes layer onto the final output strictly within the mask
        # Expand mask to 3 channels for `where` operation
        mask_3d = eye_mask[:, :, np.newaxis] == 255
        output = np.where(mask_3d, eyes_layer, output)

        # Glow around the eyes so the glitch reads as "overloaded".
        if eye_mask.any():
            eye_glow = cv2.GaussianBlur(
                cv2.bitwise_and(output, output, mask=eye_mask), (0, 0), 7
            )
            output = cv2.add(output, eye_glow)

    # --- Post: CRT phosphor bloom so the red structure emits light ---
    glow = cv2.GaussianBlur(output, (0, 0), 4)
    output = cv2.addWeighted(output, 1.0, glow, 0.45, 0)

    # Moving bright "overload" scan band sweeping down the frame.
    band_y = int((frame_idx * 9) % h)
    bh = max(8, h // 60)
    y1b, y2b = band_y, min(h, band_y + bh)
    output[y1b:y2b] = np.clip(output[y1b:y2b].astype(np.int16) + 40, 0, 255).astype(np.uint8)

    # Subtle vignette for depth.
    yy, xx = np.ogrid[:h, :w]
    cyv, cxv = h / 2.0, w / 2.0
    d = np.sqrt(((xx - cxv) / cxv) ** 2 + ((yy - cyv) / cyv) ** 2)
    vig = np.clip(1.0 - (d - 0.6) * 0.5, 0.55, 1.0).astype(np.float32)
    output = (output.astype(np.float32) * vig[:, :, None]).astype(np.uint8)

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
    face_data: dict | None = None,
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
    elif text_mode == "dither_trace":
        return draw_dither_trace(frame, preset, colors)
    elif text_mode == "ocular_overload":
        return draw_ocular_overload(frame, preset, colors, frame_idx, face_data)
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
    elif text_mode == "signal_feedback":
        return draw_signal_feedback(frame, preset, colors, frame_idx=frame_idx)
    # === NEW EFFECTS v3 ===
    elif text_mode == "signal_bloom":
        return draw_signal_bloom(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "glyph_trace":
        return draw_glyph_trace(frame, preset, colors, frame_idx=frame_idx, points=points)
    # === VIRAL TOUCHDESIGNER EFFECTS v4 ===
    elif text_mode == "slit_scan":
        return draw_slit_scan(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "ascii_core":
        return draw_ascii_core(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "kaleidoscope":
        return draw_kaleidoscope(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "tv_static":
        return draw_tv_static(frame, preset, colors, frame_idx=frame_idx)
    # === VIRAL TOUCHDESIGNER EFFECTS v6 ===
    elif text_mode == "chromatic_ghost":
        return draw_chromatic_ghost(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "crystallize":
        return draw_crystallize(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "halftone":
        return draw_halftone(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "light_trails":
        return draw_light_trails(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "ink":
        return draw_ink(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "neon_glow":
        return draw_neon_glow(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "point_cloud":
        return draw_point_cloud(frame, preset, colors, frame_idx=frame_idx)

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
    
    # Distance-faded connection lines (mesh between nearby blobs).
    max_connection_dist = preset.get("max_connection_dist", 180)
    for i in range(len(blob_centers)):
        for j in range(i + 1, len(blob_centers)):
            p1, p2 = blob_centers[i], blob_centers[j]
            dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if dist < max_connection_dist:
                a = 1.0 - dist / max_connection_dist
                c = int(40 + 180 * a)
                cv2.line(output, p1, p2, (c, c, c), 1, cv2.LINE_AA)

    accent = (255, 230, 120)  # soft cyan accent (BGR)
    # Draw each blob as a corner-bracket box with an ID + center tick.
    for (x, y, bw, bh, idx, area) in blob_boxes:
        cl = max(6, min(bw, bh) // 5)  # corner bracket length
        x2, y2 = x + bw, y + bh
        for (px, py, dx, dy) in [
            (x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(output, (px, py), (px + dx * cl, py), box_color, 1, cv2.LINE_AA)
            cv2.line(output, (px, py), (px, py + dy * cl), box_color, 1, cv2.LINE_AA)

        # Center tick.
        ccx, ccy = blob_centers[idx]
        cv2.drawMarker(output, (ccx, ccy), accent, cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)

        # ID + size readout above the box (shadowed for legibility).
        box_size = min(bw, bh)
        fscale = max(0.3, min(0.42, box_size / 220.0))
        label = f"ID {idx:02d}"
        ly = y - 5 if y > 14 else y + int(box_size * 0.2) + 6
        cv2.putText(output, label, (x + 1, ly + 1), font, fscale, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(output, label, (x, ly), font, fscale, box_color, 1, cv2.LINE_AA)

    return output


# =============================================================================
# PARTICLE SILHOUETTE EFFECT (bb.dere style)
# =============================================================================

def draw_dither_trace(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Dither Trace effect: High contrast 1-bit dithered ink effect (Atkinson style approximation).
    Uses a scaled 8x8 Bayer matrix and contrast enhancement to get distinct, chunky ink dots.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 8x8 Bayer matrix for a wider range of tones and a more organic crosshatch
    bayer = np.array([
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ], dtype=np.float32) / 64.0 * 255.0
    
    # Do NOT scale the bayer matrix. Apply it 1-to-1 to pixels for classic high-frequency 1-bit look.
    bh, bw = bayer.shape
    tiled_bayer = np.tile(bayer, (h // bh + 1, w // bw + 1))[:h, :w]
    
    # Very slight blur to reduce webcam noise, but not too much to preserve high-frequency details
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # CRITICAL: Increase contrast before dithering to preserve distinct shapes
    # S-curve function to push midtones
    smoothed_f = smoothed.astype(np.float32) / 255.0
    smoothed_f = 1.0 / (1.0 + np.exp(-10.0 * (smoothed_f - 0.5))) # Sigmoid
    smoothed = (smoothed_f * 255.0).astype(np.float32)
    
    # Apply dither threshold
    binary = smoothed > tiled_bayer
    
    # Brand Colors (BGR format)
    # Ink: #1F1E1D -> R:31 G:30 B:29
    # Paper: #FAF9F5 -> R:250 G:249 B:245
    ink = np.array([29, 30, 31], dtype=np.uint8)
    paper = np.array([245, 249, 250], dtype=np.uint8)
    
    output = np.zeros_like(frame)
    output[binary] = paper
    output[~binary] = ink
    
    return output


# =============================================================================
# NUMERIC AURA EFFECT (Subject isolation - numbers on subject, video background)
# =============================================================================

def draw_number_cloud(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Binary Bloom (Numeric Aura) effect - High-end data visualization style.
    Features:
    - Strict scrolling grid system for a structured aesthetic
    - Dim hex background tracking the subject mask
    - Bright glowing binary (0/1) foreground directly on the subject
    - Smooth mask falloff for organic integration
    - Pure sci-fi palette (cyan, deep blue, white hot)
    """
    import string
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # === SUBJECT DETECTION ===
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 30, 100)
    
    kernel = np.ones((30, 30), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    subject_mask = np.zeros((h, w), dtype=np.float32)
    
    if contours:
        best_contour = None
        best_score = 0
        center_x, center_y = w // 2, h // 2
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
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [hull], 255)
            subject_mask = temp_mask.astype(np.float32) / 255.0
    
    if np.sum(subject_mask) < h * w * 0.03:
        # Fallback mask if no subject found
        center_x, center_y = w // 2, h // 2
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(temp_mask, (center_x, center_y), (w//3, h//2 + 50), 0, 0, 360, 255, -1)
        subject_mask = temp_mask.astype(np.float32) / 255.0
        
    # Smooth the mask to create a glowing falloff instead of a hard edge
    subject_mask = cv2.GaussianBlur(subject_mask, (81, 81), 0)
    
    # === COLORS ===
    # Colors configured based on user's preference for clean, high-contrast look
    # BGR format
    bg_color = (0, 0, 0)          # Void black
    blue_dim = (140, 50, 0)       # Deep dim background blue
    cyan_bright = (255, 200, 50)  # Bright cyan for binary foreground
    white_hot = (250, 250, 250)   # White hot core
    
    output = np.full((h, w, 3), bg_color, dtype=np.uint8)
    
    # Use FONT_HERSHEY_PLAIN for the crisp "code" look
    font = cv2.FONT_HERSHEY_PLAIN
    
    # === LAYER 1: Background Hex Grid (Dim, moving upwards) ===
    # Strict grid spacing for the background
    grid_w, grid_h = 16, 20
    hex_chars = string.hexdigits.upper()
    
    # Scrolling background by offset
    y_offset = int(frame_idx * 1.5) % grid_h
    
    for y in range(grid_h - y_offset, h + grid_h, grid_h):
        for x in range(4, w, grid_w):
            mask_val = subject_mask[min(y, h-1), min(x, w-1)]
            
            # Base brightness depends slightly on the mask to wrap around the subject
            alpha = 0.15 + (mask_val * 0.5)
            color = tuple(int(c * alpha) for c in blue_dim)
            
            # Change characters based on time and position
            idx = (x + y // 5 + int(frame_idx * 0.3)) % 16
            char = hex_chars[idx]
            
            cv2.putText(output, char, (x, y), font, 1.0, color, 1, cv2.LINE_AA)
            
    # === LAYER 2: Foreground Binary Stream (Bright Cyan, fast changing) ===
    # Larger grid for the foreground binary
    bin_grid_w, bin_grid_h = 24, 30
    
    # Separate layer for glowing text
    glow_layer = np.zeros_like(output)
    
    for y in range(16, h, bin_grid_h):
        for x in range(12, w, bin_grid_w):
            mask_val = subject_mask[y, x]
            
            # Only draw where the subject is prominent
            if mask_val > 0.2:
                # Fast flickering 0 and 1
                is_one = ((x * y + frame_idx * 5) % 11) > 4
                char = "1" if is_one else "0"
                
                # Center of the subject becomes white hot
                if mask_val > 0.75:
                    text_col = white_hot
                    glow_col = cyan_bright
                    thickness = 2
                    scale = 1.3
                else:
                    # Blend the color intensity based on mask depth
                    blend = (mask_val - 0.2) / 0.55
                    text_col = tuple(
                        int(cyan_bright[i] * blend + blue_dim[i] * (1 - blend))
                        for i in range(3)
                    )
                    glow_col = blue_dim
                    thickness = 1
                    scale = 1.1 + (blend * 0.2)
                
                # Draw sharp primary text
                cv2.putText(output, char, (x, y), font, scale, text_col, thickness, cv2.LINE_4)
                
                # Draw thick glow text onto the glow layer
                if mask_val > 0.5:
                    cv2.putText(glow_layer, char, (x, y), font, scale, glow_col, thickness + 2, cv2.LINE_AA)
                    
    # Composite the glow
    glow_layer = cv2.GaussianBlur(glow_layer, (15, 15), 0)
    output = cv2.addWeighted(output, 1.0, glow_layer, 0.9, 0)
    
    return output


# =============================================================================
# MOTION TRACE EFFECT (Dense optical flow with persistent trails)
# =============================================================================

# Persistent state for motion trace effect
_motion_trace_prev_frame: np.ndarray | None = None
_motion_trace_trail_canvas: np.ndarray | None = None

def draw_motion_trace(
    frame: np.ndarray,
    points: list[TrackedPoint],
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Motion Flow: Flowing curved trails with network connections.
    
    Features:
    - Dense optical flow (Farneback) for motion detection
    - Persistent trail canvas that fades over time
    - Network lines connecting nearby motion points
    - Composited over original video
    """
    global _motion_trace_prev_frame, _motion_trace_trail_canvas
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Parameters
    flow_color = preset.get("flow_color", colors.get("trail", (255, 200, 100)))
    thickness = preset.get("line_thickness", 2)
    min_flow_mag = preset.get("min_flow_mag", 1.5)
    line_length_scale = preset.get("line_length_scale", 6)
    sample_step = preset.get("sample_step", 10)
    trail_fade = preset.get("trail_fade", 0.92)  # How much trails persist (0.9-0.98)
    max_connect_dist = preset.get("max_connect_dist", 50)  # Max distance for network lines
    frame_alpha = preset.get("frame_alpha", 0.6)  # Original frame visibility
    trail_alpha = preset.get("trail_alpha", 0.9)  # Trail canvas visibility
    
    # Initialize or reset trail canvas if needed
    if _motion_trace_trail_canvas is None or _motion_trace_trail_canvas.shape[:2] != (h, w):
        _motion_trace_trail_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Fade previous trails (creates the comet/persistence effect)
    _motion_trace_trail_canvas = ((_motion_trace_trail_canvas.astype(np.float32)) * trail_fade).astype(np.uint8)
    
    # Need previous frame for optical flow
    if _motion_trace_prev_frame is None or _motion_trace_prev_frame.shape != gray.shape:
        _motion_trace_prev_frame = gray.copy()
        # Return original frame with empty trail on first frame
        return cv2.addWeighted(frame, frame_alpha, _motion_trace_trail_canvas, trail_alpha, 0)
    
    # Compute dense optical flow (Farneback) at half resolution - visually
    # identical for this effect but roughly 4x cheaper at 1080p.
    sf = 0.5
    prev_small = cv2.resize(_motion_trace_prev_frame, (0, 0), fx=sf, fy=sf,
                            interpolation=cv2.INTER_AREA)
    gray_small = cv2.resize(gray, (0, 0), fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
    flow_small = cv2.calcOpticalFlowFarneback(
        prev_small, gray_small,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR) / sf

    # Update previous frame
    _motion_trace_prev_frame = gray.copy()
    
    # Compute flow magnitude
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    
    # Color palette for variety (BGR - cyan/blue tones)
    flow_colors = [
        flow_color,                    # Primary from preset
        (255, 180, 80),               # Cyan
        (200, 255, 150),              # Light cyan-green
        (255, 220, 180),              # Pale cyan
    ]
    
    # =========================================================================
    # COLLECT MOTION POINTS
    # =========================================================================
    motion_points = []
    color_idx = 0
    
    for y in range(sample_step, h - sample_step, sample_step):
        for x in range(sample_step, w - sample_step, sample_step):
            mag = magnitude[y, x]
            
            if mag < min_flow_mag:
                continue
            
            # Store motion point with its flow vector and color
            dx = flow_x[y, x]
            dy = flow_y[y, x]
            color = flow_colors[color_idx % len(flow_colors)]
            motion_points.append((x, y, dx, dy, mag, color))
            color_idx += 1
    
    # =========================================================================
    # DRAW FLOW LINES onto trail canvas
    # =========================================================================
    for (x, y, dx, dy, mag, color) in motion_points:
        # Calculate end point
        x2 = int(x + dx * line_length_scale)
        y2 = int(y + dy * line_length_scale)
        
        # Clamp to frame bounds
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        
        # Create curved line (3 points with perpendicular offset)
        mid_x = (x + x2) // 2 + int(dy * 0.4)
        mid_y = (y + y2) // 2 - int(dx * 0.4)
        mid_x = max(0, min(w - 1, mid_x))
        mid_y = max(0, min(h - 1, mid_y))
        
        pts = np.array([[x, y], [mid_x, mid_y], [x2, y2]], dtype=np.int32)
        
        # Brightness based on magnitude
        alpha = min(1.0, mag / 6.0)
        draw_color = tuple(int(c * alpha) for c in color)
        
        # Draw curved flow line
        cv2.polylines(_motion_trace_trail_canvas, [pts], False, draw_color, thickness, cv2.LINE_AA)
        
        # Glowing head at end point for strong motion
        if mag > min_flow_mag * 1.5:
            cv2.circle(_motion_trace_trail_canvas, (x2, y2), 3, (255, 255, 255), -1, cv2.LINE_AA)
    
    # =========================================================================
    # DRAW NETWORK CONNECTIONS between nearby motion points
    # =========================================================================
    # Cap the O(n^2) pass: a subsample keeps the mesh look while bounding cost
    # on busy frames (uncapped this could be tens of thousands of pairs).
    conn_points = motion_points
    max_conn_points = 120
    if len(conn_points) > max_conn_points:
        stride = len(conn_points) // max_conn_points
        conn_points = conn_points[::stride]

    if len(conn_points) > 1:
        for i, (x1, y1, _, _, mag1, color1) in enumerate(conn_points):
            for (x2, y2, _, _, mag2, _) in conn_points[i+1:]:
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                if dist < max_connect_dist:
                    # Fainter connection lines
                    conn_alpha = 0.4 * (1 - dist / max_connect_dist)
                    conn_color = tuple(int(c * conn_alpha) for c in color1)
                    cv2.line(_motion_trace_trail_canvas, (x1, y1), (x2, y2), conn_color, 1, cv2.LINE_AA)
    
    # =========================================================================
    # COMPOSITE: trail canvas over original frame
    # =========================================================================
    # Add subtle glow to trail canvas
    glow = cv2.GaussianBlur(_motion_trace_trail_canvas, (7, 7), 0)
    trail_with_glow = cv2.addWeighted(_motion_trace_trail_canvas, 1.0, glow, 0.4, 0)
    
    # Blend with original frame
    output = cv2.addWeighted(frame, frame_alpha, trail_with_glow, trail_alpha, 0)
    
    return output


# =============================================================================
# CONTOUR TRACE EFFECT (Edge-based visualization)
# =============================================================================

_contour_prev_edges: np.ndarray | None = None


def draw_contour_trace(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Contour Trace: Pure minimalist edge visualization.

    Clean white edges on black with subtle glow. Edges are temporally blended
    with the previous frame so lines breathe instead of strobing.
    """
    global _contour_prev_edges

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

    # Temporal smoothing: carry a fading echo of the previous edges so the
    # lines feel continuous frame-to-frame instead of flickering on/off.
    if _contour_prev_edges is not None and _contour_prev_edges.shape == edges.shape:
        edges = cv2.max(edges, (_contour_prev_edges * 0.55).astype(np.uint8))
    _contour_prev_edges = edges.copy()

    # Create output - pure black background, edge intensity preserved.
    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

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

_codenet_pts: np.ndarray | None = None
_codenet_prev_gray: np.ndarray | None = None


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
    
    global _codenet_pts, _codenet_prev_gray

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better feature detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Stabilized nodes: track existing corners with optical flow each frame and
    # only re-detect periodically (or when too many are lost). The mesh then
    # follows the image smoothly instead of jittering with fresh detections.
    redetect = (
        _codenet_pts is None
        or _codenet_prev_gray is None
        or _codenet_prev_gray.shape != gray.shape
        or len(_codenet_pts) < max(8, max_points // 4)
        or frame_idx % 12 == 0
    )

    if not redetect:
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            _codenet_prev_gray, gray, _codenet_pts.reshape(-1, 1, 2), None,
            winSize=(21, 21), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
        )
        if tracked is not None:
            ok = status.flatten() == 1
            _codenet_pts = tracked.reshape(-1, 2)[ok]
        else:
            redetect = True

    if redetect:
        corners = cv2.goodFeaturesToTrack(
            enhanced,
            maxCorners=max_points,
            qualityLevel=0.02,
            minDistance=20,
            blockSize=7,
        )
        if corners is not None:
            _codenet_pts = corners.reshape(-1, 2).astype(np.float32)

    _codenet_prev_gray = gray.copy()

    if _codenet_pts is None or len(_codenet_pts) < 3:
        return frame.copy()

    points = _codenet_pts.astype(np.float32)
    
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

_code_shadow_cache: dict = {}


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
    cell_size = max(5, int(preset.get("cell_size", 9)))
    char_palette = preset.get("char_palette", " .:-=+*o#%@")
    color_dark = np.array(preset.get("color_dark", (0, 0, 170)), dtype=np.float32)    # red BGR
    color_bright = np.array(preset.get("color_bright", (0, 220, 30)), dtype=np.float32)  # green BGR
    threshold_split = float(preset.get("threshold_split", 0.45))
    n = len(char_palette)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    grid_h, grid_w = max(1, h // cell_size), max(1, w // cell_size)
    cell_lum = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # Pre-render the palette as white glyph tiles (cached); index by brightness.
    key = (cell_size, char_palette)
    tiles = _code_shadow_cache.get(key)
    if tiles is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = cell_size / 16.0
        tiles = np.zeros((n, cell_size, cell_size, 3), dtype=np.uint8)
        for i, ch in enumerate(char_palette):
            if ch != " ":
                cv2.putText(tiles[i], ch, (0, cell_size - 2), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        _code_shadow_cache[key] = tiles

    idx = np.clip((cell_lum * (n - 1)).astype(np.int32), 0, n - 1)
    idx[cell_lum < 0.08] = 0  # very dark -> blank glyph
    mapped = tiles[idx]  # (grid_h, grid_w, cell, cell, 3)
    text_img = mapped.transpose(0, 2, 1, 3, 4).reshape(grid_h * cell_size, grid_w * cell_size, 3)

    # Per-cell color: red below the split (background), green above (subject),
    # scaled by brightness for depth.
    b = cell_lum[:, :, None]
    col = np.where(b < threshold_split, color_dark, color_bright)
    inten = np.clip(0.45 + b * 1.1, 0.0, 1.0)
    color_grid = (col * inten).astype(np.uint8)
    color_full = cv2.resize(color_grid, (text_img.shape[1], text_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    # Use the white glyphs as an alpha mask over the per-cell color.
    alpha = text_img[:, :, 0:1].astype(np.float32) / 255.0
    tinted = (color_full.astype(np.float32) * alpha).astype(np.uint8)

    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[:tinted.shape[0], :tinted.shape[1]] = tinted

    # Subtle CRT scanlines.
    output[::3] = (output[::3].astype(np.float32) * 0.7).astype(np.uint8)
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
    # SUBJECT MASK: prefer real person segmentation, fall back to contours.
    # =========================================================================
    seg = get_person_mask(frame)
    if seg is not None and np.count_nonzero(seg) > h * w * 0.02:
        _, subject_mask = cv2.threshold(seg, 110, 255, cv2.THRESH_BINARY)
    else:
        # Fallback: blur -> Canny -> morph close -> largest reasonable contour.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        subject_mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            best_contour = max(contours, key=lambda c: cv2.contourArea(c))
            area = cv2.contourArea(best_contour)
            if h * w * 0.03 < area < h * w * 0.9:
                cv2.drawContours(subject_mask, [best_contour], -1, 255, -1)
        if np.sum(subject_mask > 0) / (h * w) < 0.02:
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


# =============================================================================
# SIGNAL FEEDBACK (CRT/VHS style with noise warping and feedback trails)
# =============================================================================

# Persistent state for signal feedback effect
_signal_feedback_buffer: np.ndarray | None = None
_signal_feedback_noise: np.ndarray | None = None

def draw_signal_feedback(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Signal Feedback: CRT/VHS-style effect with noise warping and feedback trails.
    
    Features:
    - Noise-based coordinate distortion (cv2.remap)
    - Persistent feedback buffer with decay (liquid trails)
    - Chromatic aberration (RGB channel shift)
    - CRT scanlines
    """
    global _signal_feedback_buffer, _signal_feedback_noise
    
    h, w = frame.shape[:2]
    
    # Parameters
    feedback_decay = preset.get("feedback_decay", 0.88)
    distortion_amp = preset.get("distortion_amp", 8.0)
    chroma_shift = preset.get("chroma_shift", 3)
    scanline_intensity = preset.get("scanline_intensity", 0.15)
    noise_scale = preset.get("noise_scale", 0.02)  # How fast noise evolves
    
    # Convert current frame to float [0, 1]
    current_float = frame.astype(np.float32) / 255.0
    
    # =========================================================================
    # INITIALIZE STATE
    # =========================================================================
    if _signal_feedback_buffer is None or _signal_feedback_buffer.shape[:2] != (h, w):
        _signal_feedback_buffer = current_float.copy()
        _signal_feedback_noise = np.random.rand(h, w).astype(np.float32)
        return frame  # Return original on first frame
    
    # =========================================================================
    # STEP 1: Generate noise-based warp map
    # =========================================================================
    # Slowly evolve the noise field
    new_noise = np.random.rand(h, w).astype(np.float32)
    _signal_feedback_noise = _signal_feedback_noise * (1 - noise_scale) + new_noise * noise_scale
    
    # Smooth noise for organic warping
    smooth_noise = cv2.GaussianBlur(_signal_feedback_noise, (51, 51), 0)
    
    # Create base coordinate grid
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)
    
    # Add noise-based displacement
    offset_x = (smooth_noise - 0.5) * distortion_amp
    offset_y = (smooth_noise - 0.5) * distortion_amp
    
    map_x = grid_x + offset_x
    map_y = grid_y + offset_y
    
    # =========================================================================
    # STEP 2: Warp feedback buffer and blend with current frame
    # =========================================================================
    # Warp the previous feedback buffer
    warped_feedback = cv2.remap(
        _signal_feedback_buffer, 
        map_x, map_y, 
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Blend: decay old feedback, add new frame
    _signal_feedback_buffer = warped_feedback * feedback_decay + current_float * (1.0 - feedback_decay)
    
    # Convert back to uint8 for post-processing
    result = (_signal_feedback_buffer * 255).astype(np.uint8)
    
    # =========================================================================
    # STEP 3: Chromatic Aberration (RGB channel shift)
    # =========================================================================
    if chroma_shift > 0:
        b, g, r = cv2.split(result)
        
        # Shift R left, B right
        r_shifted = np.roll(r, -chroma_shift, axis=1)
        b_shifted = np.roll(b, chroma_shift, axis=1)
        
        # Clean up edges
        r_shifted[:, -chroma_shift:] = r[:, -chroma_shift:]
        b_shifted[:, :chroma_shift] = b[:, :chroma_shift]
        
        result = cv2.merge([b_shifted, g, r_shifted])
    
    # =========================================================================
    # STEP 4: CRT Scanlines
    # =========================================================================
    if scanline_intensity > 0:
        # Create scanline mask (darken every other row)
        scanline_mask = np.ones((h, w), dtype=np.float32)
        scanline_mask[1::2, :] = 1.0 - scanline_intensity
        
        # Apply to all channels
        result = (result.astype(np.float32) * scanline_mask[..., np.newaxis]).astype(np.uint8)
    
    # =========================================================================
    # STEP 5: Subtle vignette for CRT feel
    # =========================================================================
    # Create radial gradient
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1.0 - (dist_from_center / max_dist) * 0.3
    vignette = np.clip(vignette, 0.7, 1.0).astype(np.float32)
    
    result = (result.astype(np.float32) * vignette[..., np.newaxis]).astype(np.uint8)
    
    return result

# =============================================================================
# SIGNAL BLOOM (Lava-red distortion)
# =============================================================================

_signal_bloom_lut_cache: np.ndarray | None = None


def _signal_bloom_lut() -> np.ndarray:
    """Build (once) the black -> red -> orange -> yellow -> white lava LUT."""
    global _signal_bloom_lut_cache
    if _signal_bloom_lut_cache is not None:
        return _signal_bloom_lut_cache
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        if i < 40:
            t = i / 40
            lut[i, 0] = (0, 0, int(100 * t))
        elif i < 120:
            t = (i - 40) / 80
            lut[i, 0] = (0, 0, 100 + int(155 * t))
        elif i < 180:
            t = (i - 120) / 60
            lut[i, 0] = (0, int(128 * t), 255)
        elif i < 230:
            t = (i - 180) / 50
            lut[i, 0] = (0, 128 + int(127 * t), 255)
        else:
            t = (i - 230) / 25
            lut[i, 0] = (int(255 * t), 255, 255)
    _signal_bloom_lut_cache = lut
    return lut


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
    
    # 3. Apply the cached lava-thermal color map.
    lut = _signal_bloom_lut()
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

    # 5. Actual bloom: the hot (yellow/white) regions glow, for a molten,
    #    light-emitting feel that lives up to the name.
    hot = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    _, hot = cv2.threshold(hot, 160, 255, cv2.THRESH_TOZERO)
    hot_bgr = cv2.bitwise_and(output, output, mask=(hot > 0).astype(np.uint8))
    bloom = cv2.GaussianBlur(hot_bgr, (0, 0), 9)
    output = cv2.add(output, (bloom * 0.7).astype(np.uint8))

    return output


# =============================================================================
# GLYPH TRACE (ASCII INK)
# =============================================================================

def draw_glyph_trace(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
    points: list[Any] | None = None,
) -> np.ndarray:
    """
    Glyph Trace: Renders the frame using an ASCII character grid.
    Uses perfectly crisp monospaced pre-rendered text tiles mapped via NumPy
    for real-time performance and pixel-perfect clarity.
    """
    h, w = frame.shape[:2]
    
    # 1. Colors
    # Ink (#1F1E1D) -> BGR (29, 30, 31)
    # Paper (#FAF9F5) -> BGR (245, 249, 250)
    ink_color = (29, 30, 31)
    paper_color = (245, 249, 250)
    
    # 2. Pre-render crisp ASCII tiles
    # Using a 6x10 block gives a nice tall terminal look
    tw, th = 6, 10
    ascii_chars = " .:-=+*#%@"
    num_chars = len(ascii_chars)
    
    # Create the tile bank (num_chars, height, width, 3)
    tiles = np.full((num_chars, th, tw, 3), paper_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_PLAIN
    
    for i, char in enumerate(ascii_chars):
        if char == ' ':
            continue
        # FONT_HERSHEY_PLAIN at scale 0.8 is approx 8 pixels tall. 
        # (0, 8) is the bottom-left baseline for the text
        # cv2.LINE_4 avoids anti-aliasing blur for maximum crispness
        cv2.putText(tiles[i], char, (0, 8), font, 0.8, ink_color, 1, cv2.LINE_4)
        
    # 3. Downsample Image mathematically to match grid
    grid_w = w // tw
    grid_h = h // th
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    
    # 4. Enhance contrast so we hit the extreme characters more frequently
    norm = small_gray.astype(np.float32) / 255.0
    norm = np.clip((norm - 0.5) * 1.5 + 0.5, 0.0, 1.0)
    
    # 5. Map to character indices
    # We want dark areas (= low norm) to have dense characters (= high indices)
    # Bright areas (= high norm) fall into empty/light characters (= low indices)
    indices = ((1.0 - norm) * (num_chars - 1)).astype(np.int32)
    indices = np.clip(indices, 0, num_chars - 1)
    
    # 6. Build final image instantly through numpy memory mapping
    # mapped shape: (grid_h, grid_w, th, tw, 3)
    mapped = tiles[indices]
    
    # Transpose dimensions to interleave rows and columns properly:
    # From (grid_h, grid_w, cell_h, cell_w, 3) -> (grid_h, cell_h, grid_w, cell_w, 3)
    output = mapped.transpose(0, 2, 1, 3, 4).reshape(grid_h * th, grid_w * tw, 3)
    
    # If the frame size is not perfectly divisible, pad with paper color
    if output.shape[0] != h or output.shape[1] != w:
        final_out = np.full((h, w, 3), paper_color, dtype=np.uint8)
        final_out[:output.shape[0], :output.shape[1]] = output
        return final_out
        
    return output


# =============================================================================
# SLIT SCAN / TIME DISPLACEMENT (iconic TouchDesigner time-warp)
# =============================================================================

_slit_scan_buffer: np.ndarray | None = None
_slit_scan_pos: int = 0


def draw_slit_scan(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Slit Scan: each row of the output is sampled from a different moment in
    time, so vertical motion smears into a flowing time-waterfall.

    Keeps a small ring buffer of recent frames and gathers one row from each
    via fancy indexing (single vectorized op per frame).
    """
    global _slit_scan_buffer, _slit_scan_pos

    h, w = frame.shape[:2]
    n = max(2, int(preset.get("scan_frames", 24)))

    # (Re)initialize the ring buffer if shape or depth changed.
    if (
        _slit_scan_buffer is None
        or _slit_scan_buffer.shape[0] != n
        or _slit_scan_buffer.shape[1:3] != (h, w)
    ):
        _slit_scan_buffer = np.repeat(frame[None], n, axis=0).copy()
        _slit_scan_pos = 0

    # Write current frame as the newest slot.
    _slit_scan_buffer[_slit_scan_pos] = frame

    # Map each output row to an "age": row 0 = newest, last row = oldest.
    age = np.linspace(0, n - 1, h).astype(np.int64)
    buf_idx = (_slit_scan_pos - age) % n
    rows = np.arange(h)

    # out[k] = buffer[buf_idx[k]][row k]  -> (h, w, 3)
    out = _slit_scan_buffer[buf_idx, rows]

    # Advance ring head.
    _slit_scan_pos = (_slit_scan_pos + 1) % n

    return np.ascontiguousarray(out)


def reset_stateful_effects():
    """
    Reset all module-level persistent buffers used by temporal effects.

    Called at the start of each process_video() run so state never leaks
    between separate jobs (which would otherwise share these globals).
    """
    global _motion_trace_prev_frame, _motion_trace_trail_canvas
    global _signal_feedback_buffer, _signal_feedback_noise
    global _slit_scan_buffer, _slit_scan_pos
    global _ghost_buffer, _ghost_pos
    global _light_canvas
    global _contour_prev_edges
    global _codenet_pts, _codenet_prev_gray

    _motion_trace_prev_frame = None
    _motion_trace_trail_canvas = None
    _signal_feedback_buffer = None
    _signal_feedback_noise = None
    _slit_scan_buffer = None
    _slit_scan_pos = 0
    _ghost_buffer = None
    _ghost_pos = 0
    _light_canvas = None
    _contour_prev_edges = None
    _codenet_pts = None
    _codenet_prev_gray = None


# =============================================================================
# ASCII CORE (high-detail white ASCII on black)
# =============================================================================

def draw_ascii_core(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    ASCII Core: dense, detailed white ASCII characters on pure black.

    Brightness is mapped to a fine character ramp and per-cell intensity, with
    an edge-detection boost so contours render with denser glyphs. Built with
    the NumPy glyph-tile technique (pre-render each char once, index into the
    tile stack) so it stays crisp and fast even with small cells.
    """
    h, w = frame.shape[:2]

    cell = max(4, int(preset.get("ascii_cell", 7)))
    ramp = preset.get("ascii_ramp", " .`:-=+ic*tLCG#%@")
    gamma = preset.get("ascii_gamma", 0.85)
    n = len(ramp)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    grid_h, grid_w = max(1, h // cell), max(1, w // cell)

    # Per-cell luminance + edge density.
    small = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    edges = cv2.Canny(gray, 60, 160)
    edge_small = cv2.resize(edges, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    norm = np.clip(np.power(small, gamma) + edge_small * 0.35, 0.0, 1.0)
    idx = np.clip((norm * (n - 1)).astype(np.int32), 0, n - 1)

    # Pre-render each glyph to a white-on-black tile.
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = cell / 22.0
    tiles = np.zeros((n, cell, cell, 3), dtype=np.uint8)
    for i, ch in enumerate(ramp):
        if ch != " ":
            cv2.putText(tiles[i], ch, (0, cell - 1), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    mapped = tiles[idx]  # (grid_h, grid_w, cell, cell, 3)

    # Tonal depth: dim each cell by its luminance so it isn't flat white.
    inten = np.clip(0.35 + norm * 0.65, 0.0, 1.0)[:, :, None, None, None]
    mapped = (mapped.astype(np.float32) * inten).astype(np.uint8)

    out = mapped.transpose(0, 2, 1, 3, 4).reshape(grid_h * cell, grid_w * cell, 3)

    if out.shape[0] != h or out.shape[1] != w:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:out.shape[0], :out.shape[1]] = out
        out = canvas

    return out


# =============================================================================
# KALEIDOSCOPE (radial mirror mandala)
# =============================================================================

def draw_kaleidoscope(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Kaleidoscope: fold the frame into a rotating radial mandala.

    Uses a polar transform, takes one angular wedge, mirrors it, tiles it around
    the full circle, then maps back to cartesian. The wedge is rotated over time
    for a slowly turning, hypnotic symmetry.
    """
    h, w = frame.shape[:2]
    segments = max(2, int(preset.get("kaleido_segments", 8)))
    spin = preset.get("kaleido_spin", 1.5)

    center = (w / 2.0, h / 2.0)
    max_radius = float(np.hypot(w, h) / 2.0)

    # To polar: rows = angle (0..360), cols = radius.
    polar = cv2.warpPolar(
        frame, (w, h), center, max_radius, cv2.WARP_POLAR_LINEAR
    )

    # One wedge of the angle axis, mirrored to make a seamless kaleidoscope cell.
    seg = max(2, h // segments)
    wedge = polar[:seg]
    cell = np.concatenate([wedge, wedge[::-1]], axis=0)  # mirror

    # Tile the cell to cover the full angle axis.
    reps = h // cell.shape[0] + 2
    tiled = np.tile(cell, (reps, 1, 1))[:h]

    # Rotate the mandala over time by rolling the angle axis.
    shift = int((frame_idx * spin) % h)
    tiled = np.roll(tiled, shift, axis=0)

    # Back to cartesian.
    output = cv2.warpPolar(
        tiled, (w, h), center, max_radius,
        cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP,
    )

    # Subtle bloom for richness.
    glow = cv2.GaussianBlur(output, (0, 0), 3)
    output = cv2.addWeighted(output, 1.0, glow, 0.35, 0)

    return output


# =============================================================================
# SUBJECT ISOLATION + TV STATIC
# =============================================================================

def _subject_mask(frame: np.ndarray) -> np.ndarray:
    """
    Estimate a filled mask of the main subject (person/object) in the frame.

    Heuristic, ML-free: enhance contrast, find edges, dilate into regions, then
    pick the largest contour biased toward the center and fill its convex hull.
    Falls back to a centered ellipse if nothing convincing is found.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 30, 100)

    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h, w), dtype=np.uint8)
    cx0, cy0 = w / 2.0, h / 2.0
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < h * w * 0.02:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        dist = np.hypot(cx - cx0, cy - cy0)
        centrality = 1.0 - dist / np.hypot(cx0, cy0)
        score = area * (0.4 + 0.6 * centrality)
        if score > best_score:
            best_score, best = score, c

    if best is not None:
        cv2.fillPoly(mask, [cv2.convexHull(best)], 255)

    if np.count_nonzero(mask) < h * w * 0.02:
        cv2.ellipse(mask, (int(cx0), int(cy0)), (w // 3, int(h * 0.45)), 0, 0, 360, 255, -1)

    # Soften the boundary slightly for a cleaner composite.
    mask = cv2.GaussianBlur(mask, (0, 0), 3)
    return mask


# SMPTE-style color bars (BGR), left -> right.
_SMPTE_BARS = [
    (255, 255, 255),  # white
    (0, 255, 255),    # yellow
    (255, 255, 0),    # cyan
    (0, 255, 0),      # green
    (255, 0, 255),    # magenta
    (0, 0, 255),      # red
    (255, 0, 0),      # blue
]


def _broadcast_static(h: int, w: int, frame_idx: int, color_amt: float, block: int) -> np.ndarray:
    """Build a frame of broken broadcast signal: SMPTE color bars corrupted by
    RGB noise, signal-bar tearing, a melting lower edge, scanlines and RGB
    fringing — the look from classic 'no signal' / datamosh test cards."""
    # 1. Color bars base.
    bars = np.zeros((h, w, 3), dtype=np.uint8)
    n = len(_SMPTE_BARS)
    bw = w // n
    for i, col in enumerate(_SMPTE_BARS):
        x0 = i * bw
        x1 = w if i == n - 1 else (i + 1) * bw
        bars[:, x0:x1] = col
    # Lower band: inverted/darker bars (PLUGE-ish) for the broadcast look.
    band_y = int(h * 0.72)
    for i, col in enumerate(_SMPTE_BARS[::-1]):
        x0 = i * bw
        x1 = w if i == n - 1 else (i + 1) * bw
        bars[band_y:, x0:x1] = tuple(int(c * 0.35) for c in col)

    # 2. Animated RGB noise blended over the bars.
    nh, nw = max(1, h // block), max(1, w // block)
    gray_noise = np.random.randint(0, 256, (nh, nw, 1), dtype=np.uint8).repeat(3, axis=2)
    color_noise = np.random.randint(0, 256, (nh, nw, 3), dtype=np.uint8)
    noise = cv2.addWeighted(gray_noise, 1.0 - color_amt, color_noise, color_amt, 0)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)
    static = cv2.addWeighted(bars, 0.55, noise, 0.65, 0)

    # 3. Horizontal signal-bar tearing.
    rng = np.random.default_rng(frame_idx * 7 + 1)
    for _ in range(int(rng.integers(4, 9))):
        by = int(rng.integers(0, h))
        bh = int(rng.integers(2, max(3, h // 14)))
        shift = int(rng.integers(-w // 6, w // 6))
        static[by:by + bh] = np.roll(static[by:by + bh], shift, axis=1)

    # 4. Melting lower edge: rows near the bottom copy from progressively higher
    #    rows, creating the downward "signal melt" smear from the reference.
    melt_start = int(h * 0.78)
    for y in range(melt_start, h):
        amt = int((y - melt_start) / max(1, h - melt_start) * 18)
        if amt > 0:
            src = max(melt_start, y - amt)
            static[y] = static[src]

    # 5. Scanlines + RGB fringing.
    static[1::2] = (static[1::2].astype(np.float32) * 0.75).astype(np.uint8)
    b, g, r = cv2.split(static)
    static = cv2.merge([np.roll(b, -3, axis=1), g, np.roll(r, 3, axis=1)])
    return static


def draw_tv_static(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    TV Static: isolate the person (MediaPipe selfie segmentation, with a
    heuristic fallback) and replace ONLY them with broken broadcast signal —
    corrupted SMPTE color bars, RGB noise, tearing and a melting edge — while
    the real background stays untouched.
    """
    h, w = frame.shape[:2]
    block = max(1, int(preset.get("static_block", 2)))   # noise chunkiness
    color_amt = float(preset.get("static_color", 0.6))   # 0=gray .. 1=rgb

    # Real person isolation when available; heuristic subject mask otherwise.
    mask = get_person_mask(frame)
    if mask is None:
        mask = _subject_mask(frame)
    maskf = (mask.astype(np.float32) / 255.0)[:, :, None]

    static = _broadcast_static(h, w, frame_idx, color_amt, block)

    # Composite: broadcast static inside the subject, real video outside.
    output = frame.astype(np.float32) * (1.0 - maskf) + static.astype(np.float32) * maskf
    output = output.astype(np.uint8)

    # Glowing torn edge around the silhouette so it reads as "signal loss".
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    edge_glow = cv2.GaussianBlur(cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR), (0, 0), 4)
    output = cv2.addWeighted(output, 1.0, edge_glow, 0.45, 0)

    return output


# =============================================================================
# CHROMATIC GHOST (RGB time-delay motion trails)
# =============================================================================

_ghost_buffer: np.ndarray | None = None
_ghost_pos: int = 0


def draw_chromatic_ghost(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Chromatic Ghost: separate the R/G/B channels in TIME. Red shows the present,
    green a few frames ago, blue further back — so anything moving leaves a
    rainbow comet trail while still areas stay true color. Reliably gorgeous.
    """
    global _ghost_buffer, _ghost_pos

    h, w = frame.shape[:2]
    n = max(3, int(preset.get("ghost_frames", 10)))
    sat = float(preset.get("ghost_saturation", 1.4))

    if (_ghost_buffer is None or _ghost_buffer.shape[0] != n
            or _ghost_buffer.shape[1:3] != (h, w)):
        _ghost_buffer = np.repeat(frame[None], n, axis=0).copy()
        _ghost_pos = 0

    _ghost_buffer[_ghost_pos] = frame

    newest = _ghost_pos
    mid = (_ghost_pos - n // 2) % n
    oldest = (_ghost_pos - (n - 1)) % n

    out = np.empty_like(frame)
    out[:, :, 0] = _ghost_buffer[oldest][:, :, 0]   # B from the past
    out[:, :, 1] = _ghost_buffer[mid][:, :, 1]      # G mid
    out[:, :, 2] = _ghost_buffer[newest][:, :, 2]   # R present

    _ghost_pos = (_ghost_pos + 1) % n

    # Punch up saturation so the trails read as vivid color, then bloom.
    if sat != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    glow = cv2.GaussianBlur(out, (0, 0), 3)
    out = cv2.addWeighted(out, 1.0, glow, 0.4, 0)
    return out


# =============================================================================
# CRYSTALLIZE (low-poly Delaunay triangulation mosaic)
# =============================================================================

def draw_crystallize(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Crystallize: shatter the frame into a low-poly mosaic of flat-shaded
    triangles. Feature points concentrate detail on the subject while a grid
    guarantees full coverage; each triangle is filled with the color sampled at
    its centroid. Premium generative-art look.
    """
    h, w = frame.shape[:2]
    cells = int(preset.get("cells", 600))
    grid_step = int(preset.get("grid_step", max(36, (w + h) // 34)))
    facet_edges = bool(preset.get("facet_edges", True))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=cells, qualityLevel=0.01, minDistance=8)

    pts: list[tuple[float, float]] = []
    if corners is not None:
        pts.extend((float(c.ravel()[0]), float(c.ravel()[1])) for c in corners)

    # Grid + border points so the entire frame is tiled (no black gaps).
    gx = list(range(0, w, grid_step)) + [w - 1]
    gy = list(range(0, h, grid_step)) + [h - 1]
    for y in gy:
        for x in gx:
            pts.append((float(x), float(y)))

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for (x, y) in pts:
        if 0 <= x < w and 0 <= y < h:
            subdiv.insert((float(x), float(y)))

    output = np.zeros((h, w, 3), dtype=np.uint8)
    for t in subdiv.getTriangleList():
        tri = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], dtype=np.float32)
        if (tri[:, 0] < 0).any() or (tri[:, 0] > w - 1).any():
            continue
        if (tri[:, 1] < 0).any() or (tri[:, 1] > h - 1).any():
            continue
        cx = int(np.clip(tri[:, 0].mean(), 0, w - 1))
        cy = int(np.clip(tri[:, 1].mean(), 0, h - 1))
        # Average a small patch around the centroid for a stable facet color.
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        color = frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        color = [int(c) for c in color]
        poly = tri.astype(np.int32)
        cv2.fillConvexPoly(output, poly, color, cv2.LINE_AA)
        if facet_edges:
            cv2.polylines(output, [poly], True, [int(c * 0.65) for c in color], 1, cv2.LINE_AA)

    return output


# =============================================================================
# HALFTONE (classic black-and-white newsprint dots)
# =============================================================================

_halftone_cache: dict = {}


def draw_halftone(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Halftone: classic black-and-white newsprint look. Black dots on a white
    page, where each dot grows as the source gets darker. Crisp, high-contrast,
    fully vectorized via a cached radial cell pattern.
    """
    h, w = frame.shape[:2]
    dot = max(4, int(preset.get("dot_spacing", 8)))
    gamma = float(preset.get("dot_gamma", 0.9))
    contrast = float(preset.get("dot_contrast", 1.25))

    key = (h, w, dot)
    patt = _halftone_cache.get(key)
    if patt is None:
        yy, xx = np.mgrid[0:h, 0:w]
        cx = (xx % dot) - (dot - 1) / 2.0
        cy = (yy % dot) - (dot - 1) / 2.0
        patt = (np.sqrt(cx * cx + cy * cy) / ((dot / 2.0) * np.sqrt(2.0))).astype(np.float32)
        _halftone_cache[key] = patt

    # Per-cell brightness (blocky via downsample -> nearest upsample).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gw, gh = max(1, w // dot), max(1, h // dot)
    small = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
    lum = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0

    # Punch contrast, then darker source -> larger black dot.
    lum = np.clip((lum - 0.5) * contrast + 0.5, 0.0, 1.0)
    radius = np.power(np.clip(1.0 - lum, 0.0, 1.0), gamma)

    mask = patt <= radius
    output = np.full((h, w, 3), 255, dtype=np.uint8)  # white page
    output[mask] = (0, 0, 0)                            # black ink dots
    return output


# =============================================================================
# LIGHT TRAILS (long-exposure glowing motion trails)
# =============================================================================

_light_canvas: np.ndarray | None = None


def draw_light_trails(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Light Trails: long-exposure light painting. The brightest parts of each
    frame are accumulated and slowly decayed, so anything bright and moving
    paints a glowing, fading streak over a darkened version of the scene.
    Perfectly on-brand for "Aftertrace".
    """
    global _light_canvas

    h, w = frame.shape[:2]
    decay = float(preset.get("trail_decay", 0.93))
    pct = float(preset.get("bright_pct", 85))      # only the top (100-pct)% trails
    floor = int(preset.get("bright_thresh", 50))   # but never below this
    boost = float(preset.get("trail_boost", 1.3))

    if _light_canvas is None or _light_canvas.shape[:2] != (h, w):
        _light_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Decay the accumulated light (older streaks fade out).
    _light_canvas = (_light_canvas.astype(np.float32) * decay).astype(np.uint8)

    # Contribution: only the brightest parts of the current frame (adaptive to
    # exposure via a percentile) so it paints trails instead of flooding.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = max(floor, int(np.percentile(gray, pct)))
    bright = gray > thresh
    contrib = np.zeros_like(frame)
    contrib[bright] = frame[bright]
    contrib = np.clip(contrib.astype(np.float32) * boost, 0, 255).astype(np.uint8)

    # Keep the brightest of (decayed history, new light) -> persistent trails.
    _light_canvas = np.maximum(_light_canvas, contrib)

    # Bloom and composite over a dark version of the scene for context.
    glow = cv2.GaussianBlur(_light_canvas, (0, 0), 6)
    trails = cv2.addWeighted(_light_canvas, 1.0, glow, 0.9, 0)
    output = cv2.add((frame * 0.10).astype(np.uint8), trails)
    return output


# =============================================================================
# INK (black pen-and-ink sketch on white paper)
# =============================================================================

_ink_cache: dict = {}


def draw_ink(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Ink: a clean black pen-and-ink drawing on white paper. Crisp contour lines
    plus two layers of cross-hatching whose density follows shadow, for a
    hand-drawn engraving feel. Pure black on white, high contrast and elegant.
    """
    h, w = frame.shape[:2]
    hatch = max(4, int(preset.get("ink_hatch", 7)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    lum = smooth.astype(np.float32) / 255.0

    # Clean contour lines (multi-scale Canny, lightly cleaned).
    edges = cv2.bitwise_or(cv2.Canny(smooth, 30, 90), cv2.Canny(smooth, 60, 150))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    # Cached diagonal hatch line patterns (two opposing directions).
    key = (h, w, hatch)
    cached = _ink_cache.get(key)
    if cached is None:
        yy, xx = np.mgrid[0:h, 0:w]
        h1 = ((xx + yy) % hatch) == 0          # "/" lines
        h2 = ((xx - yy) % hatch) == 0          # "\" lines
        h3 = ((xx + yy) % (hatch // 2 if hatch >= 8 else hatch)) == 0  # denser
        cached = (h1, h2, h3)
        _ink_cache[key] = cached
    h1, h2, h3 = cached

    output = np.full((h, w, 3), 255, dtype=np.uint8)  # white paper

    # Shadow hatching: mids get one direction, darks get cross-hatch, deepest
    # darks get a denser pass.
    mid = lum < 0.62
    dark = lum < 0.42
    deep = lum < 0.22
    output[mid & h1] = (0, 0, 0)
    output[dark & h2] = (0, 0, 0)
    output[deep & h3] = (0, 0, 0)

    # Ink the contour lines last so they stay crisp on top.
    output[edges > 0] = (0, 0, 0)
    return output


# =============================================================================
# NEON GLOW (flowing rainbow neon outline with heavy bloom)
# =============================================================================

_neon_cache: dict = {}


def draw_neon_glow(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Neon Glow: trace the subject as a glowing rainbow neon outline on black.
    Edge lines are hue-cycled across the frame and over time so the outline
    flows like animated neon tubing, finished with a heavy multi-pass bloom.
    """
    h, w = frame.shape[:2]
    speed = float(preset.get("neon_speed", 2.0))
    thickness = int(preset.get("neon_thickness", 2))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 7, 60, 60)
    edges = cv2.bitwise_or(cv2.Canny(smooth, 40, 110), cv2.Canny(smooth, 80, 180))
    if thickness > 1:
        edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8))

    # Cached diagonal hue ramp; shift it over time for the flowing-neon feel.
    key = (h, w)
    base_hue = _neon_cache.get(key)
    if base_hue is None:
        yy, xx = np.mgrid[0:h, 0:w]
        base_hue = (((xx + yy) * 0.35) % 180).astype(np.float32)
        _neon_cache[key] = base_hue

    hue = ((base_hue + frame_idx * speed) % 180).astype(np.uint8)
    sat = np.full((h, w), 255, dtype=np.uint8)
    val = edges  # 0 off the lines, 255 on them
    hsv = cv2.merge([hue, sat, val])
    neon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Multi-pass bloom for the luminous tube glow.
    glow1 = cv2.GaussianBlur(neon, (0, 0), 3)
    glow2 = cv2.GaussianBlur(neon, (0, 0), 9)
    output = cv2.addWeighted(neon, 1.0, glow1, 0.9, 0)
    output = cv2.addWeighted(output, 1.0, glow2, 0.6, 0)
    return output


# =============================================================================
# POINT CLOUD (TouchDesigner-style 3D dotted scan, black & white)
# =============================================================================

_point_cloud_cache: dict = {}


def draw_point_cloud(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Point Cloud: a TouchDesigner-style 3D point-cloud scan in black & white.

    The subject is sampled on a grid into points; each point's brightness drives
    its depth (Z), the whole cloud slowly yaws so depth reads as a rotating
    volume (parallax), animated noise jitters the points, and random thinning
    gives the sparse dotted look. White points on black, with a soft glow.

    Mirrors the reference TD graph: threshold/isolate -> points with depth ->
    4D noise displacement -> rotating 3D render -> thin -> glow.
    """
    h, w = frame.shape[:2]
    step = max(3, int(preset.get("pc_step", 6)))
    min_bright = int(preset.get("pc_min_bright", 32))
    depth_scale = float(preset.get("pc_depth", 90.0))
    pop = float(preset.get("pc_pop", 10.0))
    noise_amp = float(preset.get("pc_noise", 2.5))
    yaw_amp = float(preset.get("pc_yaw", 0.42))
    thin_pct = int(preset.get("pc_thin", 78))   # percent of points kept

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Cached grid + per-point phase/thinning hash (depends only on h, w, step).
    key = (h, w, step)
    cached = _point_cloud_cache.get(key)
    if cached is None:
        gxs = np.arange(0, w, step)
        gys = np.arange(0, h, step)
        GX, GY = np.meshgrid(gxs, gys)
        phase = (GX * 12.9 + GY * 78.2).astype(np.float32)
        keep_hash = ((GX * 73 + GY * 131) % 100).astype(np.int32)
        cached = (GX.astype(np.float32), GY.astype(np.float32), phase, keep_hash,
                  GX.astype(np.int32), GY.astype(np.int32))
        _point_cloud_cache[key] = cached
    GXf, GYf, phase, keep_hash, GXi, GYi = cached

    bright = gray[GYi, GXi].astype(np.float32)

    # Subject isolation: real person mask when available, else brightness.
    seg = get_person_mask(frame)
    if seg is not None:
        subject = seg[GYi, GXi] > 110
    else:
        subject = bright > min_bright

    mask = subject & (bright > min_bright) & (keep_hash < thin_pct)
    if not mask.any():
        return np.zeros((h, w, 3), dtype=np.uint8)

    cx = w / 2.0
    theta = np.sin(frame_idx * 0.02) * yaw_amp   # slow oscillating yaw
    ct, st = np.cos(theta), np.sin(theta)

    # Brightness -> depth (centered so mid-grey sits near the pivot).
    z = (bright / 255.0 - 0.4) * depth_scale
    X = GXf - cx

    # Yaw about the vertical axis: project to screen x + a view-depth term.
    sx = cx + X * ct + z * st
    view_depth = -X * st + z * ct

    # Brightness also pops points slightly upward for relief.
    sy = GYf - (bright / 255.0) * pop

    # Animated noise jitter (stable per-point phase, evolves over time).
    sx = sx + np.sin(frame_idx * 0.15 + phase) * noise_amp
    sy = sy + np.cos(frame_idx * 0.13 + phase * 1.3) * noise_amp

    # Bright WHITE points. Depth only nudges brightness a little (forward points
    # the brightest) so the cloud always reads as crisp white, not grey.
    rng = float(np.ptp(view_depth)) + 1e-5
    depth_fwd = (view_depth - view_depth.min()) / rng        # 0..1, 1 = toward viewer
    inten = np.clip(205 + depth_fwd * 50, 0, 255)            # 205..255 = bright white
    inten = inten * (0.9 + 0.1 * bright / 255.0)             # tiny source variation

    xs = sx[mask].astype(np.int32)
    ys = sy[mask].astype(np.int32)
    vals = inten[mask].astype(np.float32)

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys, vals = xs[valid], ys[valid], vals[valid]

    # Scatter points (keep the brightest where they overlap).
    canvas = np.zeros((h * w,), dtype=np.float32)
    np.maximum.at(canvas, ys * w + xs, vals)
    canvas = np.clip(canvas, 0, 255).reshape(h, w).astype(np.uint8)

    # Fatten single pixels into visible round dots.
    dot = int(preset.get("pc_dot", 2))
    if dot >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dot + 1, dot + 1))
        canvas = cv2.dilate(canvas, k)

    out = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    # Additive bloom so the white dots glow and pop (saturating add stays bright).
    glow = cv2.GaussianBlur(out, (0, 0), 2.2)
    out = cv2.add(out, (glow.astype(np.float32) * 0.85).astype(np.uint8))
    return out
