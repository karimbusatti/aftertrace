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
    audio_level: float = 0.0,
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
        audio_level: Per-frame audio loudness (0..1) for audio-reactive effects

    Returns:
        Rendered frame with effects applied
    """
    colors = get_preset_colors(preset)

    if overlay_mode:
        # OVERLAY MODE: Keep original visible, blend effects on top
        output = _draw_frame_overlay(frame, points, preset, colors, frame_idx, face_data, audio_level)
    else:
        # NORMAL MODE: Replace background with effect
        output = _draw_frame_replace(frame, points, preset, colors, frame_idx, face_data, audio_level)
    
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
    audio_level: float = 0.0,
) -> np.ndarray:
    """Normal mode: darken background and draw effects on top."""

    # Check for text-based effects first (they replace the entire pipeline)
    text_result = apply_text_effect(frame, preset, colors, frame_idx=frame_idx, points=points, face_data=face_data, audio_level=audio_level)
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
    audio_level: float = 0.0,
) -> np.ndarray:
    """
    Overlay mode: blend effects at ~40% over the original frame.
    Shows "what the algorithm sees" on top of reality.
    """
    # Keep original frame intact
    original = frame.copy()

    # Check for text-based effects
    text_result = apply_text_effect(frame, preset, colors, frame_idx=frame_idx, points=points, face_data=face_data, audio_level=audio_level)
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
    frame[::3] = (frame[::3].astype(np.float32) * 0.7).astype(np.uint8)

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


_ocular_vignette_cache: dict = {}


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

    # Subtle vignette for depth (cached per resolution).
    vig = _ocular_vignette_cache.get((h, w))
    if vig is None:
        yy, xx = np.ogrid[:h, :w]
        cyv, cxv = h / 2.0, w / 2.0
        d = np.sqrt(((xx - cxv) / cxv) ** 2 + ((yy - cyv) / cyv) ** 2)
        vig = np.clip(1.0 - (d - 0.6) * 0.5, 0.55, 1.0).astype(np.float32)[:, :, None]
        _ocular_vignette_cache[(h, w)] = vig
    output = (output.astype(np.float32) * vig).astype(np.uint8)

    return output


# =============================================================================
# MATRIX MODE EFFECT (Green data rain)
# =============================================================================

_matrix_cache: dict = {}


def draw_matrix_mode(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Matrix Mode: green digital rain revealing the subject.

    Grid-locked rain like the film titles: every glyph sits in a fixed cell and
    each column's head steps down the grid at its own (deterministic) speed
    with its own trail length. Glyphs are stable and only occasionally mutate,
    so the rain reads as falling code instead of 30Hz static. Fully vectorized
    via pre-rendered glyph tiles, finished with phosphor glow + scanlines.
    """
    h, w = frame.shape[:2]

    cw = int(preset.get("matrix_cell_w", 10))
    ch = int(preset.get("matrix_cell_h", 14))
    # Hershey fonts are ASCII-only (the old katakana charset rendered as '?').
    chars = preset.get("matrix_chars", "0123456789Z=+*:<>#$&@XKM")
    n = len(chars)

    grid_w, grid_h = max(1, w // cw), max(1, h // ch)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Cached per-resolution constants: glyph tiles + per-column params ---
    key = (grid_w, grid_h, cw, ch, chars)
    cached = _matrix_cache.get(key)
    if cached is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = ch / 34.0
        tiles = np.zeros((n, ch, cw), dtype=np.uint8)  # glyph alpha masks
        for i, c in enumerate(chars):
            cv2.putText(tiles[i], c, (0, ch - 3), font, fs, 255, 1, cv2.LINE_AA)

        cols = np.arange(grid_w)
        rows = np.arange(grid_h)
        # Deterministic per-column personality (speed / trail / phase).
        col_hash = (cols * 2654435761) % 2**32
        speed = 1 + (col_hash >> 3) % 3            # head advances every 1-3 frames
        trail = 8 + (col_hash >> 7) % 14           # trail length in cells
        phase = (col_hash >> 11) % 199             # column start offset
        # Stable per-cell base glyph + which cells mutate over time.
        cell_hash = (cols[None, :] * 131 + rows[:, None] * 71) * 2654435761 % 2**32
        base_glyph = (cell_hash >> 5) % n
        mutates = ((cell_hash >> 9) % 3) == 0      # ~1/3 of cells cycle glyphs
        cached = (tiles, speed, trail, phase, base_glyph, mutates, rows, cols)
        _matrix_cache[key] = cached
    tiles, speed, trail, phase, base_glyph, mutates, rows, cols = cached

    # --- Rain intensity per cell (vectorized) ---
    cycle = grid_h + trail                                    # per-column loop length
    head_row = (phase + frame_idx // speed) % cycle           # (grid_w,)
    dist = head_row[None, :] - rows[:, None]                  # rows above the head
    on = (dist >= 0) & (dist < trail[None, :])
    fade = np.where(on, 1.0 - dist / np.maximum(trail[None, :], 1), 0.0).astype(np.float32)
    # Ease the tail (quadratic) so trails melt out instead of ending abruptly.
    fade *= fade

    # Subject reveal: rain is brighter where the source is bright.
    cell_lum = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    fade *= 0.35 + 0.65 * cell_lum

    # --- Glyph selection: stable, with occasional mutation on ~1/3 of cells ---
    glyph_idx = np.where(mutates, (base_glyph + frame_idx // 6) % n, base_glyph)

    # --- Compose: tile alpha x per-cell color (green body, pale head) ---
    alpha = tiles[glyph_idx].transpose(0, 2, 1, 3).reshape(grid_h * ch, grid_w * cw)
    alpha = alpha.astype(np.float32) / 255.0

    green = np.array((70, 255, 0), dtype=np.float32)      # BGR phosphor green
    head_col = np.array((215, 255, 215), dtype=np.float32)
    is_head = (dist == 0)
    cell_color = green[None, None, :] * fade[:, :, None]
    cell_color[is_head] = head_col * np.maximum(fade[is_head], 0.85)[:, None]

    color_full = cv2.resize(cell_color, (grid_w * cw, grid_h * ch),
                            interpolation=cv2.INTER_NEAREST)
    rain = (color_full * alpha[:, :, None]).astype(np.uint8)

    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[:rain.shape[0], :rain.shape[1]] = rain[:h, :w]

    # Faint green-tinted ghost of the scene so the subject reads through.
    output[:, :, 1] = np.maximum(output[:, :, 1], (gray * 0.12).astype(np.uint8))

    # Phosphor glow: the rain emits light.
    glow = cv2.GaussianBlur(output, (0, 0), 2.5)
    output = cv2.add(output, (glow.astype(np.float32) * 0.55).astype(np.uint8))

    # CRT scanlines (vectorized).
    output[::3] = (output[::3].astype(np.float32) * 0.85).astype(np.uint8)

    return output


def apply_text_effect(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
    points: list[TrackedPoint] | None = None,
    face_data: dict | None = None,
    audio_level: float = 0.0,
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
        return draw_point_cloud(frame, preset, colors, frame_idx=frame_idx, audio_level=audio_level)
    elif text_mode == "blacktone":
        return draw_blacktone(frame, preset, colors, frame_idx=frame_idx)
    elif text_mode == "cursor_cloud":
        return draw_cursor_cloud(frame, preset, colors, frame_idx=frame_idx, audio_level=audio_level)

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
            # Filled box with scanlines (blend only the crop - a full-frame
            # copy per box was the hottest path in this effect)
            region = output[by:by+box_h, bx:bx+box_w]
            fill = np.full_like(region, fill_color)
            output[by:by+box_h, bx:bx+box_w] = cv2.addWeighted(region, 0.4, fill, 0.6, 0)
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
                region2 = output[by2:by2+box_h, bx2:bx2+box_w]
                fill2 = np.full_like(region2, fill_color2)
                output[by2:by2+box_h, bx2:bx2+box_w] = cv2.addWeighted(region2, 0.4, fill2, 0.6, 0)
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

_blob_tracks: list[dict] = []
_blob_next_id: int = 0


def _box_iou(a, b) -> float:
    """IoU of two [x, y, w, h] boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def draw_blob_track(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
) -> np.ndarray:
    """
    Blob Track effect: clean minimal tracking - TouchDesigner style.

    Thin white boxes with corner ticks, connection lines, and STABLE tracks:
    detections are matched frame-to-frame by IoU, boxes are exponentially
    smoothed, IDs persist on the same object, and tracks fade in on birth and
    fade out for a few frames when lost - so the overlay reads as real
    object tracking instead of per-frame detection flicker.
    """
    global _blob_next_id

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

    detections: list[np.ndarray] = []
    scored = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if x <= edge_margin or y <= edge_margin:
            continue
        if x + bw >= w - edge_margin or y + bh >= h - edge_margin:
            continue
        scored.append((area, (x, y, bw, bh)))
    scored.sort(key=lambda s: s[0], reverse=True)
    detections = [np.array(box, dtype=np.float32) for _, box in scored[:max_blobs]]

    # =========================================================================
    # TEMPORAL TRACKING: match detections to existing tracks by IoU (greedy,
    # best pair first), smooth matched boxes, age out lost tracks.
    # =========================================================================
    smoothing = float(preset.get("track_smoothing", 0.45))  # det weight per frame
    max_misses = int(preset.get("track_max_misses", 5))

    pairs = []
    for ti, tr in enumerate(_blob_tracks):
        for di, det in enumerate(detections):
            iou = _box_iou(tr["box"], det)
            if iou > 0.15:
                pairs.append((iou, ti, di))
    pairs.sort(reverse=True)

    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for iou, ti, di in pairs:
        if ti in matched_tracks or di in matched_dets:
            continue
        matched_tracks.add(ti)
        matched_dets.add(di)
        tr = _blob_tracks[ti]
        tr["box"] += (detections[di] - tr["box"]) * smoothing
        tr["age"] += 1
        tr["misses"] = 0

    for ti, tr in enumerate(_blob_tracks):
        if ti not in matched_tracks:
            tr["misses"] += 1

    _blob_tracks[:] = [t for t in _blob_tracks if t["misses"] <= max_misses]

    for di, det in enumerate(detections):
        if di not in matched_dets:
            _blob_tracks.append({
                "id": _blob_next_id, "box": det.copy(), "age": 0, "misses": 0,
            })
            _blob_next_id += 1

    if not _blob_tracks:
        return output

    # Visibility: fade in over the first frames, fade out while missing.
    def track_alpha(tr) -> float:
        fade_in = min(1.0, (tr["age"] + 1) / 4.0)
        fade_out = 1.0 - tr["misses"] / (max_misses + 1.0)
        return fade_in * fade_out

    font = cv2.FONT_HERSHEY_SIMPLEX

    # White connection mesh between nearby tracks (alpha-weighted).
    max_connection_dist = preset.get("max_connection_dist", 200)
    centers = [
        (tr["box"][0] + tr["box"][2] / 2.0, tr["box"][1] + tr["box"][3] / 2.0)
        for tr in _blob_tracks
    ]
    alphas = [track_alpha(tr) for tr in _blob_tracks]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if dist < max_connection_dist:
                a = (1.0 - dist / max_connection_dist) * min(alphas[i], alphas[j])
                c = int((150 + 105 * a) * a)
                if c > 8:
                    cv2.line(output, (int(centers[i][0]), int(centers[i][1])),
                             (int(centers[j][0]), int(centers[j][1])), (c, c, c), 1, cv2.LINE_AA)

    # Draw each track: thin box + corner ticks + persistent ID label.
    for tr, alpha in zip(_blob_tracks, alphas):
        if alpha <= 0.05:
            continue
        x, y, bw, bh = (int(round(v)) for v in tr["box"])
        x2, y2 = x + bw, y + bh
        cval = int(255 * alpha)
        box_color = (cval, cval, cval)

        cv2.rectangle(output, (x, y), (x2, y2), box_color, 1, cv2.LINE_AA)

        # Subtle corner ticks (no plus in the middle).
        cl = max(4, min(bw, bh) // 6)
        for (px, py, dx, dy) in [
            (x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(output, (px, py), (px + dx * cl, py), box_color, 2, cv2.LINE_AA)
            cv2.line(output, (px, py), (px, py + dy * cl), box_color, 2, cv2.LINE_AA)

        # Persistent ID label above the box (shadowed for legibility).
        box_size = min(bw, bh)
        fscale = max(0.3, min(0.42, box_size / 220.0))
        label = f"ID {tr['id'] % 100:02d}"
        ly = y - 5 if y > 14 else y + int(box_size * 0.2) + 6
        cv2.putText(output, label, (x + 1, ly + 1), font, fscale,
                    (0, 0, 0), 1, cv2.LINE_AA)
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

_number_cloud_cache: dict = {}


def draw_number_cloud(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Numeric Aura: the subject rendered as glowing binary in a sci-fi hex field.

    - Dim deep-blue hex grid scrolling upward through the whole frame, faintly
      brightening where it passes the subject
    - Bright cyan 0/1 digits on the subject, white-hot in the core
    - Each digit cell flips on its own 4-11 frame cadence (staggered by a cell
      hash) so the field shimmers organically instead of strobing at 30Hz
    - Person segmentation when available, contour heuristic otherwise
    - Fully vectorized via cached glyph-tile banks (was thousands of putText
      calls per frame)
    """
    h, w = frame.shape[:2]

    # === SUBJECT MASK (soft falloff) ===
    # The mask is only ever sampled on coarse glyph grids, so build and blur it
    # at 1/8 resolution - full-res masking + a sigma-25 blur dominated the
    # frame cost for zero visible difference.
    seg = get_person_mask(frame)
    if seg is not None and np.count_nonzero(seg) > h * w * 0.02:
        mask_src = seg
    else:
        mask_src = _subject_mask(cv2.resize(frame, (w // 4, h // 4),
                                            interpolation=cv2.INTER_AREA))
    subject = cv2.resize(mask_src, (max(8, w // 8), max(8, h // 8)),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    subject = cv2.GaussianBlur(subject, (0, 0), 3.0)

    # === PALETTE (BGR) ===
    blue_dim = np.array((140, 50, 0), dtype=np.float32)
    cyan_bright = np.array((255, 200, 50), dtype=np.float32)
    white_hot = np.array((250, 250, 250), dtype=np.float32)

    # =========================================================================
    # LAYER 1: scrolling hex background (16x20 cells)
    # =========================================================================
    bw_, bh_ = 16, 20
    bgw, bgh = max(1, w // bw_), max(1, h // bh_) + 2  # +2 rows for scroll wrap

    key = ("bg", bw_, bh_)
    bg_tiles = _number_cloud_cache.get(key)
    if bg_tiles is None:
        hex_chars = "0123456789ABCDEF"
        bg_tiles = np.zeros((16, bh_, bw_), dtype=np.uint8)
        for i, c in enumerate(hex_chars):
            cv2.putText(bg_tiles[i], c, (2, bh_ - 5), cv2.FONT_HERSHEY_PLAIN,
                        1.0, 255, 1, cv2.LINE_AA)
        _number_cloud_cache[key] = bg_tiles

    scroll = frame_idx * 1.5
    row_off = int(scroll) // bh_
    pix_off = int(scroll) % bh_

    brows = np.arange(bgh)[:, None] + row_off
    bcols = np.arange(bgw)[None, :]
    bg_idx = ((bcols * 7 + brows * 13 + frame_idx // 12) % 16).astype(np.int32)

    bg_alpha = bg_tiles[bg_idx].transpose(0, 2, 1, 3).reshape(bgh * bh_, bgw * bw_)
    bg_alpha = np.roll(bg_alpha, -pix_off, axis=0)[:h, :]  # smooth upward scroll
    bg_alpha = bg_alpha.astype(np.float32) / 255.0

    # Cell brightness: dim everywhere, lifting near the subject.
    sub_small = cv2.resize(subject, (bgw, bgh - 2), interpolation=cv2.INTER_AREA)
    bg_gain = 0.15 + sub_small * 0.5
    bg_gain_full = cv2.resize(bg_gain, (bgw * bw_, h), interpolation=cv2.INTER_NEAREST)

    output = (blue_dim[None, None, :] * (bg_alpha * bg_gain_full)[:, :, None])

    # Pad to frame width if the grid doesn't divide evenly.
    if output.shape[1] != w:
        padded = np.zeros((h, w, 3), dtype=np.float32)
        padded[:, :output.shape[1]] = output[:, :w]
        output = padded

    # =========================================================================
    # LAYER 2: binary digits on the subject (24x30 cells)
    # =========================================================================
    fw_, fh_ = 24, 30
    fgw, fgh = max(1, w // fw_), max(1, h // fh_)

    key = ("fg", fw_, fh_)
    fg_tiles = _number_cloud_cache.get(key)
    if fg_tiles is None:
        # index: 0 blank, 1='0' normal, 2='1' normal, 3='0' hot, 4='1' hot
        fg_tiles = np.zeros((5, fh_, fw_), dtype=np.uint8)
        for i, (c, th, fs) in enumerate([("0", 1, 1.4), ("1", 1, 1.4),
                                         ("0", 2, 1.7), ("1", 2, 1.7)]):
            cv2.putText(fg_tiles[i + 1], c, (3, fh_ - 8), cv2.FONT_HERSHEY_PLAIN,
                        fs, 255, th, cv2.LINE_AA)
        _number_cloud_cache[key] = fg_tiles

    frows = np.arange(fgh)[:, None]
    fcols = np.arange(fgw)[None, :]
    cell_hash = (fcols * 131 + frows * 71) * 2654435761 % 2**32
    # Staggered flips: every cell has its own period (4-11 frames) and phase.
    period = 4 + (cell_hash >> 4) % 8
    bit = (((cell_hash >> 8) + frame_idx // period) % 2).astype(np.int32)

    mask_small = cv2.resize(subject, (fgw, fgh), interpolation=cv2.INTER_AREA)
    on = mask_small > 0.2
    hot = mask_small > 0.75

    fg_idx = np.where(on, 1 + bit + np.where(hot, 2, 0), 0)
    fg_alpha = fg_tiles[fg_idx].transpose(0, 2, 1, 3).reshape(fgh * fh_, fgw * fw_)
    fg_alpha = fg_alpha.astype(np.float32) / 255.0

    # Depth-blended color: dim blue at the silhouette edge -> cyan -> white core.
    blend = np.clip((mask_small - 0.2) / 0.55, 0.0, 1.0)[:, :, None]
    cell_col = blue_dim[None, None, :] * (1 - blend) + cyan_bright[None, None, :] * blend
    cell_col[hot] = white_hot
    col_full = cv2.resize(cell_col, (fgw * fw_, fgh * fh_),
                          interpolation=cv2.INTER_NEAREST)

    fg_layer = np.zeros((h, w, 3), dtype=np.float32)
    fh_full, fw_full = min(h, fgh * fh_), min(w, fgw * fw_)
    fg_layer[:fh_full, :fw_full] = (col_full * fg_alpha[:, :, None])[:fh_full, :fw_full]

    output = np.clip(output + fg_layer, 0, 255).astype(np.uint8)

    # Cyan glow around the bright digits.
    glow_src = np.zeros((h, w, 3), dtype=np.uint8)
    glow_mask = cv2.resize((mask_small > 0.5).astype(np.float32), (w, h),
                           interpolation=cv2.INTER_NEAREST)[:, :, None]
    glow_src[:] = (fg_layer * glow_mask).astype(np.uint8)
    glow = cv2.GaussianBlur(glow_src, (0, 0), 5.0)
    output = cv2.add(output, (glow.astype(np.float32) * 0.85).astype(np.uint8))

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
    
    # Initialize or reset trail canvas if needed. float32: repeated uint8
    # decay quantized the falloff, so faint trails died in visible steps.
    if _motion_trace_trail_canvas is None or _motion_trace_trail_canvas.shape[:2] != (h, w):
        _motion_trace_trail_canvas = np.zeros((h, w, 3), dtype=np.float32)

    # Fade previous trails (creates the comet/persistence effect)
    _motion_trace_trail_canvas *= trail_fade

    # Need previous frame for optical flow
    if _motion_trace_prev_frame is None or _motion_trace_prev_frame.shape != gray.shape:
        _motion_trace_prev_frame = gray.copy()
        # Return original frame with empty trail on first frame
        return cv2.addWeighted(
            frame, frame_alpha,
            _motion_trace_trail_canvas.astype(np.uint8), trail_alpha, 0,
        )
    
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
    # Update previous frame
    _motion_trace_prev_frame = gray.copy()

    # Color palette for variety (BGR - cyan/blue tones)
    flow_colors = [
        flow_color,                    # Primary from preset
        (255, 180, 80),               # Cyan
        (200, 255, 150),              # Light cyan-green
        (255, 220, 180),              # Pale cyan
    ]

    # =========================================================================
    # COLLECT MOTION POINTS (vectorized). The flow field is only ever read on
    # the sample grid, so sample flow_small directly instead of upscaling the
    # whole field back to frame resolution first.
    # =========================================================================
    ys = np.arange(sample_step, h - sample_step, sample_step)
    xs = np.arange(sample_step, w - sample_step, sample_step)
    GX, GY = np.meshgrid(xs, ys)
    sy = np.clip((GY * sf).astype(np.int32), 0, flow_small.shape[0] - 1)
    sx = np.clip((GX * sf).astype(np.int32), 0, flow_small.shape[1] - 1)
    gdx = flow_small[sy, sx, 0] / sf
    gdy = flow_small[sy, sx, 1] / sf
    gmag = np.sqrt(gdx * gdx + gdy * gdy)
    moving = gmag >= min_flow_mag

    motion_points = [
        (int(x), int(y), float(dx), float(dy), float(mag),
         flow_colors[i % len(flow_colors)])
        for i, (x, y, dx, dy, mag) in enumerate(zip(
            GX[moving], GY[moving], gdx[moving], gdy[moving], gmag[moving]))
    ]
    
    # =========================================================================
    # DRAW FLOW LINES onto a uint8 ink layer (cv2 drawing on float images is
    # much slower), then merge into the float trail canvas below.
    # =========================================================================
    ink = np.zeros((h, w, 3), dtype=np.uint8)
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
        cv2.polylines(ink, [pts], False, draw_color, thickness, cv2.LINE_AA)

        # Glowing head at end point for strong motion
        if mag > min_flow_mag * 1.5:
            cv2.circle(ink, (x2, y2), 3, (255, 255, 255), -1, cv2.LINE_AA)
    
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
                    cv2.line(ink, (x1, y1), (x2, y2), conn_color, 1, cv2.LINE_AA)

    # Merge fresh ink into the persistent float canvas (brightest wins, same
    # semantics as drawing directly but without float-draw cost).
    np.maximum(_motion_trace_trail_canvas, ink.astype(np.float32),
               out=_motion_trace_trail_canvas)

    # =========================================================================
    # COMPOSITE: trail canvas over original frame
    # =========================================================================
    canvas8 = np.clip(_motion_trace_trail_canvas, 0, 255).astype(np.uint8)

    # Add subtle glow to trail canvas
    glow = cv2.GaussianBlur(canvas8, (7, 7), 0)
    trail_with_glow = cv2.addWeighted(canvas8, 1.0, glow, 0.4, 0)

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
    
    # Glitch arrives in short 2-3 frame bursts on a hashed schedule with
    # eased strength, instead of a single-frame tick exactly every N frames
    # (the old metronome read as a rhythmic pop).
    if glitch_freq > 0:
        window = frame_idx // glitch_freq
        burst = ((window * 2654435761) % 97) < 40      # ~40% of windows glitch
        pos = frame_idx % glitch_freq
        burst_len = 2 + (window % 2)
        if burst and pos < burst_len:
            ease = np.sin((pos + 1) / (burst_len + 1) * np.pi)  # in-out
            output = apply_glitch(output, glitch_strength * (0.6 + 0.8 * ease), window)
    
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
_codenet_ids: np.ndarray | None = None
_codenet_next_id: int = 0


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
    
    global _codenet_pts, _codenet_prev_gray, _codenet_ids, _codenet_next_id

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better feature detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Stabilized nodes: track existing corners with optical flow every frame.
    # The old version threw the whole set away every 12 frames, which made the
    # entire mesh (and every label) pop and renumber on a visible beat. Now
    # tracked points persist with stable ids and fresh corners are only merged
    # into empty regions when the set runs thin.
    if (_codenet_pts is not None and _codenet_prev_gray is not None
            and _codenet_prev_gray.shape == gray.shape and len(_codenet_pts) >= 3):
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            _codenet_prev_gray, gray, _codenet_pts.reshape(-1, 1, 2), None,
            winSize=(21, 21), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
        )
        if tracked is not None:
            ok = status.flatten() == 1
            pts = tracked.reshape(-1, 2)[ok]
            ids = _codenet_ids[ok] if _codenet_ids is not None else None
            inb = ((pts[:, 0] > 1) & (pts[:, 0] < w - 2)
                   & (pts[:, 1] > 1) & (pts[:, 1] < h - 2))
            _codenet_pts = pts[inb]
            _codenet_ids = ids[inb] if ids is not None else None
        else:
            _codenet_pts = None
            _codenet_ids = None
    elif _codenet_prev_gray is not None and _codenet_prev_gray.shape != gray.shape:
        _codenet_pts = None
        _codenet_ids = None

    if _codenet_pts is None or len(_codenet_pts) < max(8, int(max_points * 0.7)):
        corners = cv2.goodFeaturesToTrack(
            enhanced,
            maxCorners=max_points,
            qualityLevel=0.02,
            minDistance=20,
            blockSize=7,
        )
        fresh = (corners.reshape(-1, 2).astype(np.float32)
                 if corners is not None else np.empty((0, 2), np.float32))
        if _codenet_pts is None or len(_codenet_pts) == 0:
            _codenet_pts = fresh
            _codenet_ids = np.arange(_codenet_next_id,
                                     _codenet_next_id + len(fresh), dtype=np.int64)
            _codenet_next_id += len(fresh)
        elif len(fresh):
            # Merge only corners that don't sit on an existing node (coarse
            # occupancy grid ~ the detector's minDistance).
            occ = np.zeros((h // 16 + 2, w // 16 + 2), dtype=bool)
            occ[(_codenet_pts[:, 1] // 16).astype(int),
                (_codenet_pts[:, 0] // 16).astype(int)] = True
            keep = ~occ[(fresh[:, 1] // 16).astype(int),
                        (fresh[:, 0] // 16).astype(int)]
            new_pts = fresh[keep][: max_points - len(_codenet_pts)]
            if len(new_pts):
                _codenet_pts = np.concatenate([_codenet_pts, new_pts])
                new_ids = np.arange(_codenet_next_id,
                                    _codenet_next_id + len(new_pts), dtype=np.int64)
                _codenet_next_id += len(new_pts)
                _codenet_ids = np.concatenate([_codenet_ids, new_ids])

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
    for pt, pid in zip(points, _codenet_ids):
        x, y = pt
        if 0 < x < w - 1 and 0 < y < h - 1:
            subdiv.insert((float(x), float(y)))
            valid_points.append((int(x), int(y), int(pid)))
    
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
    for (px, py, _pid) in valid_points:
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
    for (px, py, pid) in valid_points:
        label = f"codecore {pid % 1000 + 1}"
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

_binary_bloom_cache: dict = {}


def draw_binary_bloom(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Binary Bloom: 0/1 digits inside the subject silhouette on a solid deep-blue
    field. Edge cells are brighter, larger and denser than interior cells so
    the silhouette pops.

    Vectorized glyph tiles; each cell flips its digit on its own 5-12 frame
    cadence (hash-staggered) instead of the whole field re-rolling at once.
    """
    h, w = frame.shape[:2]

    bg_color = preset.get("bg_color", (160, 40, 0))          # Deep blue BGR
    cell = max(8, int(preset.get("bloom_cell", 12)))

    # =========================================================================
    # SUBJECT MASK: prefer real person segmentation, fall back to contours.
    # =========================================================================
    seg = get_person_mask(frame)
    if seg is not None and np.count_nonzero(seg) > h * w * 0.02:
        _, subject_mask = cv2.threshold(seg, 110, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

    # Silhouette edge band (for emphasis).
    edge_mask = cv2.Canny(subject_mask, 50, 150)
    edge_mask = cv2.dilate(edge_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                           iterations=2)

    # =========================================================================
    # TILE COMPOSE: one grid; interior cells dim/small, edge cells bright/big
    # =========================================================================
    grid_w, grid_h = max(1, w // cell), max(1, h // cell)

    key = ("tiles", cell)
    tiles = _binary_bloom_cache.get(key)
    if tiles is None:
        # 0 blank | 1-2: '0','1' interior | 3-4: '0','1' edge (bigger, brighter)
        tiles = np.zeros((5, cell, cell), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs_int = cell / 30.0
        fs_edge = cell / 24.0
        for i, (c, fs, val) in enumerate([
            ("0", fs_int, 185), ("1", fs_int, 185),
            ("0", fs_edge, 255), ("1", fs_edge, 255),
        ]):
            # Center the glyph in the tile (measured, so nothing clips).
            (tw_, th_), _ = cv2.getTextSize(c, font, fs, 1)
            org = (max(0, (cell - tw_) // 2), cell - max(1, (cell - th_) // 2))
            cv2.putText(tiles[i + 1], c, org, font, fs, int(val), 1, cv2.LINE_AA)
        _binary_bloom_cache[key] = tiles

    inside = cv2.resize(subject_mask, (grid_w, grid_h), interpolation=cv2.INTER_AREA) > 100
    on_edge = cv2.resize(edge_mask, (grid_w, grid_h), interpolation=cv2.INTER_AREA) > 40

    rows = np.arange(grid_h)[:, None]
    cols = np.arange(grid_w)[None, :]
    cell_hash = (cols * 131 + rows * 71) * 2654435761 % 2**32
    period = 5 + (cell_hash >> 4) % 8
    bit = (((cell_hash >> 8) + frame_idx // period) % 2).astype(np.int32)

    # Interior digits are sparser: drop ~1/3 of interior cells (stable choice).
    interior_keep = ((cell_hash >> 12) % 3) != 0

    idx = np.zeros((grid_h, grid_w), dtype=np.int32)
    idx[inside & interior_keep] = (1 + bit)[inside & interior_keep]
    idx[on_edge] = (3 + bit)[on_edge]

    alpha = tiles[idx].transpose(0, 2, 1, 3).reshape(grid_h * cell, grid_w * cell)

    output = np.full((h, w, 3), bg_color, dtype=np.uint8)
    ah, aw = min(h, alpha.shape[0]), min(w, alpha.shape[1])
    layer = alpha[:ah, :aw].astype(np.float32) / 255.0

    white = np.array((255, 255, 255), dtype=np.float32)
    region = output[:ah, :aw].astype(np.float32)
    output[:ah, :aw] = (region * (1 - layer[:, :, None])
                        + white[None, None, :] * layer[:, :, None]).astype(np.uint8)

    # Soft bloom on the bright edge band so the silhouette glows.
    edge_glow_src = np.zeros((h, w), dtype=np.uint8)
    edge_full = cv2.resize(on_edge.astype(np.uint8) * 255, (aw, ah),
                           interpolation=cv2.INTER_NEAREST)
    edge_glow_src[:ah, :aw] = (alpha[:ah, :aw] * (edge_full > 0)).astype(np.uint8)
    glow = cv2.GaussianBlur(edge_glow_src, (0, 0), 4.0)
    output = cv2.add(output, cv2.cvtColor((glow * 0.6).astype(np.uint8), cv2.COLOR_GRAY2BGR))

    return output


# =============================================================================
# SIGNAL FEEDBACK (CRT/VHS style with noise warping and feedback trails)
# =============================================================================

# Persistent state for signal feedback effect
_signal_feedback_buffer: np.ndarray | None = None
_signal_feedback_noise: np.ndarray | None = None  # (2, nh, nw) low-res x/y fields
_signal_feedback_grid: dict = {}  # cached per-(h,w) coordinate grid + vignette

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
    # Noise lives at 1/8 resolution: blurring per-pixel white noise at full res
    # was expensive and produced busy mid-scale wobble; a smoothed low-res
    # field upscaled to frame size gives the broad liquid warp this effect
    # wants. Two independent fields (x and y) so the displacement can swirl in
    # any direction instead of shearing along the diagonal only.
    nh, nw = max(8, h // 8), max(8, w // 8)
    if (_signal_feedback_buffer is None
            or _signal_feedback_buffer.shape[:2] != (h, w)
            or _signal_feedback_noise is None
            or _signal_feedback_noise.shape != (2, nh, nw)):
        _signal_feedback_buffer = current_float.copy()
        _signal_feedback_noise = np.random.rand(2, nh, nw).astype(np.float32)
        return frame  # Return original on first frame

    # =========================================================================
    # STEP 1: Generate noise-based warp map
    # =========================================================================
    # Slowly evolve the noise fields
    new_noise = np.random.rand(2, nh, nw).astype(np.float32)
    _signal_feedback_noise *= (1 - noise_scale)
    _signal_feedback_noise += new_noise * noise_scale

    smooth_x = cv2.GaussianBlur(_signal_feedback_noise[0], (0, 0), 3.0)
    smooth_y = cv2.GaussianBlur(_signal_feedback_noise[1], (0, 0), 3.0)
    offset_x = cv2.resize(smooth_x, (w, h), interpolation=cv2.INTER_LINEAR)
    offset_y = cv2.resize(smooth_y, (w, h), interpolation=cv2.INTER_LINEAR)

    # Cached base coordinate grid (+ vignette, used in STEP 5)
    cached = _signal_feedback_grid.get((h, w))
    if cached is None:
        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        cy, cx = h // 2, w // 2
        dist_from_center = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = np.clip(1.0 - (dist_from_center / max_dist) * 0.3, 0.7, 1.0)
        cached = (grid_x, grid_y, vignette.astype(np.float32)[..., np.newaxis])
        _signal_feedback_grid[(h, w)] = cached
    grid_x, grid_y, vignette = cached

    map_x = grid_x + (offset_x - 0.5) * distortion_amp
    map_y = grid_y + (offset_y - 0.5) * distortion_amp
    
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
    # STEP 5: Subtle vignette for CRT feel (precomputed in the grid cache)
    # =========================================================================
    result = (result.astype(np.float32) * vignette).astype(np.uint8)

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

_glyph_trace_tiles: np.ndarray | None = None


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
    global _glyph_trace_tiles

    h, w = frame.shape[:2]

    # 1. Colors
    # Ink (#1F1E1D) -> BGR (29, 30, 31)
    # Paper (#FAF9F5) -> BGR (245, 249, 250)
    ink_color = (29, 30, 31)
    paper_color = (245, 249, 250)

    # 2. Pre-render crisp ASCII tiles (once - constants, so no cache key)
    # Using a 6x10 block gives a nice tall terminal look
    tw, th = 6, 10
    ascii_chars = " .:-=+*#%@"
    num_chars = len(ascii_chars)

    if _glyph_trace_tiles is None:
        # Tile bank (num_chars, height, width, 3)
        tiles = np.full((num_chars, th, tw, 3), paper_color, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_PLAIN
        for i, char in enumerate(ascii_chars):
            if char == ' ':
                continue
            # FONT_HERSHEY_PLAIN at scale 0.8 is approx 8 pixels tall.
            # (0, 8) is the bottom-left baseline for the text
            # cv2.LINE_4 avoids anti-aliasing blur for maximum crispness
            cv2.putText(tiles[i], char, (0, 8), font, 0.8, ink_color, 1, cv2.LINE_4)
        _glyph_trace_tiles = tiles
    tiles = _glyph_trace_tiles

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
    global _codenet_pts, _codenet_prev_gray, _codenet_ids, _codenet_next_id
    global _pc_prev_small, _pc_energy
    global _blob_next_id
    global _crystal_pts, _crystal_prev_gray
    global _neon_prev_edges

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
    _codenet_ids = None
    _codenet_next_id = 0
    _pc_prev_small = None
    _pc_energy = None
    _blob_tracks.clear()
    _blob_next_id = 0
    _crystal_pts = None
    _crystal_prev_gray = None
    _neon_prev_edges = None


# =============================================================================
# ASCII CORE (high-detail white ASCII on black)
# =============================================================================

_ascii_core_cache: dict = {}


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

    # Pre-render each glyph to a white-on-black tile (cached per cell/ramp).
    tiles = _ascii_core_cache.get((cell, ramp))
    if tiles is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = cell / 22.0
        tiles = np.zeros((n, cell, cell, 3), dtype=np.uint8)
        for i, ch in enumerate(ramp):
            if ch != " ":
                cv2.putText(tiles[i], ch, (0, cell - 1), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        _ascii_core_cache[(cell, ramp)] = tiles

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

    # 4. Melting lower edge: rows near the bottom copy from progressively
    #    higher rows, creating the downward "signal melt" smear (one gather).
    melt_start = int(h * 0.78)
    if melt_start < h:
        ys = np.arange(melt_start, h)
        amt = ((ys - melt_start) / max(1, h - melt_start) * 18).astype(np.int32)
        static[ys] = static[np.maximum(melt_start, ys - amt)]

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

_crystal_pts: np.ndarray | None = None
_crystal_prev_gray: np.ndarray | None = None


def draw_crystallize(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Crystallize: shatter the frame into a low-poly mosaic of flat-shaded
    triangles. Feature points concentrate detail on the subject while a grid
    guarantees full coverage.

    The seed points are TRACKED with optical flow frame-to-frame (with a
    periodic top-up of fresh corners), so the triangulation deforms smoothly
    with the motion instead of re-randomizing - the mosaic flows like faceted
    glass rather than strobing. Facet colors are sampled from a pre-blurred
    frame (one op) instead of averaging a patch per triangle.
    """
    global _crystal_pts, _crystal_prev_gray

    h, w = frame.shape[:2]
    cells = int(preset.get("cells", 600))
    grid_step = int(preset.get("grid_step", max(36, (w + h) // 34)))
    facet_edges = bool(preset.get("facet_edges", True))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Temporally coherent seed points ---
    pts = None
    if (_crystal_pts is not None and _crystal_prev_gray is not None
            and _crystal_prev_gray.shape == gray.shape and len(_crystal_pts) >= 8):
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            _crystal_prev_gray, gray, _crystal_pts.reshape(-1, 1, 2), None,
            winSize=(21, 21), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
        )
        if tracked is not None:
            ok = status.flatten() == 1
            pts = tracked.reshape(-1, 2)[ok]
            # Drop points that drifted out of frame.
            inb = ((pts[:, 0] > 1) & (pts[:, 0] < w - 2)
                   & (pts[:, 1] > 1) & (pts[:, 1] < h - 2))
            pts = pts[inb]

    # Top up with fresh corners when thin (or on a slow cadence so new detail
    # gets facets without disturbing existing ones).
    if pts is None or len(pts) < cells * 0.6 or frame_idx % 45 == 0:
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=cells, qualityLevel=0.01, minDistance=8)
        fresh = (corners.reshape(-1, 2).astype(np.float32)
                 if corners is not None else np.empty((0, 2), np.float32))
        if pts is None or len(pts) == 0:
            pts = fresh
        elif len(fresh):
            # Keep tracked points; add fresh ones only where no point already
            # lives (coarse occupancy grid keeps this O(n)).
            occ = np.zeros((h // 8 + 1, w // 8 + 1), dtype=bool)
            oy = (pts[:, 1] // 8).astype(int)
            ox = (pts[:, 0] // 8).astype(int)
            occ[oy, ox] = True
            fy = (fresh[:, 1] // 8).astype(int)
            fx = (fresh[:, 0] // 8).astype(int)
            new = fresh[~occ[fy, fx]]
            if len(new):
                pts = np.concatenate([pts, new])[: cells]

    _crystal_pts = pts.astype(np.float32) if pts is not None else None
    _crystal_prev_gray = gray

    # --- Triangulate: tracked features + static grid/border for coverage ---
    all_pts: list[tuple[float, float]] = []
    if pts is not None:
        all_pts.extend((float(x), float(y)) for x, y in pts)
    gx = list(range(0, w, grid_step)) + [w - 1]
    gy = list(range(0, h, grid_step)) + [h - 1]
    for y in gy:
        for x in gx:
            all_pts.append((float(x), float(y)))

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for (x, y) in all_pts:
        if 0 <= x < w and 0 <= y < h:
            subdiv.insert((float(x), float(y)))

    # Facet color = pre-blurred frame sampled at the centroid (stable, cheap).
    smooth = cv2.blur(frame, (7, 7))

    output = np.zeros((h, w, 3), dtype=np.uint8)
    tris = subdiv.getTriangleList()
    if len(tris):
        tris = tris.reshape(-1, 3, 2)
        # Keep triangles fully inside the frame.
        keep = ((tris[:, :, 0] >= 0).all(axis=1) & (tris[:, :, 0] <= w - 1).all(axis=1)
                & (tris[:, :, 1] >= 0).all(axis=1) & (tris[:, :, 1] <= h - 1).all(axis=1))
        tris = tris[keep]
        cxs = np.clip(tris[:, :, 0].mean(axis=1), 0, w - 1).astype(np.int32)
        cys = np.clip(tris[:, :, 1].mean(axis=1), 0, h - 1).astype(np.int32)
        cols = smooth[cys, cxs]  # (n, 3) uint8
        edge_cols = (cols.astype(np.float32) * 0.65).astype(np.uint8)
        polys = tris.astype(np.int32)
        for poly, col, ecol in zip(polys, cols, edge_cols):
            c = (int(col[0]), int(col[1]), int(col[2]))
            cv2.fillConvexPoly(output, poly, c, cv2.LINE_AA)
            if facet_edges:
                cv2.polylines(output, [poly], True,
                              (int(ecol[0]), int(ecol[1]), int(ecol[2])), 1, cv2.LINE_AA)

    return output


# =============================================================================
# HALFTONE / BLACKTONE (newsprint dot screens, positive and negative)
# =============================================================================

_halftone_cache: dict = {}


def _halftone_pattern(h: int, w: int, dot: int) -> np.ndarray:
    """
    Cached per-pixel distance-to-dot-center pattern on a classic 45-degree
    printing screen (rotated grid reads as authentic newsprint rather than a
    computer grid). 0 at cell centers -> ~1 at cell corners.
    """
    key = (h, w, dot)
    patt = _halftone_cache.get(key)
    if patt is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Rotate coordinates 45 degrees, then tile into cells.
        s = np.float32(1.0 / np.sqrt(2.0))
        u = (xx + yy) * s
        v = (xx - yy) * s
        cu = (u % dot) - (dot - 1) / 2.0
        cv_ = (v % dot) - (dot - 1) / 2.0
        patt = (np.sqrt(cu * cu + cv_ * cv_) / ((dot / 2.0) * np.sqrt(2.0))).astype(np.float32)
        _halftone_cache[key] = patt
    return patt


def _halftone_radius(frame: np.ndarray, dot: int, contrast: float, gamma: float) -> np.ndarray:
    """Per-pixel dot radius (0..1) from block-averaged, contrast-punched luma."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gw, gh = max(1, w // dot), max(1, h // dot)
    small = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
    lum = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
    lum = np.clip((lum - 0.5) * contrast + 0.5, 0.0, 1.0)
    return np.power(np.clip(1.0 - lum, 0.0, 1.0), gamma)


def draw_halftone(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Halftone: classic black-and-white newsprint. Black dots on a white page on
    a 45-degree screen; each dot grows as the source gets darker.
    """
    h, w = frame.shape[:2]
    dot = max(4, int(preset.get("dot_spacing", 8)))
    gamma = float(preset.get("dot_gamma", 0.9))
    contrast = float(preset.get("dot_contrast", 1.25))

    patt = _halftone_pattern(h, w, dot)
    radius = _halftone_radius(frame, dot, contrast, gamma)

    mask = patt <= radius
    output = np.full((h, w, 3), 255, dtype=np.uint8)  # white page
    output[mask] = (0, 0, 0)                            # black ink dots
    return output


def draw_blacktone(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Blacktone: Halftone's photo negative - emissive white dots on black, same
    dot-size logic, finished with a soft glow.
    """
    h, w = frame.shape[:2]
    dot = max(4, int(preset.get("dot_spacing", 8)))
    gamma = float(preset.get("dot_gamma", 0.9))
    contrast = float(preset.get("dot_contrast", 1.25))

    patt = _halftone_pattern(h, w, dot)
    radius = _halftone_radius(frame, dot, contrast, gamma)

    mask = patt <= radius
    canvas = np.zeros((h, w, 3), dtype=np.uint8)   # black page
    canvas[mask] = (255, 255, 255)                  # white ink dots

    glow = cv2.GaussianBlur(canvas, (0, 0), max(1.0, dot * 0.35))
    out = cv2.add(canvas, (glow.astype(np.float32) * 0.5).astype(np.uint8))
    return out


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

    # float32 canvas: uint8 decay quantized the fade, so long streaks died in
    # visible brightness steps instead of melting away smoothly.
    if _light_canvas is None or _light_canvas.shape[:2] != (h, w):
        _light_canvas = np.zeros((h, w, 3), dtype=np.float32)

    # Decay the accumulated light (older streaks fade out).
    _light_canvas *= decay

    # Contribution: only the brightest parts of the current frame (adaptive to
    # exposure via a percentile) so it paints trails instead of flooding.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = max(floor, int(np.percentile(gray, pct)))
    bright = gray > thresh
    contrib = np.zeros((h, w, 3), dtype=np.float32)
    contrib[bright] = frame[bright]
    contrib = np.clip(contrib * boost, 0, 255)

    # Keep the brightest of (decayed history, new light) -> persistent trails.
    np.maximum(_light_canvas, contrib, out=_light_canvas)

    # Bloom and composite over a dark version of the scene for context.
    canvas8 = _light_canvas.astype(np.uint8)
    glow = cv2.GaussianBlur(canvas8, (0, 0), 6)
    trails = cv2.addWeighted(canvas8, 1.0, glow, 0.9, 0)
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
_neon_prev_edges: np.ndarray | None = None


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

    global _neon_prev_edges

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 7, 60, 60)
    edges = cv2.bitwise_or(cv2.Canny(smooth, 40, 110), cv2.Canny(smooth, 80, 180))
    if thickness > 1:
        edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8))

    # Tube persistence: carry a fading echo of previous edges so lines that
    # Canny drops for a frame decay smoothly instead of blinking off, like a
    # real neon tube cooling down.
    if _neon_prev_edges is not None and _neon_prev_edges.shape == edges.shape:
        edges = cv2.max(edges, (_neon_prev_edges * 0.58).astype(np.uint8))
    _neon_prev_edges = edges.copy()

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
_pc_prev_small: np.ndarray | None = None
_pc_energy: np.ndarray | None = None


def draw_point_cloud(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
    audio_level: float = 0.0,
) -> np.ndarray:
    """
    Point Cloud: a TouchDesigner-style 3D point-cloud scan in black & white.

    Detail comes from POINT DENSITY (stable, hash-based) tracking local
    brightness, so highlights pack densely and shadows stay empty. The cloud has
    life: a motion-fed energy field pushes points like RIPPLES IN WATER where the
    subject moves, a slow buoyancy makes the whole cloud float in gravity, and a
    gentle yaw gives it volume. Strongly audio-reactive: loudness expands depth,
    blasts points radially outward on the beat, supercharges the ripples, packs
    in more points and brightens. Bright white dots on black.
    """
    global _pc_prev_small, _pc_energy

    h, w = frame.shape[:2]
    hstep = max(2, int(preset.get("pc_step", 2)))
    vstep = max(2, int(preset.get("pc_row", 3)))
    min_bright = int(preset.get("pc_min_bright", 22))
    depth_scale = float(preset.get("pc_depth", 85.0))
    pop = float(preset.get("pc_pop", 8.0))
    noise_amp = float(preset.get("pc_noise", 1.6))
    yaw_amp = float(preset.get("pc_yaw", 0.28))
    density_gamma = float(preset.get("pc_density_gamma", 1.25))
    dot = int(preset.get("pc_dot", 2))
    float_amp = float(preset.get("pc_float", 6.0))     # buoyancy / floating
    ripple_amp = float(preset.get("pc_ripple", 26.0))  # water-ripple displacement

    # Punchy audio response (emphasize peaks).
    a = float(np.clip(audio_level, 0.0, 1.0)) ** 0.6

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Cached grid + stable per-cell hash (deterministic density => readable,
    # stable detail that tracks the subject instead of random strobing).
    key = (h, w, hstep, vstep)
    cached = _point_cloud_cache.get(key)
    if cached is None:
        gxs = np.arange(0, w, hstep)
        gys = np.arange(0, h, vstep)
        GX, GY = np.meshgrid(gxs, gys)
        phase = (GX * 12.9 + GY * 78.2).astype(np.float32)
        khash = (((GX * 131 + GY * 71) % 1000) / 1000.0).astype(np.float32)
        cached = (GX.astype(np.float32), GY.astype(np.float32), phase, khash,
                  GX.astype(np.int32), GY.astype(np.int32))
        _point_cloud_cache[key] = cached
    GXf, GYf, phase, khash, GXi, GYi = cached

    bright = gray[GYi, GXi].astype(np.float32)
    bnorm = bright / 255.0

    # Subject isolation: real person mask when available, else brightness.
    seg = get_person_mask(frame)
    if seg is not None:
        subject = seg[GYi, GXi] > 110
    else:
        subject = bright > min_bright

    # Density follows brightness (stable hash, so the face is consistent/detailed
    # and only changes as the subject moves). Audio packs in extra points.
    prob = np.power(np.clip(bnorm, 0.0, 1.0), density_gamma)
    prob = np.clip(prob * (1.0 + a * 0.6), 0.0, 1.0)
    mask = subject & (bright > min_bright) & (khash < prob)
    if not mask.any():
        return np.zeros((h, w, 3), dtype=np.uint8)

    # --- WATER-RIPPLE ENERGY FIELD (fed by motion, diffuses outward) ---
    ef_w, ef_h = max(8, w // 8), max(8, h // 8)
    gsmall = cv2.resize(gray, (ef_w, ef_h), interpolation=cv2.INTER_AREA)
    if (_pc_prev_small is None or _pc_prev_small.shape != gsmall.shape
            or _pc_energy is None or _pc_energy.shape != gsmall.shape):
        _pc_prev_small = gsmall.copy()
        _pc_energy = np.zeros((ef_h, ef_w), dtype=np.float32)
    motion = cv2.absdiff(gsmall, _pc_prev_small).astype(np.float32) / 255.0
    _pc_prev_small = gsmall.copy()
    # Accumulate + diffuse: impulses where the subject moves spread like ripples.
    _pc_energy = _pc_energy * 0.86 + motion * 2.2
    _pc_energy = cv2.GaussianBlur(_pc_energy, (0, 0), 2.0)
    np.clip(_pc_energy, 0.0, 4.0, out=_pc_energy)
    # Ripple displacement = gradient of the energy field (push from wave fronts),
    # oscillating in time so the surface undulates.
    gxE = cv2.Sobel(_pc_energy, cv2.CV_32F, 1, 0, ksize=3)
    gyE = cv2.Sobel(_pc_energy, cv2.CV_32F, 0, 1, ksize=3)
    fy = np.clip(GYi // 8, 0, ef_h - 1)
    fx = np.clip(GXi // 8, 0, ef_w - 1)
    e_here = _pc_energy[fy, fx]
    wave = np.sin(e_here * 6.0 - frame_idx * 0.4)
    rip_scale = ripple_amp * (1.0 + a * 2.5) * (0.5 + 0.5 * wave)
    rip_x = gxE[fy, fx] * rip_scale
    rip_y = gyE[fy, fx] * rip_scale

    cx, cy = w / 2.0, h / 2.0
    theta = np.sin(frame_idx * 0.025) * (yaw_amp + a * 0.2)
    ct, st = np.cos(theta), np.sin(theta)

    # Brightness -> depth; audio inflates the whole volume.
    depth = (bnorm - 0.45) * depth_scale * (1.0 + a * 1.6)
    X = GXf - cx
    sx = cx + X * ct + depth * st
    sy = GYf - bnorm * pop

    # Floating gravity: a slow buoyant swell makes the cloud drift like it's
    # suspended in fluid (rows bob out of phase).
    sy = sy + np.sin(frame_idx * 0.05 + GXf * 0.012) * float_amp
    sx = sx + np.cos(frame_idx * 0.04 + GYf * 0.010) * (float_amp * 0.4)

    # Noise shimmer (audio adds life) + the ripple push.
    jit = noise_amp * (1.0 + a * 1.5)
    sx = sx + np.sin(frame_idx * 0.18 + phase) * jit + rip_x
    sy = sy + np.cos(frame_idx * 0.15 + phase * 1.3) * jit + rip_y

    # Audio BEAT BLAST: shove every point radially outward from the centre on
    # loud moments, so the cloud visibly bursts with the music.
    if a > 0.02:
        dx, dy = GXf - cx, GYf - cy
        dist = np.sqrt(dx * dx + dy * dy) + 1e-3
        blast = a * 55.0
        sx = sx + (dx / dist) * blast
        sy = sy + (dy / dist) * blast

    # Bright white, lifted by source brightness; audio boosts brightness hard.
    inten = np.clip(170 + bnorm * 85 + a * 60, 0, 255)

    xs = sx[mask].astype(np.int32)
    ys = sy[mask].astype(np.int32)
    vals = inten[mask].astype(np.float32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys, vals = xs[valid], ys[valid], vals[valid]

    # Scatter points (keep the brightest where they overlap).
    canvas = np.zeros((h * w,), dtype=np.float32)
    np.maximum.at(canvas, ys * w + xs, vals)
    canvas = np.clip(canvas, 0, 255).reshape(h, w).astype(np.uint8)

    # Clean SQUARE dots (RECT kernel - no plus/cross artifact).
    if dot >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dot, dot))
        canvas = cv2.dilate(canvas, k)

    out = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    # Crisp near-glow on each dot...
    near = cv2.GaussianBlur(out, (0, 0), 1.4 + a * 1.5)
    out = cv2.add(out, (near.astype(np.float32) * (0.55 + a * 0.45)).astype(np.uint8))
    # ...plus a wide soft AURA / halo that breathes around the whole cloud so it
    # glows like a suspended energy field.
    aura = cv2.GaussianBlur(out, (0, 0), 13.0 + a * 8.0)
    out = cv2.add(out, (aura.astype(np.float32) * (0.30 + a * 0.35)).astype(np.uint8))
    return out


# =============================================================================
# CURSOR CLOUD (subject built from tiny white pixel cursors on black)
# =============================================================================

_cursor_cache: dict = {}


def _cursor_sprite_mask(cell: int) -> np.ndarray:
    """A small SOLID white pixel-cursor (classic arrow) mask sized to `cell`."""
    cached = _cursor_cache.get(("mask", cell))
    if cached is not None:
        return cached
    spr = np.zeros((cell, cell), dtype=np.uint8)
    # Classic filled arrow silhouette in a normalized box (reads even when tiny).
    norm = [
        (0.05, 0.02), (0.05, 0.82), (0.26, 0.63), (0.40, 0.98),
        (0.55, 0.90), (0.41, 0.57), (0.70, 0.55),
    ]
    pts = np.array([[int(x * (cell - 1)), int(y * (cell - 1))] for x, y in norm], dtype=np.int32)
    cv2.fillPoly(spr, [pts], 255, cv2.LINE_AA)
    _cursor_cache[("mask", cell)] = spr
    return spr


def draw_cursor_cloud(
    frame: np.ndarray,
    preset: dict[str, Any],
    colors: dict,
    frame_idx: int = 0,
    audio_level: float = 0.0,
) -> np.ndarray:
    """
    Cursor Cloud: the subject built from hundreds of tiny white pixel cursors on
    black - like Point Cloud / Halftone but every dot is a little arrow cursor.
    Density follows brightness; brighter cells get brighter cursors. Fully
    vectorized via a glyph-tile stack, so it stays dense and fast. Audio brightens
    and packs in more cursors.
    """
    h, w = frame.shape[:2]
    cell = max(6, int(preset.get("cur_cell", 11)))
    gamma = float(preset.get("cur_density_gamma", 1.25))
    min_bright = int(preset.get("cur_min_bright", 22))
    a = float(np.clip(audio_level, 0.0, 1.0)) ** 0.6

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    grid_h, grid_w = max(1, h // cell), max(1, w // cell)
    cell_lum = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    seg = get_person_mask(frame)
    if seg is not None:
        subject = cv2.resize(seg, (grid_w, grid_h), interpolation=cv2.INTER_AREA) > 110
    else:
        subject = cell_lum * 255 > min_bright

    # Pre-render a brightness-stack of cursor tiles (index 0 = blank), cached
    # per cell size.
    levels = 6
    tiles = _cursor_cache.get(("tiles", cell))
    if tiles is None:
        mask = _cursor_sprite_mask(cell).astype(np.float32) / 255.0
        tiles = np.zeros((levels + 1, cell, cell, 3), dtype=np.uint8)
        for L in range(1, levels + 1):
            val = 150 + int(105 * (L - 1) / (levels - 1))
            tiles[L] = (mask[:, :, None] * val).astype(np.uint8)
        _cursor_cache[("tiles", cell)] = tiles

    # Density follows brightness (stable hash, no strobe); audio packs in more.
    gy, gx = np.indices((grid_h, grid_w))
    khash = ((gx * 131 + gy * 71) % 1000) / 1000.0
    prob = np.clip(np.power(np.clip(cell_lum, 0, 1), gamma) * (1.0 + a * 0.5), 0, 1)
    keep = subject & (cell_lum * 255 > min_bright) & (khash < prob)

    lvl = 1 + np.clip((cell_lum * (levels - 1) + a * 1.5), 0, levels - 1).astype(np.int32)
    idx = np.where(keep, lvl, 0)

    mapped = tiles[idx]  # (grid_h, grid_w, cell, cell, 3)
    out_grid = mapped.transpose(0, 2, 1, 3, 4).reshape(grid_h * cell, grid_w * cell, 3)

    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:out_grid.shape[0], :out_grid.shape[1]] = out_grid

    glow = cv2.GaussianBlur(out, (0, 0), 1.4 + a * 1.5)
    out = cv2.add(out, (glow.astype(np.float32) * (0.45 + a * 0.4)).astype(np.uint8))
    return out
