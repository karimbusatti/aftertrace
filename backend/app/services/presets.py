"""
Preset definitions for Aftertrace visual effects.

Each preset is a complete "vibe" configuration that controls:
- How points spawn and live
- How they're drawn (shape, size, trails)
- How they connect (grid lines)
- Color palette and effects

To add a new preset:
1. Add an entry to PRESETS dict
2. That's it - the pipeline reads from here automatically
"""

from typing import Any


# =============================================================================
# COLOR PALETTES
# =============================================================================

COLOR_PALETTES = {
    "clean_white": {
        "point": (255, 255, 255),
        "trail": (200, 200, 200),
        "line": (150, 150, 150),
        "glow": (255, 255, 255),
        "background_tint": None,
    },
    "neon_grid": {
        "point": (255, 255, 255),
        "trail": (200, 200, 200),
        "line": (100, 100, 100),
        "glow": (255, 255, 255),
        "background_tint": None,
    },
    "soft_pastel": {
        "point": (255, 180, 220),
        "trail": (180, 140, 200),
        "line": (120, 100, 140),
        "glow": (200, 150, 255),
        "background_tint": (20, 10, 30),
    },
    "alert": {
        "point": (0, 255, 80),
        "trail": (0, 200, 60),
        "line": (0, 150, 50),
        "glow": (0, 255, 100),
        "background_tint": None,
    },
    "cctv_green": {
        "point": (0, 255, 120),
        "trail": (0, 220, 100),
        "line": (0, 180, 80),
        "glow": (0, 255, 150),
        "background_tint": (0, 10, 5),
    },
    "thermal": {
        "point": (0, 120, 255),
        "trail": (0, 80, 200),
        "line": (0, 60, 150),
        "glow": (50, 150, 255),
        "background_tint": (10, 5, 0),
    },
    "ember": {
        "point": (80, 180, 255),
        "trail": (60, 140, 230),
        "line": (40, 100, 180),
        "glow": (100, 200, 255),
        "background_tint": (10, 5, 0),
    },
    "ice": {
        "point": (255, 220, 180),
        "trail": (255, 200, 150),
        "line": (200, 180, 130),
        "glow": (255, 230, 200),
        "background_tint": (10, 8, 5),
    },
    "void": {
        "point": (180, 180, 180),
        "trail": (100, 100, 100),
        "line": (50, 50, 50),
        "glow": (150, 150, 150),
        "background_tint": (0, 0, 0),
    },
    "data_green": {
        "point": (80, 255, 80),
        "trail": (60, 200, 60),
        "line": (40, 150, 40),
        "glow": (100, 255, 100),
        "background_tint": None,
    },
    "numeric_gold": {
        "point": (50, 200, 255),
        "trail": (40, 180, 230),
        "line": (30, 140, 180),
        "glow": (60, 220, 255),
        "background_tint": (5, 10, 15),
    },
    "catodic": {
        "point": (255, 255, 255),
        "trail": (200, 200, 200),
        "line": (255, 180, 100),
        "glow": (255, 160, 80),
        "background_tint": (8, 5, 3),
    },
    "biometric": {
        "point": (200, 255, 200),
        "trail": (180, 230, 180),
        "line": (100, 200, 150),
        "glow": (220, 255, 220),
        "background_tint": (5, 10, 5),
    },
    "particle_cream": {
        "point": (230, 235, 255),
        "trail": (200, 210, 240),
        "line": (150, 160, 180),
        "glow": (240, 245, 255),
        "background_tint": None,
    },
}


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

PRESETS: dict[str, dict[str, Any]] = {
    # =========================================================================
    # BLOB TRACKING (TouchDesigner style - NO TRAILS)
    # =========================================================================
    
    "blob_track": {
        "name": "Blob Track",
        "description": "Clean boxes with IDs and connections",
        
        # No point spawning - this effect uses contour detection
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,  # NO TRAILS
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        # Blob tracking - TouchDesigner style with visible video
        "text_mode": "blob_track",
        "blob_threshold": 18,
        "blob_blur": 11,
        "min_blob_area": 100,
        "max_blobs": 200,
        "max_connection_dist": 200,
        "label_scale": 0.35,
        "start_id": 100,
        "bg_alpha": 0.7,  # Show video at 70% brightness
    },
    
    # =========================================================================
    # NUMERIC AURA (Subject isolation - numbers on subject, video background)
    # =========================================================================
    
    "numeric_aura": {
        "name": "Numeric Aura",
        "description": "Subject becomes numbers, video background",
        
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        # Numeric aura - subject isolation with video background
        "text_mode": "numeric_aura",
        "number_density": 0.05,
        "number_font_scale": 0.28,
        "start_number": 19000,
        "max_numbers": 4000,
    },
    
    # =========================================================================
    # THERMAL SCAN (Skepta "Ignorance is Bliss" style)
    # =========================================================================
    
    "thermal_scan": {
        "name": "Thermal Scan",
        "description": "Skepta-style heat vision",
        
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        "text_mode": "thermal_scan",
    },
    
    # =========================================================================
    # PARTICLE SILHOUETTE (bb.dere style - NO TRAILS)
    # =========================================================================
    
    "particle_silhouette": {
        "name": "Particle Cloud",
        "description": "Ethereal point silhouette",
        
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,  # NO TRAILS
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "particle_cream",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        # Particle silhouette - denser, more ethereal
        "text_mode": "particle_silhouette",
        "particle_density": 0.05,
        "brightness_threshold": 25,
        "scatter_range": 2,
        "particle_glow": 0.8,
        "connect_particles": False,
    },
    
    # =========================================================================
    # CONTOUR TRACE (Edge detection - NO TRAILS)
    # =========================================================================
    
    "contour_trace": {
        "name": "Contour",
        "description": "Pure edge visualization",
        
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,  # NO TRAILS
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0.4,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        "thick_edges": False,
        
        "text_mode": "contour_trace",
    },
    
    # =========================================================================
    # FACE SCANNER (Clean minimal - MINIMAL TRAILS)
    # =========================================================================
    
    "face_scanner": {
        "name": "Face Scanner",
        "description": "Clean face detection",
        
        # No point tracking - pure face detection
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.75,  # Show video clearly
        
        # Clean white face detection
        "detect_faces": True,
        "face_boxes": True,
        "face_glow": False,
        "cctv_overlay": False,
        "biometric_data": True,  # Show data readouts
        "biometric_style": "clean",
    },
    
    # =========================================================================
    # MATRIX MODE (Green digital rain)
    # =========================================================================
    
    "matrix_mode": {
        "name": "Matrix Mode",
        "description": "Digital rain over subject",
        
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        "text_mode": "matrix_mode",
    },
    
    # =========================================================================
    # SIGNAL MAP (Data visualization / bit mapping style)
    # =========================================================================
    
    "signal_map": {
        "name": "Signal Map",
        "description": "Data visualization overlay",
        
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        "text_mode": "signal_map",
    },
    
    # =========================================================================
    # FACE MESH (468 points - NO TRAILS)
    # =========================================================================
    
    "face_mesh": {
        "name": "Face Mesh",
        "description": "468-point face landmarks",
        
        "spawn_per_beat": 10,
        "max_points": 40,
        
        "life_frames": 16,
        "trail_length": 0,  # NO TRAILS for cleaner mesh
        "trail_fade": False,
        
        "shape": "circle",
        "point_size": 1,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "ice",
        "blur_radius": 0,
        "glow_intensity": 0.4,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
        
        "detect_faces": True,
        "detect_mesh": True,
        "face_glow": True,
    },
    
    # =========================================================================
    # DATA BODY (Silhouette from text - NO TRAILS)
    # =========================================================================
    
    "data_body": {
        "name": "Data Body",
        "description": "Silhouette rebuilt from code",
        
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,  # NO TRAILS
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 2,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "data_green",
        "blur_radius": 0,
        "glow_intensity": 0.15,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.95,
        
        "text_mode": "data_body",
        "glyph_chars": "ABCDEF0123456789@#$%&",
        "glyph_cell_size": 7,
        "glyph_jitter": 2,
        "min_brightness": 30,
        "glyph_font_scale": 0.30,
        "invert_background": False,
    },
    
    "motion_flow": {
        "name": "Motion Flow",
        "description": "Curved flowing data trails",
        
        "spawn_per_beat": 15,
        "max_points": 100,
        
        "life_frames": 40,
        "trail_length": 35,      # Long trails
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 2,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white", # Will be overridden by effect logic
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        # Motion Flow specific
        "text_mode": "motion_flow",
        "flow_color": (255, 200, 100), # Cyan/Blue-ish (BGR)
        "line_thickness": 1,
        "smoothing": True,
    },
    
    # =========================================================================
    # MOTION TRACE (Clean flowing lines - KEEP TRAILS)
    # =========================================================================
    
    "motion_trace": {
        "name": "Motion Trace",
        "description": "Elegant flowing motion trails",
        
        "spawn_per_beat": 30,
        "max_points": 100,
        
        "life_frames": 28,
        "trail_length": 20,  # TRAILS make sense here
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 2,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0.2,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.9,
        "bg_alpha": 0.06,
    },
    
    # =========================================================================
    # GRID TRACE (Geometric network - SHORT TRAILS)
    # =========================================================================
    
    "grid_trace": {
        "name": "Grid",
        "description": "Geometric network",
        
        "spawn_per_beat": 20,
        "max_points": 80,
        
        "life_frames": 20,
        "trail_length": 8,  # Short trails for grid effect
        "trail_fade": True,
        
        "shape": "square",
        "point_size": 2,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 60,
        "connection_thickness": 1,
        
        "color_mode": "neon_grid",
        "blur_radius": 0,
        "glow_intensity": 0.15,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.9,
    },
    
    # =========================================================================
    # THERMAL (Heat map - TRAILS for heat dissipation look)
    # =========================================================================
    
    "heat_map": {
        "name": "Thermal",
        "description": "Heat signature visualization",
        
        "spawn_per_beat": 35,
        "max_points": 120,
        
        "life_frames": 28,
        "trail_length": 16,  # Trails for heat dissipation
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 10,
        "trace_thickness": 5,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "thermal",
        "blur_radius": 12,
        "glow_intensity": 0.5,
        "scanlines": False,
        "high_contrast_bw": False,
        "use_colormap": True,
        
        "darken_factor": 0.85,
    },
    
    # =========================================================================
    # CATODIC CUBE (CRT depth)
    # =========================================================================
    
    "catodic_cube": {
        "name": "Catodic",
        "description": "CRT screen depth with RGB glitch",
        
        "spawn_per_beat": 10,
        "max_points": 40,
        
        "life_frames": 16,
        "trail_length": 10,
        "trail_fade": True,
        
        "shape": "square",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 60,
        "connection_thickness": 1,
        
        "color_mode": "catodic",
        "blur_radius": 0,
        "glow_intensity": 0.35,
        "scanlines": True,
        "high_contrast_bw": False,
        
        "darken_factor": 0.78,
        
        "cube_mode": True,
        "depth_amount": 0.55,
        "wireframe_layers": 7,
        "wireframe_intensity": 0.85,
        "rgb_offset_px": 6,
        "glitch_frequency": 5,
        "glitch_strength": 0.45,
        "motion_amplify": 2.0,
    },
    
    # =========================================================================
    # EMBER TRAILS (Particle sparks - TRAILS for spark effect)
    # =========================================================================
    
    "ember_trails": {
        "name": "Ember",
        "description": "Spark trails following motion",
        
        "spawn_per_beat": 40,
        "max_points": 150,
        
        "life_frames": 18,
        "trail_length": 24,  # Trails for spark streaks
        "trail_fade": True,
        
        "shape": "diamond",
        "point_size": 3,
        "trace_thickness": 2,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "ember",
        "blur_radius": 4,
        "glow_intensity": 0.6,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.82,
    },
    
    # =========================================================================
    # SOFT BLOBS (Organic shapes - TRAILS for dreamy flow)
    # =========================================================================
    
    "soft_blobs": {
        "name": "Soft Blobs",
        "description": "Dreamy organic flow",
        
        "spawn_per_beat": 15,
        "max_points": 60,
        
        "life_frames": 36,
        "trail_length": 22,  # Trails for dreamy effect
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 14,
        "trace_thickness": 8,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "soft_pastel",
        "blur_radius": 16,
        "glow_intensity": 0.4,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
    },
    
    # =========================================================================
    # CODENET OVERLAY (Feature network with Delaunay mesh + labels)
    # =========================================================================
    
    "codenet_overlay": {
        "name": "CodeNet",
        "description": "Feature network with labeled nodes",
        
        "spawn_per_beat": 0,
        "max_points": 80,  # Max feature points to detect
        
        "life_frames": 1,
        "trail_length": 0,
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        # CodeNet specific
        "text_mode": "codenet_overlay",
        "connection_max_dist": 150,
        "node_radius": 4,
        "label_font_scale": 0.28,
        "blend_alpha": 0.85,
    },
    
    # =========================================================================
    # CODE SHADOW (ASCII/Matrix density effect)
    # =========================================================================
    
    "code_shadow": {
        "name": "CodeShadow",
        "description": "ASCII code forming the image",
        
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        # CodeShadow specific
        "text_mode": "code_shadow",
        "cell_size": 8,
        "char_palette": " .·:;=+*#@",
        "color_dark": (0, 0, 140),      # Deep red BGR
        "color_bright": (0, 200, 0),    # Green BGR
        "threshold_split": 0.45,
    },
    
    # =========================================================================
    # BINARY BLOOM (0/1 digits on solid color background)
    # =========================================================================
    
    "binary_bloom": {
        "name": "Binary Bloom",
        "description": "0/1 digits forming the subject",
        
        "spawn_per_beat": 0,
        "max_points": 0,
        
        "life_frames": 1,
        "trail_length": 0,
        "trail_fade": False,
        
        "shape": "none",
        "point_size": 0,
        "trace_thickness": 0,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        # Binary Bloom specific
        "text_mode": "binary_bloom",
        "bg_color": (160, 40, 0),         # Deep Azure Blue (BGR)
        "digit_color": (255, 255, 255),   # White
        "edge_color": (255, 255, 255),    # Bright white edges
        "accent_color": (200, 50, 255),   # Bright Magenta
        "grid_step": 8,                   # Denser grid
        "edge_grid_step": 5,              # Very dense edges
        "accent_ratio": 0.08,
        "binary_font_scale": 0.35,
        "edge_font_scale": 0.40,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_preset(name: str) -> dict[str, Any]:
    """Get a preset by name, with fallback to blob_track."""
    return PRESETS.get(name, PRESETS["blob_track"])


def get_preset_colors(preset: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    """Get the color palette for a preset."""
    color_mode = preset.get("color_mode", "clean_white")
    return COLOR_PALETTES.get(color_mode, COLOR_PALETTES["clean_white"])


def list_presets() -> list[dict[str, str]]:
    """List all available presets with metadata."""
    return [
        {
            "id": key,
            "name": preset["name"],
            "description": preset["description"],
        }
        for key, preset in PRESETS.items()
    ]


def validate_preset(preset: dict[str, Any]) -> dict[str, Any]:
    """Validate and fill in defaults for a preset config."""
    defaults = {
        "spawn_per_beat": 30,
        "max_points": 150,
        "life_frames": 30,
        "trail_length": 25,
        "trail_fade": True,
        "shape": "circle",
        "point_size": 5,
        "trace_thickness": 1,
        "connect_points": False,
        "max_connect_distance": 100,
        "connection_thickness": 1,
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        "use_colormap": False,
        "darken_factor": 0.9,
    }
    
    result = defaults.copy()
    result.update(preset)
    return result
