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
        # Surveillance green - the classic look
        "point": (0, 255, 80),
        "trail": (0, 200, 60),
        "line": (0, 150, 50),
        "glow": (0, 255, 100),
        "background_tint": None,
    },
    "cctv_green": {
        # Slightly different surveillance green
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
        # Matrix-style green on black
        "point": (80, 255, 80),
        "trail": (60, 200, 60),
        "line": (40, 150, 40),
        "glow": (100, 255, 100),
        "background_tint": None,
    },
    "numeric_gold": {
        # Warm amber/gold for numeric aura
        "point": (50, 200, 255),       # BGR: amber/gold
        "trail": (40, 180, 230),
        "line": (30, 140, 180),
        "glow": (60, 220, 255),
        "background_tint": (5, 10, 15),
    },
    "catodic": {
        # CRT monitor / retro tech aesthetic
        "point": (255, 255, 255),      # Bright white core
        "trail": (200, 200, 200),
        "line": (255, 180, 100),       # Cyan-ish wireframe
        "glow": (255, 160, 80),
        "background_tint": (8, 5, 3),
    },
    "biometric": {
        # Clinical blue-green for biometric scans
        "point": (200, 255, 200),
        "trail": (180, 230, 180),
        "line": (100, 200, 150),
        "glow": (220, 255, 220),
        "background_tint": (5, 10, 5),
    },
}


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

PRESETS: dict[str, dict[str, Any]] = {
    # =========================================================================
    # DETECTION PRESETS (First - these are the flagship)
    # =========================================================================
    
    "face_scanner": {
        "name": "Face Scanner",
        "description": "AI detection with CCTV-style tracking boxes",
        
        "spawn_per_beat": 25,
        "max_points": 80,
        
        "life_frames": 25,
        "trail_length": 15,
        "trail_fade": True,
        
        "shape": "cross",
        "point_size": 4,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 60,
        "connection_thickness": 1,
        
        "color_mode": "cctv_green",
        "blur_radius": 0,
        "glow_intensity": 0.4,
        "scanlines": True,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
        
        # Face detection options
        "detect_faces": True,
        "face_boxes": True,
        "face_glow": True,
        "cctv_overlay": True,  # Add CCTV-style timestamp overlay
        "biometric_data": False,
    },
    
    "biometric": {
        "name": "Biometric",
        "description": "Full biometric analysis mode",
        
        "spawn_per_beat": 35,
        "max_points": 120,
        
        "life_frames": 20,
        "trail_length": 12,
        "trail_fade": True,
        
        "shape": "square",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 50,
        "connection_thickness": 1,
        
        "color_mode": "biometric",
        "blur_radius": 0,
        "glow_intensity": 0.3,
        "scanlines": True,
        "high_contrast_bw": False,
        
        "darken_factor": 0.88,
        
        # Full detection
        "detect_faces": True,
        "detect_mesh": True,
        "face_boxes": True,
        "face_glow": True,
        "cctv_overlay": True,
        "biometric_data": True,
    },
    
    "face_mesh": {
        "name": "Face Mesh",
        "description": "468-point face landmark visualization",
        
        "spawn_per_beat": 15,
        "max_points": 60,
        
        "life_frames": 30,
        "trail_length": 20,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 2,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "ice",
        "blur_radius": 0,
        "glow_intensity": 0.5,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.82,
        
        # Face mesh options
        "detect_faces": True,
        "detect_mesh": True,
        "face_glow": True,
    },
    
    "surveillance_glow": {
        "name": "Surveillance Glow",
        "description": "Cold tracking overlay with clinical precision",
        
        "spawn_per_beat": 30,
        "max_points": 120,
        
        "life_frames": 25,
        "trail_length": 18,
        "trail_fade": True,
        
        "shape": "cross",
        "point_size": 5,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 70,
        "connection_thickness": 1,
        
        "color_mode": "alert",
        "blur_radius": 0,
        "glow_intensity": 0.5,
        "scanlines": True,
        "high_contrast_bw": True,
        
        "darken_factor": 0.90,
    },
    
    # =========================================================================
    # VISUAL / TEXT EFFECTS
    # =========================================================================
    
    "data_body": {
        "name": "Data Body",
        "description": "Your silhouette rebuilt from letters and numbers",
        
        # Minimal point tracking (text replaces points visually)
        "spawn_per_beat": 8,
        "max_points": 30,
        
        "life_frames": 18,
        "trail_length": 10,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 2,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "data_green",
        "blur_radius": 0,
        "glow_intensity": 0.2,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.95,
        
        # Text effect options
        "text_mode": "data_body",
        "glyph_chars": "ABCDEF0123456789@#$%",
        "glyph_cell_size": 8,           # Smaller = denser
        "glyph_jitter": 2,              # Random offset (px)
        "min_brightness": 35,           # Threshold to place glyph (0-255)
        "glyph_font_scale": 0.32,       # cv2.putText scale
        "invert_background": False,     # True = white bg, False = dark bg
    },
    
    "numeric_aura": {
        "name": "Numeric Aura",
        "description": "Glowing 0s and 1s trace your digital presence",
        
        # Minimal point tracking (text is the main visual)
        "spawn_per_beat": 8,
        "max_points": 30,
        
        "life_frames": 22,
        "trail_length": 12,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 2,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "numeric_gold",
        "blur_radius": 0,
        "glow_intensity": 0,            # Handled by text glow separately
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.88,
        
        # Text effect options
        "text_mode": "numeric_aura",
        "glyph_chars": "01",
        "edge_threshold": 40,           # Canny edge threshold (lower = more edges)
        "glyph_density": 0.5,           # Probability of placing glyph at edge point
        "glyph_font_scale": 0.38,
        "text_glow_radius": 15,         # Blur kernel for halo
        "text_glow_intensity": 0.7,     # Additive blend strength
    },
    
    "catodic_cube": {
        "name": "Catodic Cube",
        "description": "Screen breaking into 3D wireframe depth",
        
        # Minimal point tracking (cube effect is the star)
        "spawn_per_beat": 12,
        "max_points": 50,
        
        "life_frames": 18,
        "trail_length": 12,
        "trail_fade": True,
        
        "shape": "square",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 70,
        "connection_thickness": 1,
        
        "color_mode": "catodic",
        "blur_radius": 0,
        "glow_intensity": 0.4,
        "scanlines": True,
        "high_contrast_bw": False,
        
        "darken_factor": 0.80,
        
        # Catodic Cube effect options
        "cube_mode": True,
        "depth_amount": 0.5,            # How far lines recede (0-1)
        "wireframe_layers": 6,          # Number of nested rectangles
        "wireframe_intensity": 0.8,     # Line brightness (0-1)
        "rgb_offset_px": 5,             # Chromatic aberration offset
        "glitch_frequency": 6,          # Every N frames, add glitch
        "glitch_strength": 0.4,         # Intensity of horizontal slice shift
        "motion_amplify": 1.8,          # Multiplier for motion-based effects
    },
    
    "heat_map": {
        "name": "Heat Map",
        "description": "Thermal vision tracking your heat signature",
        
        "spawn_per_beat": 45,
        "max_points": 160,
        
        "life_frames": 35,
        "trail_length": 25,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 12,
        "trace_thickness": 5,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "thermal",
        "blur_radius": 12,
        "glow_intensity": 0.6,
        "scanlines": False,
        "high_contrast_bw": False,
        "use_colormap": True,
        
        "darken_factor": 0.85,
    },
    
    # =========================================================================
    # ABSTRACT / ARTISTIC EFFECTS
    # =========================================================================
    
    "grid_trace": {
        # Metadata
        "name": "Grid Trace",
        "description": "Sharp geometric network that follows motion",
        
        # Spawning
        "spawn_per_beat": 35,
        "max_points": 180,
        
        # Lifetime & trails
        "life_frames": 30,
        "trail_length": 22,
        "trail_fade": True,
        
        # Shape & size
        "shape": "square",  # square, circle, diamond, cross
        "point_size": 4,
        "trace_thickness": 1,
        
        # Connections
        "connect_points": True,
        "max_connect_distance": 90,
        "connection_thickness": 1,
        
        # Colors & effects
        "color_mode": "neon_grid",
        "blur_radius": 0,
        "glow_intensity": 0.2,
        "scanlines": False,
        "high_contrast_bw": False,
        
        # Background
        "darken_factor": 0.88,
    },
    
    "soft_blobs": {
        "name": "Soft Blobs",
        "description": "Dreamy organic shapes that float and merge",
        
        "spawn_per_beat": 20,
        "max_points": 100,
        
        "life_frames": 45,
        "trail_length": 30,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 16,
        "trace_thickness": 8,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "soft_pastel",
        "blur_radius": 18,
        "glow_intensity": 0.4,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
    },
    
    "ember_trails": {
        "name": "Ember Trails",
        "description": "Sparks and embers that trace your movement",
        
        "spawn_per_beat": 55,
        "max_points": 220,
        
        "life_frames": 22,
        "trail_length": 35,
        "trail_fade": True,
        
        "shape": "diamond",
        "point_size": 4,
        "trace_thickness": 2,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "ember",
        "blur_radius": 4,
        "glow_intensity": 0.7,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.80,
    },
    
    "minimal_void": {
        "name": "Minimal Void",
        "description": "Sparse, understated presence detection",
        
        "spawn_per_beat": 12,
        "max_points": 50,
        
        "life_frames": 55,
        "trail_length": 40,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 130,
        "connection_thickness": 1,
        
        "color_mode": "void",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.94,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_preset(name: str) -> dict[str, Any]:
    """Get a preset by name, with fallback to face_scanner."""
    return PRESETS.get(name, PRESETS["face_scanner"])


def get_preset_colors(preset: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    """Get the color palette for a preset."""
    color_mode = preset.get("color_mode", "neon_grid")
    return COLOR_PALETTES.get(color_mode, COLOR_PALETTES["neon_grid"])


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
        "color_mode": "neon_grid",
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
