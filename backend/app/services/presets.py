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
        "line": (80, 80, 80),
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
        "line": (0, 100, 40),
        "glow": (0, 255, 100),
        "background_tint": None,
    },
    "thermal": {
        "point": (0, 120, 255),
        "trail": (0, 80, 200),
        "line": (0, 60, 150),
        "glow": (50, 150, 255),
        "background_tint": (10, 5, 0),
    },
    "ember": {
        "point": (255, 120, 50),
        "trail": (255, 80, 30),
        "line": (200, 60, 20),
        "glow": (255, 150, 80),
        "background_tint": (15, 5, 0),
    },
    "ice": {
        "point": (200, 240, 255),
        "trail": (150, 200, 255),
        "line": (80, 120, 180),
        "glow": (180, 220, 255),
        "background_tint": (5, 10, 20),
    },
    "void": {
        "point": (180, 180, 180),
        "trail": (100, 100, 100),
        "line": (40, 40, 40),
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
        "line": (255, 200, 100),       # Cyan-ish wireframe
        "glow": (255, 180, 80),
        "background_tint": (5, 5, 10),
    },
}


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

PRESETS: dict[str, dict[str, Any]] = {
    "face_scanner": {
        "name": "Face Scanner",
        "description": "AI detection with CCTV-style tracking boxes",
        
        "spawn_per_beat": 30,
        "max_points": 100,
        
        "life_frames": 30,
        "trail_length": 20,
        "trail_fade": True,
        
        "shape": "cross",
        "point_size": 4,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 80,
        "connection_thickness": 1,
        
        "color_mode": "alert",
        "blur_radius": 0,
        "glow_intensity": 0.3,
        "scanlines": True,
        "high_contrast_bw": False,
        
        "darken_factor": 0.90,
        
        # Face detection options
        "detect_faces": True,
        "face_boxes": True,
        "face_glow": True,
    },
    
    "grid_trace": {
        # Metadata
        "name": "Grid Trace",
        "description": "Sharp geometric network that follows motion",
        
        # Spawning
        "spawn_per_beat": 40,
        "max_points": 200,
        
        # Lifetime & trails
        "life_frames": 35,
        "trail_length": 25,
        "trail_fade": True,
        
        # Shape & size
        "shape": "square",  # square, circle, diamond, cross
        "point_size": 4,
        "trace_thickness": 1,
        
        # Connections
        "connect_points": True,
        "max_connect_distance": 100,
        "connection_thickness": 1,
        
        # Colors & effects
        "color_mode": "neon_grid",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        # Background
        "darken_factor": 0.92,
    },
    
    "soft_blobs": {
        "name": "Soft Blobs",
        "description": "Dreamy organic shapes that float and merge",
        
        "spawn_per_beat": 25,
        "max_points": 120,
        
        "life_frames": 50,
        "trail_length": 35,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 14,
        "trace_thickness": 6,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "soft_pastel",
        "blur_radius": 15,
        "glow_intensity": 0.3,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.88,
    },
    
    "surveillance_glow": {
        "name": "Surveillance Glow",
        "description": "Cold tracking overlay with clinical precision",
        
        "spawn_per_beat": 35,
        "max_points": 150,
        
        "life_frames": 30,
        "trail_length": 20,
        "trail_fade": True,
        
        "shape": "cross",
        "point_size": 6,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 80,
        "connection_thickness": 1,
        
        "color_mode": "alert",
        "blur_radius": 0,
        "glow_intensity": 0.4,
        "scanlines": True,
        "high_contrast_bw": True,
        
        "darken_factor": 0.95,
    },
    
    "heat_map": {
        "name": "Heat Map",
        "description": "Thermal vision tracking your heat signature",
        
        "spawn_per_beat": 50,
        "max_points": 180,
        
        "life_frames": 40,
        "trail_length": 30,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 10,
        "trace_thickness": 4,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "thermal",
        "blur_radius": 8,
        "glow_intensity": 0.5,
        "scanlines": False,
        "high_contrast_bw": False,
        "use_colormap": True,
        
        "darken_factor": 0.90,
    },
    
    "ember_trails": {
        "name": "Ember Trails",
        "description": "Sparks and embers that trace your movement",
        
        "spawn_per_beat": 60,
        "max_points": 250,
        
        "life_frames": 25,
        "trail_length": 40,
        "trail_fade": True,
        
        "shape": "diamond",
        "point_size": 5,
        "trace_thickness": 2,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "ember",
        "blur_radius": 3,
        "glow_intensity": 0.6,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
    },
    
    "minimal_void": {
        "name": "Minimal Void",
        "description": "Sparse, understated presence detection",
        
        "spawn_per_beat": 15,
        "max_points": 60,
        
        "life_frames": 60,
        "trail_length": 45,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 150,
        "connection_thickness": 1,
        
        "color_mode": "void",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.96,
    },
    
    "face_mesh": {
        "name": "Face Mesh",
        "description": "468-point face landmark visualization",
        
        "spawn_per_beat": 20,
        "max_points": 80,
        
        "life_frames": 35,
        "trail_length": 25,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "ice",
        "blur_radius": 0,
        "glow_intensity": 0.5,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.88,
        
        # Face mesh options
        "detect_faces": True,
        "detect_mesh": True,
        "face_glow": True,
    },
    
    "biometric": {
        "name": "Biometric",
        "description": "Full biometric analysis mode",
        
        "spawn_per_beat": 40,
        "max_points": 150,
        
        "life_frames": 25,
        "trail_length": 15,
        "trail_fade": True,
        
        "shape": "square",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 60,
        "connection_thickness": 1,
        
        "color_mode": "alert",
        "blur_radius": 0,
        "glow_intensity": 0.4,
        "scanlines": True,
        "high_contrast_bw": True,
        
        "darken_factor": 0.92,
        
        # Full detection
        "detect_faces": True,
        "detect_mesh": True,
        "face_boxes": True,
        "face_glow": True,
    },
    
    # =========================================================================
    # TEXT-BASED EFFECTS
    # =========================================================================
    
    "data_body": {
        "name": "Data Body",
        "description": "Your silhouette rebuilt from letters and numbers",
        
        # Minimal point tracking (text replaces points visually)
        "spawn_per_beat": 10,
        "max_points": 40,
        
        "life_frames": 20,
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
        "glyph_chars": "ABCDEF0123456789",
        "glyph_cell_size": 10,          # Sample every N pixels
        "glyph_jitter": 2,              # Random offset (px)
        "min_brightness": 40,           # Threshold to place glyph (0-255)
        "glyph_font_scale": 0.35,       # cv2.putText scale
        "invert_background": False,     # True = white bg, False = dark bg
    },
    
    "numeric_aura": {
        "name": "Numeric Aura",
        "description": "Glowing 0s and 1s trace your digital presence",
        
        # Minimal point tracking (text is the main visual)
        "spawn_per_beat": 10,
        "max_points": 40,
        
        "life_frames": 25,
        "trail_length": 15,
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
        
        "darken_factor": 0.92,
        
        # Text effect options
        "text_mode": "numeric_aura",
        "glyph_chars": "01",
        "edge_threshold": 50,           # Canny edge threshold
        "glyph_density": 0.4,           # Probability of placing glyph at edge point
        "glyph_font_scale": 0.4,
        "text_glow_radius": 11,         # Blur kernel for halo
        "text_glow_intensity": 0.6,     # Additive blend strength
    },
    
    # =========================================================================
    # DEPTH / GLITCH EFFECTS
    # =========================================================================
    
    "catodic_cube": {
        "name": "Catodic Cube",
        "description": "Screen breaking into 3D wireframe depth",
        
        # Minimal point tracking (cube effect is the star)
        "spawn_per_beat": 15,
        "max_points": 60,
        
        "life_frames": 20,
        "trail_length": 15,
        "trail_fade": True,
        
        "shape": "square",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 80,
        "connection_thickness": 1,
        
        "color_mode": "catodic",
        "blur_radius": 0,
        "glow_intensity": 0.3,
        "scanlines": True,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
        
        # Catodic Cube effect options
        "cube_mode": True,
        "depth_amount": 0.4,            # How far lines recede (0-1)
        "wireframe_layers": 5,          # Number of nested rectangles
        "wireframe_intensity": 0.7,     # Line brightness (0-1)
        "rgb_offset_px": 4,             # Chromatic aberration offset
        "glitch_frequency": 8,          # Every N frames, add glitch
        "glitch_strength": 0.3,         # Intensity of horizontal slice shift
        "motion_amplify": 1.5,          # Multiplier for motion-based effects
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_preset(name: str) -> dict[str, Any]:
    """Get a preset by name, with fallback to grid_trace."""
    return PRESETS.get(name, PRESETS["grid_trace"])


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

