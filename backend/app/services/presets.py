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
    # BLOB TRACKING (TouchDesigner style - FIRST)
    # =========================================================================
    
    "blob_track": {
        "name": "Blob Track",
        "description": "Clean white boxes with coordinate labels",
        
        "spawn_per_beat": 15,
        "max_points": 50,
        
        "life_frames": 20,
        "trail_length": 10,
        "trail_fade": True,
        
        "shape": "square",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
        
        # Blob tracking options
        "text_mode": "blob_track",
        "blob_threshold": 25,
        "blob_blur": 7,
        "min_blob_area": 300,
        "max_blobs": 60,
        "label_scale": 0.32,
        "bg_alpha": 0.12,
    },
    
    # =========================================================================
    # PARTICLE SILHOUETTE (bb.dere style)
    # =========================================================================
    
    "particle_silhouette": {
        "name": "Particle Cloud",
        "description": "Dense point cloud forming your silhouette",
        
        "spawn_per_beat": 10,
        "max_points": 40,
        
        "life_frames": 25,
        "trail_length": 15,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 1,
        "trace_thickness": 1,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "particle_cream",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 1.0,
        
        # Particle silhouette options
        "text_mode": "particle_silhouette",
        "particle_density": 0.025,
        "brightness_threshold": 35,
        "particle_size": 1,
        "scatter_range": 2,
        "particle_glow": 0.6,
        "connect_particles": True,
    },
    
    # =========================================================================
    # FACE SCANNER (Clean minimal)
    # =========================================================================
    
    "face_scanner": {
        "name": "Face Scanner",
        "description": "Minimal detection boxes with labels",
        
        "spawn_per_beat": 20,
        "max_points": 60,
        
        "life_frames": 20,
        "trail_length": 12,
        "trail_fade": True,
        
        "shape": "cross",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 50,
        "connection_thickness": 1,
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0.2,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.82,
        
        # Face detection options
        "detect_faces": True,
        "face_boxes": True,
        "face_glow": False,
        "cctv_overlay": False,
        "biometric_data": False,
    },
    
    # =========================================================================
    # NUMBER CLOUD
    # =========================================================================
    
    "number_cloud": {
        "name": "Number Cloud",
        "description": "Frame IDs scattered across motion",
        
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
        
        "color_mode": "clean_white",
        "blur_radius": 0,
        "glow_intensity": 0,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
        
        # Number cloud options
        "text_mode": "number_cloud",
        "number_density": 0.35,
        "number_font_scale": 0.28,
        "show_coordinates": False,
        "start_number": 19000,
        "max_numbers": 1000,
        "bg_alpha": 0.18,
    },
    
    # =========================================================================
    # BIOMETRIC (Full analysis)
    # =========================================================================
    
    "biometric": {
        "name": "Biometric",
        "description": "Full identity analysis mode",
        
        "spawn_per_beat": 30,
        "max_points": 100,
        
        "life_frames": 18,
        "trail_length": 10,
        "trail_fade": True,
        
        "shape": "square",
        "point_size": 3,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 45,
        "connection_thickness": 1,
        
        "color_mode": "biometric",
        "blur_radius": 0,
        "glow_intensity": 0.25,
        "scanlines": True,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
        
        # Full detection
        "detect_faces": True,
        "detect_mesh": True,
        "face_boxes": True,
        "face_glow": True,
        "cctv_overlay": True,
        "biometric_data": True,
    },
    
    # =========================================================================
    # FACE MESH (468 points)
    # =========================================================================
    
    "face_mesh": {
        "name": "Face Mesh",
        "description": "468-point face landmark visualization",
        
        "spawn_per_beat": 12,
        "max_points": 50,
        
        "life_frames": 28,
        "trail_length": 18,
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
        
        "darken_factor": 0.80,
        
        "detect_faces": True,
        "detect_mesh": True,
        "face_glow": True,
    },
    
    # =========================================================================
    # DATA BODY (Silhouette from text)
    # =========================================================================
    
    "data_body": {
        "name": "Data Body",
        "description": "Your silhouette rebuilt from code",
        
        "spawn_per_beat": 8,
        "max_points": 25,
        
        "life_frames": 16,
        "trail_length": 8,
        "trail_fade": True,
        
        "shape": "circle",
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
    
    # =========================================================================
    # GRID TRACE (Geometric network)
    # =========================================================================
    
    "grid_trace": {
        "name": "Grid Trace",
        "description": "Sharp geometric network following motion",
        
        "spawn_per_beat": 30,
        "max_points": 150,
        
        "life_frames": 28,
        "trail_length": 20,
        "trail_fade": True,
        
        "shape": "square",
        "point_size": 4,
        "trace_thickness": 1,
        
        "connect_points": True,
        "max_connect_distance": 80,
        "connection_thickness": 1,
        
        "color_mode": "neon_grid",
        "blur_radius": 0,
        "glow_intensity": 0.15,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.85,
    },
    
    # =========================================================================
    # THERMAL (Heat map)
    # =========================================================================
    
    "heat_map": {
        "name": "Thermal",
        "description": "Heat signature visualization",
        
        "spawn_per_beat": 40,
        "max_points": 140,
        
        "life_frames": 32,
        "trail_length": 22,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 14,
        "trace_thickness": 6,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "thermal",
        "blur_radius": 14,
        "glow_intensity": 0.6,
        "scanlines": False,
        "high_contrast_bw": False,
        "use_colormap": True,
        
        "darken_factor": 0.82,
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
    # EMBER TRAILS (Particle sparks)
    # =========================================================================
    
    "ember_trails": {
        "name": "Ember",
        "description": "Sparks tracing your movement",
        
        "spawn_per_beat": 50,
        "max_points": 200,
        
        "life_frames": 20,
        "trail_length": 32,
        "trail_fade": True,
        
        "shape": "diamond",
        "point_size": 4,
        "trace_thickness": 2,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "ember",
        "blur_radius": 5,
        "glow_intensity": 0.75,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.78,
    },
    
    # =========================================================================
    # SOFT BLOBS (Organic shapes)
    # =========================================================================
    
    "soft_blobs": {
        "name": "Soft Blobs",
        "description": "Dreamy organic flow",
        
        "spawn_per_beat": 18,
        "max_points": 80,
        
        "life_frames": 42,
        "trail_length": 28,
        "trail_fade": True,
        
        "shape": "circle",
        "point_size": 18,
        "trace_thickness": 10,
        
        "connect_points": False,
        "max_connect_distance": 0,
        "connection_thickness": 0,
        
        "color_mode": "soft_pastel",
        "blur_radius": 20,
        "glow_intensity": 0.45,
        "scanlines": False,
        "high_contrast_bw": False,
        
        "darken_factor": 0.82,
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
