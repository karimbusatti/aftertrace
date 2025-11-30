"""
Shared type definitions for the processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np


PresetName = Literal[
    "grid_trace",
    "soft_blobs",
    "surveillance_glow",
    "heat_map",
    "ember_trails",
    "minimal_void",
]


@dataclass
class TrackedPoint:
    """A point being tracked across frames."""
    id: int
    position: np.ndarray  # [x, y]
    birth_frame: int
    trail: list[np.ndarray] = field(default_factory=list)
    alive: bool = True
    age: int = 0  # Frames since birth


@dataclass
class ProcessingMetadata:
    """Metadata returned after processing."""
    
    # Basic stats
    frames_processed: int = 0
    total_points_spawned: int = 0
    average_points_per_frame: float = 0.0
    beats_detected: int = 0
    duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    output_path: str = ""
    original_path: str = ""  # Path to compressed original for comparison
    preset_used: str = ""
    
    # Surveillance stats (Phase 4)
    max_continuous_tracking_frames: int = 0  # Longest a single point was tracked
    longest_track_seconds: float = 0.0       # Same, converted to seconds
    trackability_score: int = 0              # 0-100, higher = more trackable
    people_detected: int = 0                 # Stub for now (future: face/pose detection)
    
    def to_dict(self) -> dict:
        return {
            # Basic stats
            "frames_processed": self.frames_processed,
            "total_points_spawned": self.total_points_spawned,
            "average_points_per_frame": round(self.average_points_per_frame, 1),
            "beats_detected": self.beats_detected,
            "duration_seconds": round(self.duration_seconds, 2),
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "output_path": self.output_path,
            "original_path": self.original_path,
            "preset_used": self.preset_used,
            # Surveillance stats
            "max_continuous_tracking_frames": self.max_continuous_tracking_frames,
            "longest_track_seconds": round(self.longest_track_seconds, 2),
            "trackability_score": self.trackability_score,
            "people_detected": self.people_detected,
        }
