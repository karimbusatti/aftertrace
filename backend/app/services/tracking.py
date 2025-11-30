"""
Feature detection and optical flow tracking.

Handles:
- Detecting good features (Shi-Tomasi corners on edges)
- Lucas-Kanade optical flow tracking
- Point lifecycle management (spawn, track, age, die)
"""

import cv2
import numpy as np
from typing import Any

from .types import TrackedPoint


# Lucas-Kanade optical flow parameters
LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}

# Feature detection parameters
FEATURE_PARAMS = {
    "maxCorners": 100,
    "qualityLevel": 0.05,
    "minDistance": 15,
    "blockSize": 7,
}


def detect_features(
    gray_frame: np.ndarray,
    mask: np.ndarray | None = None,
    max_points: int = 50,
    use_edges: bool = True,
) -> np.ndarray:
    """
    Detect good features to track in a frame.
    
    Args:
        gray_frame: Grayscale frame
        mask: Optional mask (255 = detect here, 0 = ignore)
        max_points: Maximum number of points to return
        use_edges: If True, focus detection on edge regions
    
    Returns:
        Array of points, shape (N, 1, 2), or empty array if none found.
    """
    if use_edges:
        # Apply Canny edge detection and dilate to create edge mask
        edges = cv2.Canny(gray_frame, 50, 150)
        edges = cv2.dilate(edges, None, iterations=2)
        
        # Combine with input mask if provided
        if mask is not None:
            edges = cv2.bitwise_and(edges, mask)
        detection_mask = edges
    else:
        detection_mask = mask
    
    # Detect Shi-Tomasi corners
    params = FEATURE_PARAMS.copy()
    params["maxCorners"] = max_points
    
    points = cv2.goodFeaturesToTrack(gray_frame, mask=detection_mask, **params)
    
    if points is None:
        return np.array([]).reshape(0, 1, 2).astype(np.float32)
    
    return points.astype(np.float32)


def compute_trackability_score(
    max_track_frames: int,
    avg_lifespan: float,
    total_points_spawned: int,
    total_frames: int,
    max_points_capacity: int,
    life_frames_limit: int,
) -> int:
    """
    Compute a 0-100 "trackability" score.
    
    Higher score = easier to track = more surveillance-vulnerable.
    
    The score combines three factors:
    1. Longevity (40%): How long the longest track lasted relative to limit
    2. Avg lifespan (30%): Average point survival relative to limit
    3. Density (30%): How many points were spawned relative to capacity
    
    Args:
        max_track_frames: Longest any point survived
        avg_lifespan: Average age at death
        total_points_spawned: Total points that were tracked
        total_frames: Video length in frames
        max_points_capacity: Max points allowed by preset
        life_frames_limit: Max life per point allowed by preset
    
    Returns:
        Integer score 0-100
    """
    if total_frames == 0 or life_frames_limit == 0:
        return 0
    
    # Factor 1: Longevity - how close did the longest track get to the limit?
    # If a point survived the full life_frames_limit, that's 100% for this factor
    longevity_ratio = min(max_track_frames / life_frames_limit, 1.0)
    
    # Factor 2: Average lifespan - how well did points survive on average?
    avg_lifespan_ratio = min(avg_lifespan / life_frames_limit, 1.0)
    
    # Factor 3: Density - how many points were spawned relative to video length?
    # Expected: ~1 spawn event every 15 frames, ~30 points each = 2 points/frame capacity
    # Normalize by frames and capacity
    expected_spawns = total_frames / 15  # rough spawn frequency
    density_ratio = min(total_points_spawned / (expected_spawns * 10 + 1), 1.0)
    
    # Weighted combination
    score = (
        longevity_ratio * 40 +      # 40% weight: longest track
        avg_lifespan_ratio * 30 +   # 30% weight: average survival
        density_ratio * 30          # 30% weight: point density
    )
    
    return int(round(score))


def track_points_lk(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track points from previous frame to current frame using Lucas-Kanade.
    
    Args:
        prev_gray: Previous grayscale frame
        curr_gray: Current grayscale frame
        prev_points: Points to track, shape (N, 1, 2)
    
    Returns:
        (new_points, status) - new positions and status array (1 = tracked, 0 = lost)
    """
    if len(prev_points) == 0:
        return np.array([]).reshape(0, 1, 2).astype(np.float32), np.array([])
    
    # Forward tracking
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **LK_PARAMS
    )
    
    # Backward tracking for validation (reduces drift)
    if new_points is not None and len(new_points) > 0:
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, new_points, None, **LK_PARAMS
        )
        
        # Check forward-backward consistency
        if back_points is not None:
            diff = np.abs(prev_points - back_points).reshape(-1, 2).max(axis=1)
            status = status.flatten() & (diff < 1.0).astype(np.uint8)
    
    return new_points, status.flatten() if status is not None else np.array([])


class PointTracker:
    """
    Manages a collection of tracked points across frames.
    Handles spawning, tracking, aging, and pruning based on preset config.
    Also collects surveillance stats for trackability analysis.
    """
    
    def __init__(self, preset: dict[str, Any]):
        """
        Initialize tracker with preset configuration.
        
        Reads from preset:
            - max_points: Maximum simultaneous tracked points
            - life_frames: How long a point lives before dying
            - trail_length: Maximum trail history to keep
        """
        self.max_points = preset.get("max_points", 150)
        self.life_frames = preset.get("life_frames", 30)
        self.trail_length = preset.get("trail_length", 25)
        
        self.points: list[TrackedPoint] = []
        self._next_id = 0
        
        # Surveillance stats - track across entire video
        self._max_age_seen = 0          # Longest any point survived
        self._total_point_ages = 0      # Sum of all final ages (for averaging)
        self._total_points_died = 0     # Count of points that completed their lifecycle
        self._ages_at_death: list[int] = []  # Record of all final ages
    
    def spawn_points(
        self,
        gray_frame: np.ndarray,
        frame_idx: int,
        count: int,
    ) -> int:
        """
        Detect and spawn new tracking points.
        
        Args:
            gray_frame: Grayscale frame for detection
            frame_idx: Current frame index
            count: Target number of points to spawn
        
        Returns:
            Number of points actually spawned.
        """
        # Create mask to avoid spawning near existing points
        h, w = gray_frame.shape
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        for p in self.points:
            if p.alive:
                x, y = int(p.position[0]), int(p.position[1])
                cv2.circle(mask, (x, y), 20, 0, -1)
        
        # Check remaining capacity
        alive_count = sum(1 for p in self.points if p.alive)
        remaining = self.max_points - alive_count
        detect_count = min(count, remaining)
        
        if detect_count <= 0:
            return 0
        
        # Detect new features
        new_pts = detect_features(gray_frame, mask=mask, max_points=detect_count)
        
        # Create TrackedPoint objects
        spawned = 0
        for pt in new_pts:
            pos = pt.flatten()
            self.points.append(TrackedPoint(
                id=self._next_id,
                position=pos,
                birth_frame=frame_idx,
                trail=[],
                alive=True,
                age=0,
            ))
            self._next_id += 1
            spawned += 1
        
        return spawned
    
    def update(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        frame_idx: int,
        frame_shape: tuple[int, int],
    ) -> int:
        """
        Track all alive points to new positions and age them.
        
        Args:
            prev_gray: Previous grayscale frame
            curr_gray: Current grayscale frame
            frame_idx: Current frame index
            frame_shape: (height, width) for bounds checking
        
        Returns:
            Number of points still alive after update.
        """
        alive_points = [p for p in self.points if p.alive]
        if not alive_points:
            return 0
        
        # Gather points for tracking
        pts_array = np.array(
            [p.position for p in alive_points]
        ).reshape(-1, 1, 2).astype(np.float32)
        
        # Track with optical flow
        new_pts, status = track_points_lk(prev_gray, curr_gray, pts_array)
        
        h, w = frame_shape
        
        # Update points
        for i, point in enumerate(alive_points):
            # Age the point
            point.age += 1
            
            # Check if point is too old
            if point.age >= self.life_frames:
                self._record_point_death(point)
                point.alive = False
                continue
            
            # Check tracking status and bounds
            if i >= len(status) or i >= len(new_pts) or status[i] != 1:
                self._record_point_death(point)
                point.alive = False
                continue
            
            # Update position
            new_pos = new_pts[i].flatten()
            
            # Check bounds
            if not (0 <= new_pos[0] < w and 0 <= new_pos[1] < h):
                self._record_point_death(point)
                point.alive = False
                continue
            
            # Update trail and position
            point.trail.append(point.position.copy())
            if len(point.trail) > self.trail_length:
                point.trail.pop(0)
            point.position = new_pos
        
        # Prune dead points with empty trails
        self.points = [
            p for p in self.points
            if p.alive or len(p.trail) > 0
        ]
        
        # Fade trails on dead points
        for p in self.points:
            if not p.alive and len(p.trail) > 0:
                p.trail.pop(0)
        
        return sum(1 for p in self.points if p.alive)
    
    def get_alive_points(self) -> list[TrackedPoint]:
        """Return list of currently alive points."""
        return [p for p in self.points if p.alive]
    
    def get_all_points(self) -> list[TrackedPoint]:
        """Return all points (including those with fading trails)."""
        return self.points
    
    def get_stats(self) -> dict:
        """Return tracking statistics."""
        alive = [p for p in self.points if p.alive]
        return {
            "total": len(self.points),
            "alive": len(alive),
            "avg_age": sum(p.age for p in alive) / len(alive) if alive else 0,
            "avg_trail_len": sum(len(p.trail) for p in self.points) / len(self.points) if self.points else 0,
        }
    
    def _record_point_death(self, point: TrackedPoint):
        """Record stats when a point dies (for surveillance metrics)."""
        age = point.age
        self._ages_at_death.append(age)
        self._total_point_ages += age
        self._total_points_died += 1
        
        if age > self._max_age_seen:
            self._max_age_seen = age
    
    def finalize_remaining_points(self):
        """
        Call at end of video to record ages of points still alive.
        Ensures we count points that never died naturally.
        """
        for point in self.points:
            if point.alive:
                self._record_point_death(point)
                point.alive = False
    
    def get_surveillance_stats(
        self,
        total_frames: int,
        fps: float,
        max_possible_points: int,
        total_points_spawned: int,
    ) -> dict:
        """
        Compute surveillance/trackability statistics.
        
        Args:
            total_frames: Total frames in the video
            fps: Frames per second (to convert frames to seconds)
            max_possible_points: Max points that could be tracked (preset.max_points)
            total_points_spawned: Actual count of points spawned during processing
        
        Returns:
            dict with:
                - max_continuous_tracking_frames: Longest any single point was tracked
                - longest_track_seconds: Same, in seconds
                - average_point_lifespan: Mean age at death
                - trackability_score: 0-100 heuristic score
        """
        # Finalize any still-alive points
        self.finalize_remaining_points()
        
        max_frames = self._max_age_seen
        longest_seconds = max_frames / fps if fps > 0 else 0
        
        avg_lifespan = (
            self._total_point_ages / self._total_points_died
            if self._total_points_died > 0 else 0
        )
        
        # Compute trackability score (0-100)
        # Based on: point density + how long points survived
        trackability = compute_trackability_score(
            max_track_frames=max_frames,
            avg_lifespan=avg_lifespan,
            total_points_spawned=total_points_spawned,
            total_frames=total_frames,
            max_points_capacity=max_possible_points,
            life_frames_limit=self.life_frames,
        )
        
        return {
            "max_continuous_tracking_frames": max_frames,
            "longest_track_seconds": longest_seconds,
            "average_point_lifespan": avg_lifespan,
            "trackability_score": trackability,
        }
