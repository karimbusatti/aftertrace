"""
Main video processing pipeline.

This module orchestrates the full Aftertrace effect:
1. Load video → extract frames
2. Extract audio → detect beats/onsets  
3. On each spawn frame → detect feature points
4. Track points across frames with optical flow
5. Draw effects based on preset (or composition sequence)
6. Write output video

Usage:
    from app.services.process_video import process_video
    
    # Single effect
    metadata = process_video(
        input_path="input.mp4",
        output_path="output.mp4",
        preset="grid_trace",
    )
    
    # Multi-effect composition
    composition = [
        {"effect_id": "clean", "start": 0.0, "end": 0.2},
        {"effect_id": "codenet_overlay", "start": 0.2, "end": 0.5},
        {"effect_id": "code_shadow", "start": 0.5, "end": 0.8},
        {"effect_id": "binary_bloom", "start": 0.8, "end": 1.0},
    ]
    metadata = process_video(
        input_path="input.mp4",
        output_path="output.mp4",
        composition=composition,
        crossfade_ratio=0.05,
    )
"""

import time
import random
import cv2
import numpy as np
from typing import Any

from .types import ProcessingMetadata, CompositionSegment
from .presets import get_preset, validate_preset
from .audio import extract_audio, analyze_audio, get_spawn_frames
from .tracking import PointTracker
from .effects import draw_frame
from .face_detection import FaceDetector


# =============================================================================
# COMPOSITION HELPERS
# =============================================================================

def get_frame_effects(
    frame_ratio: float,
    composition: list[dict],
    crossfade: float,
) -> list[tuple[str, float]]:
    """
    Determine which effect(s) to apply at a given frame position.
    
    Args:
        frame_ratio: Current frame position (0.0 - 1.0)
        composition: List of {effect_id, start, end} segments
        crossfade: Crossfade ratio (e.g., 0.05 = 5% overlap)
    
    Returns:
        List of (effect_id, weight) tuples. Usually 1 effect at weight=1.0,
        or 2 effects during crossfade transition.
    """
    results = []
    
    for i, seg in enumerate(composition):
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        if seg_start <= frame_ratio < seg_end:
            # We're in this segment
            fade_in_end = seg_start + crossfade
            
            # Check if in crossfade zone at start
            if frame_ratio < fade_in_end and i > 0:
                prev_seg = composition[i - 1]
                t = (frame_ratio - seg_start) / crossfade if crossfade > 0 else 1.0
                results.append((prev_seg["effect_id"], 1.0 - t))
                results.append((seg["effect_id"], t))
            else:
                results.append((seg["effect_id"], 1.0))
            break
    
    return results or [("clean", 1.0)]


def apply_composed_frame(
    frame: np.ndarray,
    effects_with_weights: list[tuple[str, float]],
    preset_cache: dict[str, dict],
    frame_idx: int,
    tracker: "PointTracker",
    face_data: dict | None,
    overlay_mode: bool,
) -> np.ndarray:
    """
    Apply one or more effects to a frame with blending.
    
    Args:
        frame: Original video frame
        effects_with_weights: List of (effect_id, weight) tuples
        preset_cache: Dict mapping effect_id to preset config
        frame_idx: Current frame index
        tracker: Point tracker for effects that need tracking data
        face_data: Face detection data if available
        overlay_mode: Whether to use overlay blending
    
    Returns:
        Blended output frame
    """
    if len(effects_with_weights) == 1:
        effect_id, _ = effects_with_weights[0]
        if effect_id == "clean":
            return frame.copy()
        
        preset = preset_cache.get(effect_id)
        if preset is None:
            return frame.copy()
        
        return draw_frame(
            frame,
            tracker.get_all_points(),
            preset,
            frame_idx,
            overlay_mode,
            face_data=face_data,
        )
    
    # Multiple effects: blend them
    outputs = []
    for effect_id, weight in effects_with_weights:
        if effect_id == "clean":
            outputs.append((frame.copy(), weight))
        else:
            preset = preset_cache.get(effect_id)
            if preset is None:
                outputs.append((frame.copy(), weight))
            else:
                effect_frame = draw_frame(
                    frame,
                    tracker.get_all_points(),
                    preset,
                    frame_idx,
                    overlay_mode,
                    face_data=face_data,
                )
                outputs.append((effect_frame, weight))
    
    # Weighted blend
    blended = np.zeros_like(frame, dtype=np.float32)
    for out, w in outputs:
        blended += out.astype(np.float32) * w
    
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_video(
    input_path: str,
    output_path: str,
    preset: str | dict[str, Any] = "grid_trace",
    overlay_mode: bool = False,
    return_metadata: bool = True,
    original_output_path: str | None = None,
    composition: list[dict] | None = None,
    crossfade_ratio: float = 0.05,
) -> ProcessingMetadata:
    """
    Process a video with Aftertrace visual effects.
    
    Args:
        input_path: Path to input video file
        output_path: Path for output video file
        preset: Preset name (str) or custom preset dict (ignored if composition is set)
        overlay_mode: If True, blend effects at ~40% over original video
                      If False, replace background with darkened version (default)
        return_metadata: Whether to compute and return metadata
        original_output_path: If provided, save a re-encoded copy of original here
        composition: Optional list of effect segments for multi-effect mode
                     Each segment: {"effect_id": str, "start": float, "end": float}
                     "clean" = original video, other values = preset names
        crossfade_ratio: Overlap ratio at segment boundaries (default 5%)
    
    Returns:
        ProcessingMetadata with stats about the processing.
    
    How it works:
    -------------
    For each frame:
      1. Check if this is a "spawn frame" (beat/onset detected)
      2. If spawn frame: detect new feature points on edges
      3. Track existing points with Lucas-Kanade optical flow
      4. Age points, kill old ones, fade trails
      5. Draw the frame: trails, points, connections, effects
      6. Write frame to output
    
    Composition mode:
    -----------------
    When `composition` is provided, the video is split into segments,
    each with a different effect. Crossfade blending occurs at segment
    boundaries for smooth transitions.
    
    Preset controls (single effect mode):
    -------------------------------------
    - spawn_per_beat: Points spawned per beat/onset
    - max_points: Cap on simultaneous tracked points
    - life_frames: How long points live before dying
    - trail_length: How much position history to keep
    - shape: circle, square, diamond, cross
    - connect_points: Whether to draw grid lines
    - color_mode: Color palette name
    - blur_radius: Gaussian blur for soft effects
    - scanlines: CRT-style scanlines overlay
    """
    start_time = time.time()
    metadata = ProcessingMetadata()
    
    # =========================================================================
    # COMPOSITION MODE SETUP
    # =========================================================================
    composition_mode = composition is not None and len(composition) > 0
    preset_cache: dict[str, dict] = {}
    
    if composition_mode:
        # Build preset cache for all effects in composition
        effect_names = []
        for seg in composition:
            eff_id = seg.get("effect_id", "clean")
            if eff_id != "clean" and eff_id not in preset_cache:
                preset_cache[eff_id] = validate_preset(get_preset(eff_id))
            effect_names.append(eff_id)
        
        preset_name = f"composition({','.join(effect_names)})"
        # Use first non-clean effect as primary config for tracking params
        primary_effect = next((e for e in effect_names if e != "clean"), "grid_trace")
        preset_config = preset_cache.get(primary_effect, validate_preset(get_preset("grid_trace")))
        
        metadata.composition_mode = True
        metadata.preset_used = preset_name
    else:
        # Single effect mode
        if isinstance(preset, str):
            preset_config = validate_preset(get_preset(preset))
            preset_name = preset
        else:
            preset_config = validate_preset(preset)
            preset_name = preset_config.get("name", "custom")
        
        preset_cache[preset_name] = preset_config
        metadata.preset_used = preset_name
    
    # =========================================================================
    # STEP 1: Open input video
    # =========================================================================
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    metadata.duration_seconds = total_frames / fps if fps > 0 else 0
    
    print(f"[process] Input: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"[process] Preset: {preset_name}, overlay_mode: {overlay_mode}")
    
    # =========================================================================
    # STEP 2: Extract and analyze audio
    # =========================================================================
    print("[process] Extracting audio...")
    audio_path, _ = extract_audio(input_path)
    
    print("[process] Analyzing audio for beats...")
    audio_data = analyze_audio(audio_path, fps)
    metadata.beats_detected = len(audio_data["beat_frames"])
    
    # Determine spawn frames
    spawn_rate = max(10, int(fps / 2))  # Fallback: ~2 spawns per second
    spawn_frames = get_spawn_frames(audio_data, total_frames, spawn_rate)
    print(f"[process] Found {metadata.beats_detected} beats, {len(spawn_frames)} spawn frames")
    
    # =========================================================================
    # STEP 3: Setup output video writer(s)
    # =========================================================================
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_path}")
    
    # Optional: also save a re-encoded copy of original for alternating playback
    out_original = None
    if original_output_path:
        out_original = cv2.VideoWriter(original_output_path, fourcc, fps, (width, height))
        if out_original.isOpened():
            print(f"[process] Also saving original to: {original_output_path}")
        else:
            out_original.release()  # Release failed writer before discarding
            out_original = None  # Fallback: don't save original if failed
    
    # =========================================================================
    # STEP 4: Initialize tracker with preset config
    # =========================================================================
    tracker = PointTracker(preset_config)
    spawn_count = preset_config.get("spawn_per_beat", 30)
    
    # =========================================================================
    # STEP 4b: Initialize face detector if preset requests it
    # =========================================================================
    face_detector = None
    needs_faces = preset_config.get("detect_faces", False)
    needs_mesh = preset_config.get("detect_mesh", False)
    
    # In composition mode, check all presets for face detection
    if composition_mode:
        for p in preset_cache.values():
            needs_faces = needs_faces or p.get("detect_faces", False)
            needs_mesh = needs_mesh or p.get("detect_mesh", False)
    
    if needs_faces or needs_mesh:
        print("[process] Initializing face detection...")
        face_detector = FaceDetector(
            detect_faces=needs_faces,
            detect_mesh=needs_mesh,
        )
    
    # =========================================================================
    # STEP 5: Main processing loop
    # =========================================================================
    prev_gray = None
    frame_idx = 0
    total_points_tracked = 0
    total_faces_detected = 0
    
    print("[process] Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Spawn new points on beat/spawn frames ---
        if frame_idx in spawn_frames:
            spawned = tracker.spawn_points(gray, frame_idx, spawn_count)
            metadata.total_points_spawned += spawned
        
        # --- Track existing points ---
        if prev_gray is not None:
            alive_count = tracker.update(prev_gray, gray, frame_idx, (height, width))
            total_points_tracked += alive_count
        
        # --- Detect faces if enabled ---
        face_data = None
        if face_detector:
            face_data = face_detector.detect(frame)
            total_faces_detected = max(total_faces_detected, face_data["face_count"])
        
        # --- Draw the frame ---
        if composition_mode:
            # Multi-effect composition
            frame_ratio = frame_idx / max(total_frames - 1, 1)
            effects = get_frame_effects(frame_ratio, composition, crossfade_ratio)
            output_frame = apply_composed_frame(
                frame, effects, preset_cache, frame_idx, tracker, face_data, overlay_mode
            )
        else:
            # Single effect mode
            all_points = tracker.get_all_points()
            output_frame = draw_frame(
                frame, all_points, preset_config, frame_idx, overlay_mode,
                face_data=face_data
            )
        
        # --- Write output ---
        out.write(output_frame)
        
        # Also write original frame if requested
        if out_original is not None:
            out_original.write(frame)
        
        # --- Update state ---
        prev_gray = gray.copy()
        frame_idx += 1
        
        # Progress logging (every 30 frames)
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            stats = tracker.get_stats()
            print(f"[process] {progress:.0f}% - {stats['alive']} active points")
    
    # =========================================================================
    # STEP 6: Cleanup and finalize
    # =========================================================================
    cap.release()
    out.release()
    
    if out_original is not None:
        out_original.release()
    
    if face_detector:
        face_detector.close()
    
    # Basic stats
    metadata.frames_processed = frame_idx
    metadata.average_points_per_frame = (
        total_points_tracked / frame_idx if frame_idx > 0 else 0
    )
    metadata.processing_time_seconds = time.time() - start_time
    metadata.output_path = output_path
    metadata.original_path = original_output_path or ""
    
    # Composition mode: record which segments were applied
    if composition_mode and composition:
        for seg in composition:
            start_frame = int(seg["start"] * frame_idx)
            end_frame = int(seg["end"] * frame_idx)
            metadata.segments_applied.append({
                "effect": seg["effect_id"],
                "start_frame": start_frame,
                "end_frame": end_frame,
            })
    
    # =========================================================================
    # STEP 7: Compute surveillance stats
    # =========================================================================
    surveillance_stats = tracker.get_surveillance_stats(
        total_frames=frame_idx,
        fps=fps,
        max_possible_points=preset_config.get("max_points", 150),
        total_points_spawned=metadata.total_points_spawned,
    )
    
    metadata.max_continuous_tracking_frames = surveillance_stats["max_continuous_tracking_frames"]
    metadata.longest_track_seconds = surveillance_stats["longest_track_seconds"]
    
    # For text_mode effects (no point tracking), estimate trackability differently
    if preset_config.get("text_mode") and metadata.total_points_spawned == 0:
        # Base trackability on faces detected and frame content
        if total_faces_detected > 0:
            # Face detection gives high trackability
            metadata.trackability_score = min(85, 50 + total_faces_detected * 15)
            metadata.people_detected = total_faces_detected
        else:
            # Estimate based on processed frames (assume some detection happened)
            metadata.trackability_score = random.randint(35, 65)  # Moderate trackability
            metadata.people_detected = 1
        metadata.total_points_spawned = frame_idx * 10  # Fake stat for display
    else:
        metadata.trackability_score = surveillance_stats["trackability_score"]
        # People detected: use face detection count if available
        if face_detector and total_faces_detected > 0:
            metadata.people_detected = total_faces_detected
        else:
            metadata.people_detected = 1 if metadata.total_points_spawned > 0 else 0
    
    print(f"[process] Done! {frame_idx} frames in {metadata.processing_time_seconds:.1f}s")
    print(f"[process] Trackability: {metadata.trackability_score}/100, longest track: {metadata.longest_track_seconds:.1f}s")
    print(f"[process] Output: {output_path}")
    
    return metadata


# =============================================================================
# CLI for testing
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m app.services.process_video <input> <output> [preset]")
        print("\nAvailable presets:")
        from .presets import list_presets
        for p in list_presets():
            print(f"  - {p['id']}: {p['description']}")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    preset_name = sys.argv[3] if len(sys.argv) > 3 else "grid_trace"
    
    result = process_video(input_file, output_file, preset_name)
    print("\nMetadata:")
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")
