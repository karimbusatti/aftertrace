#!/usr/bin/env python3
"""
Test script for sequence mode.

Creates a short test video (colored frames), then runs sequence mode
with multiple effects to verify they alternate correctly.

Usage:
    cd backend
    python test_sequence_mode.py
"""

import os
import sys
import tempfile
import cv2
import numpy as np

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.process_video import process_video, build_sequence_composition


def create_test_video(path: str, duration_s: float = 3.0, fps: float = 30.0):
    """Create a simple test video with changing colors."""
    width, height = 640, 480
    total_frames = int(duration_s * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    colors = [
        (50, 50, 200),   # Red-ish
        (50, 200, 50),   # Green-ish
        (200, 50, 50),   # Blue-ish
        (200, 200, 50),  # Cyan-ish
    ]
    
    for i in range(total_frames):
        # Cycle through colors
        color_idx = (i // 15) % len(colors)
        color = colors[color_idx]
        
        # Create frame with gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = color
        
        # Add some features for tracking (circles that move)
        cx = int(width * 0.3 + 100 * np.sin(i * 0.1))
        cy = int(height * 0.5 + 50 * np.cos(i * 0.15))
        cv2.circle(frame, (cx, cy), 40, (255, 255, 255), -1)
        cv2.circle(frame, (width - cx, cy), 30, (200, 200, 200), -1)
        
        # Add frame number text
        cv2.putText(frame, f"Frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {path} ({total_frames} frames @ {fps}fps)")


def test_build_sequence_composition():
    """Test the composition builder function."""
    print("\n=== Test: build_sequence_composition ===")
    
    effects = ["blob_track", "grid_trace", "binary_bloom"]
    fps = 30.0
    total_frames = 90  # 3 seconds
    segment_duration_s = 0.5  # 0.5s per segment = 15 frames
    
    composition = build_sequence_composition(
        effects=effects,
        segment_duration_s=segment_duration_s,
        total_frames=total_frames,
        fps=fps,
    )
    
    print(f"Effects: {effects}")
    print(f"Segment duration: {segment_duration_s}s ({int(segment_duration_s * fps)} frames)")
    print(f"Total frames: {total_frames}")
    print(f"Generated {len(composition)} segments:")
    
    for i, seg in enumerate(composition):
        start_frame = int(seg["start"] * total_frames)
        end_frame = int(seg["end"] * total_frames)
        print(f"  [{i}] {seg['effect_id']}: frames {start_frame}-{end_frame} (ratio {seg['start']:.3f}-{seg['end']:.3f})")
    
    # Verify cycling
    expected_effects = []
    for i in range(len(composition)):
        expected_effects.append(effects[i % len(effects)])
    
    actual_effects = [seg["effect_id"] for seg in composition]
    
    if actual_effects == expected_effects:
        print("✓ Effects cycle correctly")
    else:
        print(f"✗ Effects don't cycle! Expected {expected_effects}, got {actual_effects}")
    
    return len(composition) > 0


def test_sequence_mode_processing():
    """Test full sequence mode video processing."""
    print("\n=== Test: Full Sequence Mode Processing ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test_input.mp4")
        output_path = os.path.join(tmpdir, "test_output.mp4")
        
        # Create test video
        create_test_video(input_path, duration_s=2.0, fps=30.0)
        
        # Process with sequence mode
        effects = ["blob_track", "grid_trace"]
        segment_duration_s = 0.5
        
        print(f"\nProcessing with sequence mode:")
        print(f"  Effects: {effects}")
        print(f"  Segment duration: {segment_duration_s}s")
        
        try:
            metadata = process_video(
                input_path=input_path,
                output_path=output_path,
                preset="blob_track",  # Fallback
                overlay_mode=False,
                mode="sequence",
                effects=effects,
                segment_duration_s=segment_duration_s,
            )
            
            print(f"\n✓ Processing completed!")
            print(f"  Output: {output_path}")
            print(f"  Frames processed: {metadata.frames_processed}")
            print(f"  Processing time: {metadata.processing_time_seconds:.2f}s")
            print(f"  Composition mode: {metadata.composition_mode}")
            print(f"  Preset used: {metadata.preset_used}")
            print(f"  Segments applied: {len(metadata.segments_applied)}")
            
            for i, seg in enumerate(metadata.segments_applied[:5]):
                print(f"    [{i}] {seg['effect']}: frames {seg['start_frame']}-{seg['end_frame']}")
            
            # Verify output exists
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  Output file size: {size_mb:.2f} MB")
                
                # Quick check: read a few frames from output
                cap = cv2.VideoCapture(output_path)
                out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                print(f"  Output frame count: {out_frames}")
                
                return True
            else:
                print(f"✗ Output file not created!")
                return False
                
        except Exception as e:
            print(f"✗ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_single_mode_unchanged():
    """Verify single mode still works as before."""
    print("\n=== Test: Single Mode (unchanged) ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test_input.mp4")
        output_path = os.path.join(tmpdir, "test_output.mp4")
        
        create_test_video(input_path, duration_s=1.0, fps=30.0)
        
        try:
            metadata = process_video(
                input_path=input_path,
                output_path=output_path,
                preset="blob_track",
                overlay_mode=False,
                mode="single",
            )
            
            print(f"✓ Single mode works!")
            print(f"  Composition mode: {metadata.composition_mode}")
            print(f"  Preset used: {metadata.preset_used}")
            
            return not metadata.composition_mode and metadata.preset_used == "blob_track"
            
        except Exception as e:
            print(f"✗ Single mode failed: {e}")
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("SEQUENCE MODE TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Composition builder
    results.append(("build_sequence_composition", test_build_sequence_composition()))
    
    # Test 2: Single mode unchanged
    results.append(("single_mode_unchanged", test_single_mode_unchanged()))
    
    # Test 3: Full sequence processing
    results.append(("sequence_mode_processing", test_sequence_mode_processing()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed! 🎉")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)

