
import cv2
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.effects import draw_signal_bloom, draw_vector_signal
from app.services.presets import get_preset

def test_effects():
    print("Initializing test...")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Test Signal Bloom
    print("Testing signal_bloom...")
    try:
        preset = get_preset("signal_bloom")
        # Run multiple frames to check for temporal bugs
        for i in range(10):
            _ = draw_signal_bloom(frame.copy(), preset, {}, frame_idx=i)
        print("✓ signal_bloom passed")
    except Exception as e:
        print(f"❌ signal_bloom FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test Vector Signal
    print("Testing vector_signal...")
    try:
        preset = get_preset("vector_signal")
        # Mock tracked points
        from app.services.types import TrackedPoint
        points = [
            TrackedPoint(id=1, position=np.array([100.0, 100.0]), birth_frame=0),
            TrackedPoint(id=2, position=np.array([200.0, 200.0]), birth_frame=0),
            TrackedPoint(id=3, position=np.array([150.0, 300.0]), birth_frame=0),
        ]
        
        for i in range(10):
            _ = draw_vector_signal(frame.copy(), preset, {}, frame_idx=i, points=points)
        print("✓ vector_signal passed")
    except Exception as e:
        print(f"❌ vector_signal FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_effects()
