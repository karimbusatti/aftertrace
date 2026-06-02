"""
Person segmentation using MediaPipe Selfie Segmentation.

Returns a soft person mask so effects can isolate the subject from the
background. Degrades gracefully (returns None) when MediaPipe is unavailable,
letting callers fall back to a heuristic mask.
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


_segmenter = None
_segmenter_failed = False


def _get_segmenter():
    """Lazily create and cache the selfie segmenter (reused across frames)."""
    global _segmenter, _segmenter_failed
    if not MEDIAPIPE_AVAILABLE or _segmenter_failed:
        return None
    if _segmenter is None:
        try:
            _segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 1 = general (full body), 0 = landscape/close
            )
        except Exception as e:
            print(f"[segmentation] could not init selfie segmentation: {e}")
            _segmenter_failed = True
            return None
    return _segmenter


def get_person_mask(frame: np.ndarray) -> np.ndarray | None:
    """
    Return a single-channel uint8 mask (0-255) of the person in the frame, or
    None if segmentation is unavailable.
    """
    seg = _get_segmenter()
    if seg is None:
        return None
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = seg.process(rgb)
        if result.segmentation_mask is None:
            return None

        prob = np.clip(result.segmentation_mask, 0.0, 1.0).astype(np.float32)

        # Sharpen the soft probability around the 0.5 boundary so the silhouette
        # is decisive but edges (hair, fingers) stay feathered rather than jagged.
        prob = np.clip((prob - 0.5) * 3.0 + 0.5, 0.0, 1.0)
        mask = (prob * 255.0).astype(np.uint8)

        # Close small holes inside the body, then feather the boundary.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        return mask
    except Exception as e:
        print(f"[segmentation] mask failed: {e}")
        return None
