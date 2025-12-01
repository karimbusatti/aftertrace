"""
Face and pose detection using MediaPipe.

Provides face detection, face mesh landmarks, and pose estimation
for enhanced surveillance visualization effects.
"""

import cv2
import numpy as np
from typing import Any
import time

# MediaPipe imports - handle gracefully if not installed
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


class FaceDetector:
    """
    MediaPipe-based face detection with mesh landmarks.
    Falls back gracefully if MediaPipe isn't available.
    """
    
    def __init__(self, 
                 detect_faces: bool = True,
                 detect_mesh: bool = False,
                 min_detection_confidence: float = 0.5):
        """
        Initialize the face detector.
        
        Args:
            detect_faces: Enable face bounding box detection
            detect_mesh: Enable 468-point face mesh landmarks
            min_detection_confidence: Minimum confidence for detection
        """
        self.detect_faces = detect_faces
        self.detect_mesh = detect_mesh
        self._face_detection = None
        self._face_mesh = None
        
        if not MEDIAPIPE_AVAILABLE:
            print("[face_detection] MediaPipe not available, using fallback")
            return
        
        if detect_faces:
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # 0 for close-range, 1 for full-range
                min_detection_confidence=min_detection_confidence
            )
        
        if detect_mesh:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
    
    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Detect faces and landmarks in a frame.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            dict with:
                - faces: list of face bounding boxes [(x, y, w, h, confidence), ...]
                - mesh_points: list of face mesh landmarks per face
                - face_count: number of faces detected
        """
        result = {
            "faces": [],
            "mesh_points": [],
            "face_count": 0,
        }
        
        if not MEDIAPIPE_AVAILABLE:
            return result
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Face detection (bounding boxes)
        if self._face_detection:
            detection_result = self._face_detection.process(rgb_frame)
            if detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    box_w = int(bbox.width * w)
                    box_h = int(bbox.height * h)
                    confidence = detection.score[0] if detection.score else 0.5
                    result["faces"].append((x, y, box_w, box_h, confidence))
        
        # Face mesh (468 landmarks per face)
        if self._face_mesh:
            mesh_result = self._face_mesh.process(rgb_frame)
            if mesh_result.multi_face_landmarks:
                for face_landmarks in mesh_result.multi_face_landmarks:
                    points = []
                    for landmark in face_landmarks.landmark:
                        px = int(landmark.x * w)
                        py = int(landmark.y * h)
                        pz = landmark.z  # depth
                        points.append((px, py, pz))
                    result["mesh_points"].append(points)
        
        result["face_count"] = max(len(result["faces"]), len(result["mesh_points"]))
        return result
    
    def close(self):
        """Release resources."""
        if self._face_detection:
            self._face_detection.close()
        if self._face_mesh:
            self._face_mesh.close()


def draw_cctv_overlay(
    frame: np.ndarray,
    frame_idx: int,
    fps: float = 30.0,
):
    """
    Draw CCTV-style overlay with timestamp, camera ID, and recording indicator.
    
    Args:
        frame: Frame to draw on (modified in-place)
        frame_idx: Current frame index for timestamp
        fps: Video FPS for accurate timestamp
    """
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay areas
    overlay = frame.copy()
    
    # Top bar - dark gradient
    cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
    frame[:35] = cv2.addWeighted(frame[:35], 0.4, overlay[:35], 0.6, 0)
    
    # Camera ID (top left)
    cam_id = "CAM-01"
    cv2.putText(
        frame, cam_id, (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    
    # Timestamp (top right)
    seconds = frame_idx / fps
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # Add frame-based time offset for realism
    ms = int((seconds % 1) * 1000)
    timestamp_full = f"{timestamp}.{ms:03d}"
    
    text_size = cv2.getTextSize(timestamp_full, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(
        frame, timestamp_full, (w - text_size[0] - 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    
    # REC indicator (pulsing)
    if (frame_idx // 15) % 2 == 0:  # Pulse every ~0.5s at 30fps
        cv2.circle(frame, (w - 20, 55), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame, "REC", (w - 55, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
        )
    
    # Bottom info bar
    cv2.rectangle(overlay, (0, h - 25), (w, h), (0, 0, 0), -1)
    frame[h-25:] = cv2.addWeighted(frame[h-25:], 0.4, overlay[h-25:], 0.6, 0)
    
    # Analysis mode indicator
    cv2.putText(
        frame, "ANALYSIS: ACTIVE", (12, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA
    )
    
    # Frame counter
    frame_text = f"F:{frame_idx:06d}"
    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    cv2.putText(
        frame, frame_text, (w - text_size[0] - 12, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA
    )


def draw_face_boxes(
    frame: np.ndarray,
    faces: list,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
    show_confidence: bool = True,
    frame_idx: int = 0,
    style: str = "minimal",
):
    """
    Draw face bounding boxes - minimal clean style.
    
    Args:
        frame: Frame to draw on (modified in-place)
        faces: List of (x, y, w, h, confidence) tuples
        color: BGR color for boxes (default white)
        thickness: Line thickness
        show_confidence: Show confidence percentage
        frame_idx: Current frame for animations
        style: "minimal" for clean boxes, "cctv" for corner brackets
    """
    h_frame, w_frame = frame.shape[:2]
    
    for idx, (x, y, w, h, conf) in enumerate(faces):
        # Clamp to frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if style == "minimal":
            # Clean white rectangle - TouchDesigner style
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
            
            # Calculate normalized coordinates
            cx = (x + w / 2) / w_frame
            cy = (y + h / 2) / h_frame
            
            # Minimal coordinate label
            label = f"x:{cx:.3f} y:{cy:.3f}"
            cv2.putText(
                frame, label, (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA
            )
            
            # Small center marker
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.drawMarker(frame, (center_x, center_y), color, cv2.MARKER_CROSS, 8, 1)
            
        elif style == "cctv":
            # Corner brackets style
            corner_len = max(min(w, h) // 4, 12)
            _draw_corner_brackets(frame, x, y, w, h, corner_len, color, thickness)
            
            # Subject label
            subject_label = f"SUBJ-{idx + 1:02d}"
            cv2.putText(
                frame, subject_label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )
            
        else:
            # Full rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
            
            # Confidence label
            if show_confidence:
                conf_text = f"{int(conf * 100)}%"
                cv2.putText(
                    frame, conf_text, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
                )


def _draw_corner_brackets(frame, x, y, w, h, corner_len, color, thickness):
    """Draw the four corner brackets of a detection box."""
    # Top-left corner
    cv2.line(frame, (x, y), (x + corner_len, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x, y + corner_len), color, thickness, cv2.LINE_AA)
    
    # Top-right corner
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness, cv2.LINE_AA)
    
    # Bottom-left corner
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness, cv2.LINE_AA)
    
    # Bottom-right corner
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness, cv2.LINE_AA)


def draw_face_mesh(
    frame: np.ndarray,
    mesh_points: list,
    color: tuple = (0, 255, 0),
    draw_contours: bool = True,
    draw_points: bool = False,
    glow: bool = True,
):
    """
    Draw face mesh landmarks with optional glow effect.
    
    Args:
        frame: Frame to draw on (modified in-place)
        mesh_points: List of [(x, y, z), ...] for each face
        color: BGR color for mesh
        draw_contours: Draw mesh contour lines
        draw_points: Draw individual points
        glow: Add glow effect to mesh
    """
    if not MEDIAPIPE_AVAILABLE:
        return
    
    # MediaPipe face mesh connection indices for contours
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
                  270, 269, 267, 0, 37, 39, 40, 185]
    
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                159, 160, 161, 246]
    
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388,
                 387, 386, 385, 384, 398]
    
    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    
    NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
    
    # Create glow layer if enabled
    if glow:
        glow_layer = np.zeros_like(frame)
    
    for face_points in mesh_points:
        if not face_points:
            continue
        
        if draw_contours:
            # Draw all contours
            contours = [FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS_OUTER, 
                       LEFT_EYEBROW, RIGHT_EYEBROW, NOSE]
            
            for contour in contours:
                if glow:
                    _draw_contour(glow_layer, face_points, contour, color, thickness=2)
                _draw_contour(frame, face_points, contour, color, thickness=1)
        
        if draw_points:
            for i, (x, y, z) in enumerate(face_points):
                # Only draw every 3rd point to avoid clutter
                if i % 3 == 0:
                    if glow:
                        cv2.circle(glow_layer, (x, y), 2, color, -1)
                    cv2.circle(frame, (x, y), 1, color, -1)
    
    # Apply glow
    if glow and mesh_points:
        glow_layer = cv2.GaussianBlur(glow_layer, (15, 15), 0)
        frame[:] = cv2.addWeighted(frame, 1.0, glow_layer, 0.5, 0)


def _draw_contour(frame, points, indices, color, thickness=1):
    """Draw connected contour from point indices."""
    for i in range(len(indices)):
        idx1 = indices[i]
        idx2 = indices[(i + 1) % len(indices)]
        if idx1 < len(points) and idx2 < len(points):
            pt1 = (points[idx1][0], points[idx1][1])
            pt2 = (points[idx2][0], points[idx2][1])
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_face_glow(
    frame: np.ndarray,
    faces: list,
    color: tuple = (0, 255, 80),
    intensity: float = 0.4,
):
    """
    Draw a soft glow around detected faces.
    
    Args:
        frame: Frame to draw on (modified in-place)
        faces: List of (x, y, w, h, confidence) tuples
        color: BGR color for glow
        intensity: Glow intensity (0-1)
    """
    if not faces:
        return
    
    h, w = frame.shape[:2]
    glow_layer = np.zeros_like(frame)
    
    for (x, y, bw, bh, conf) in faces:
        # Draw filled ellipse for face glow
        center = (x + bw // 2, y + bh // 2)
        axes = (bw // 2 + 30, bh // 2 + 30)
        cv2.ellipse(glow_layer, center, axes, 0, 0, 360, color, -1)
    
    # Heavy blur for soft glow
    glow_layer = cv2.GaussianBlur(glow_layer, (61, 61), 0)
    
    # Blend with frame
    frame[:] = cv2.addWeighted(frame, 1.0, glow_layer, intensity, 0)


def draw_biometric_data(
    frame: np.ndarray,
    faces: list,
    mesh_points: list,
    frame_idx: int,
    color: tuple = (0, 255, 0),
):
    """
    Draw advanced biometric analysis overlay.
    Shows fake biometric data for surveillance aesthetic.
    
    Args:
        frame: Frame to draw on (modified in-place)
        faces: List of face bounding boxes
        mesh_points: List of face mesh landmarks
        frame_idx: Current frame for animations
        color: BGR color for overlay
    """
    h, w = frame.shape[:2]
    
    for idx, (x, y, bw, bh, conf) in enumerate(faces):
        # Data panel background (right side of face)
        panel_x = min(x + bw + 10, w - 150)
        panel_y = max(y, 10)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 140, panel_y + 120), 
                     (0, 0, 0), -1)
        frame[panel_y:panel_y+120, panel_x:panel_x+140] = cv2.addWeighted(
            frame[panel_y:panel_y+120, panel_x:panel_x+140], 0.3,
            overlay[panel_y:panel_y+120, panel_x:panel_x+140], 0.7, 0
        )
        
        # Fake biometric data
        metrics = [
            ("FACE-ID", f"#{idx+1:04d}"),
            ("CONF", f"{conf*100:.1f}%"),
            ("DIST", f"{np.random.uniform(1.5, 4.5):.1f}m"),
            ("POSE", f"{np.random.randint(-15, 15)}°"),
            ("EYE-D", f"{np.random.uniform(55, 70):.1f}mm"),
        ]
        
        for i, (label, value) in enumerate(metrics):
            y_pos = panel_y + 15 + i * 20
            cv2.putText(frame, label, (panel_x + 5, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
            cv2.putText(frame, value, (panel_x + 70, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        
        # Scanning animation at bottom
        scan_progress = (frame_idx % 60) / 60.0
        bar_width = int(130 * scan_progress)
        cv2.rectangle(frame, (panel_x + 5, panel_y + 110), 
                     (panel_x + 5 + bar_width, panel_y + 115), color, -1)
