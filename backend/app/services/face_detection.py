"""
Face and pose detection using MediaPipe.

Provides face detection, face mesh landmarks, and pose estimation
for enhanced surveillance visualization effects.
"""

import cv2
import numpy as np
from typing import Any

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


def draw_face_boxes(
    frame: np.ndarray,
    faces: list,
    color: tuple = (0, 255, 255),
    thickness: int = 2,
    show_confidence: bool = True,
):
    """
    Draw face bounding boxes with surveillance-style corners.
    
    Args:
        frame: Frame to draw on (modified in-place)
        faces: List of (x, y, w, h, confidence) tuples
        color: BGR color for boxes
        thickness: Line thickness
        show_confidence: Show confidence percentage
    """
    for (x, y, w, h, conf) in faces:
        # Draw corner brackets instead of full rectangle
        corner_len = min(w, h) // 4
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
        
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)
        
        # Confidence label
        if show_confidence:
            label = f"{int(conf * 100)}%"
            cv2.putText(
                frame, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )


def draw_face_mesh(
    frame: np.ndarray,
    mesh_points: list,
    color: tuple = (0, 255, 0),
    draw_contours: bool = True,
    draw_points: bool = False,
):
    """
    Draw face mesh landmarks.
    
    Args:
        frame: Frame to draw on (modified in-place)
        mesh_points: List of [(x, y, z), ...] for each face
        color: BGR color for mesh
        draw_contours: Draw mesh contour lines
        draw_points: Draw individual points
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
    
    for face_points in mesh_points:
        if not face_points:
            continue
        
        if draw_contours:
            # Draw face oval
            _draw_contour(frame, face_points, FACE_OVAL, color)
            # Draw eyes
            _draw_contour(frame, face_points, LEFT_EYE, color)
            _draw_contour(frame, face_points, RIGHT_EYE, color)
            # Draw lips
            _draw_contour(frame, face_points, LIPS_OUTER, color)
        
        if draw_points:
            for i, (x, y, z) in enumerate(face_points):
                # Only draw every 5th point to avoid clutter
                if i % 5 == 0:
                    cv2.circle(frame, (x, y), 1, color, -1)


def _draw_contour(frame, points, indices, color):
    """Draw connected contour from point indices."""
    for i in range(len(indices)):
        idx1 = indices[i]
        idx2 = indices[(i + 1) % len(indices)]
        if idx1 < len(points) and idx2 < len(points):
            pt1 = (points[idx1][0], points[idx1][1])
            pt2 = (points[idx2][0], points[idx2][1])
            cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)


def draw_face_glow(
    frame: np.ndarray,
    faces: list,
    color: tuple = (255, 100, 50),
    intensity: float = 0.5,
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
        axes = (bw // 2 + 20, bh // 2 + 20)
        cv2.ellipse(glow_layer, center, axes, 0, 0, 360, color, -1)
    
    # Heavy blur for soft glow
    glow_layer = cv2.GaussianBlur(glow_layer, (51, 51), 0)
    
    # Blend with frame
    frame[:] = cv2.addWeighted(frame, 1.0, glow_layer, intensity, 0)



