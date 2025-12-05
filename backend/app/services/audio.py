"""
Audio processing module.
Extracts audio from video and detects beats/onsets using Librosa.
"""

import os
import tempfile
import numpy as np
import librosa
import cv2

# moviepy is noisy on import, suppress it
import logging
logging.getLogger("moviepy").setLevel(logging.ERROR)

# Handle moviepy import gracefully - may not be available on all platforms
try:
from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    VideoFileClip = None


def _get_duration_opencv(video_path: str) -> float:
    """Get video duration using OpenCV as fallback."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    cap.release()
    return duration


def extract_audio(video_path: str) -> tuple[str | None, float]:
    """
    Extract audio track from video file.
    
    Returns:
        (audio_path, duration) - path to temp WAV file, video duration in seconds.
        audio_path is None if video has no audio.
    """
    if not MOVIEPY_AVAILABLE:
        print("[audio] MoviePy not available, cannot extract audio")
        return None, _get_duration_opencv(video_path)
    
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        
        if clip.audio is None:
            clip.close()
            return None, duration
        
        # Write to temp file
        audio_path = tempfile.mktemp(suffix=".wav")
        clip.audio.write_audiofile(audio_path, logger=None, verbose=False)
        clip.close()
        
        return audio_path, duration
    except Exception as e:
        print(f"[audio] Failed to extract audio: {e}")
        return None, _get_duration_opencv(video_path)


def analyze_audio(audio_path: str | None, fps: float) -> dict:
    """
    Analyze audio to detect beats and onsets.
    
    Args:
        audio_path: Path to audio file (WAV). None if no audio.
        fps: Video frame rate, used to convert times to frame indices.
    
    Returns:
        dict with:
            - beat_frames: list of frame indices where beats occur
            - onset_frames: list of frame indices where onsets occur
            - tempo: estimated BPM
    """
    if audio_path is None or not os.path.exists(audio_path):
        return {"beat_frames": [], "onset_frames": [], "tempo": 0.0}
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Detect tempo and beats
        tempo, beat_frames_audio = librosa.beat.beat_track(y=y, sr=sr)
        
        # Convert beat frames (in audio samples) to video frames
        beat_times = librosa.frames_to_time(beat_frames_audio, sr=sr)
        beat_frames = [int(t * fps) for t in beat_times]
        
        # Detect onsets (transients - more granular than beats)
        onset_frames_audio = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames_audio, sr=sr)
        onset_frames = [int(t * fps) for t in onset_times]
        
        return {
            "beat_frames": beat_frames,
            "onset_frames": onset_frames,
            "tempo": float(tempo) if np.isscalar(tempo) else float(tempo[0]),
        }
    except Exception as e:
        print(f"[audio] Analysis failed: {e}")
        return {"beat_frames": [], "onset_frames": [], "tempo": 0.0}
    finally:
        # Always clean up temp audio file
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def get_spawn_frames(audio_data: dict, total_frames: int, spawn_rate: int = 15) -> set[int]:
    """
    Determine which frames should spawn new tracking points.
    
    Uses beat frames if available, otherwise falls back to regular intervals.
    
    Args:
        audio_data: Result from analyze_audio()
        total_frames: Total number of frames in video
        spawn_rate: Fallback spawn interval if no audio
    
    Returns:
        Set of frame indices where points should spawn.
    """
    if audio_data["beat_frames"]:
        # Use beats, plus occasional onsets for variation
        frames = set(audio_data["beat_frames"])
        # Add every 3rd onset for extra density
        frames.update(audio_data["onset_frames"][::3])
        return frames
    else:
        # No audio - spawn at regular intervals
        return set(range(0, total_frames, spawn_rate))
