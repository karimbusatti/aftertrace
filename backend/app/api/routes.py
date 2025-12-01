"""
API routes for Aftertrace.
"""

import uuid
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import cv2

from app.services.process_video import process_video
from app.services.presets import list_presets, PRESETS

router = APIRouter()

# Directory for temporary files
TEMP_DIR = Path(tempfile.gettempdir()) / "aftertrace"
TEMP_DIR.mkdir(exist_ok=True)

# Upload limits
MAX_DURATION_SECONDS = 30.0
MAX_DIMENSION = 2160  # Longest edge - 4K (works for both portrait and landscape)


def check_video_limits(video_path: str) -> dict | None:
    """
    Check if video exceeds duration or resolution limits.
    Returns None if OK, or a dict with error details if limits exceeded.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return {"error": "couldn't read your video. try a different format?"}
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
        
        issues = []
        
        if duration > MAX_DURATION_SECONDS:
            issues.append(f"{duration:.0f}s is a bit long")
        
        # Check if longest edge exceeds limit (handles both portrait and landscape)
        longest_edge = max(width, height)
        if longest_edge > MAX_DIMENSION:
            issues.append(f"{width}×{height} is higher res than we need")
        
        if issues:
            return {
                "error": f"this clip is a bit too heavy ({', '.join(issues)}). try something under 20 seconds or 1080p.",
                "validation": True,
                "duration": round(duration, 1),
                "width": width,
                "height": height,
            }
        
        return None
    finally:
        cap.release()


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/presets")
async def get_presets():
    """List available visual presets."""
    return {"presets": list_presets()}


@router.get("/presets/{preset_id}")
async def get_preset_detail(preset_id: str):
    """Get detailed configuration for a specific preset."""
    if preset_id not in PRESETS:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{preset_id}' not found"
        )
    
    preset = PRESETS[preset_id]
    return {
        "id": preset_id,
        "name": preset["name"],
        "description": preset["description"],
        "config": preset,
    }


@router.post("/process")
async def process_video_endpoint(
    file: UploadFile = File(...),
    preset: Optional[str] = Form(default="grid_trace"),
    overlay_mode: bool = Form(default=False),
):
    """
    Process a video with Aftertrace visual effects.
    
    Args:
        file: Video file (mp4, mov, webm, etc.)
        preset: Visual preset name
        overlay_mode: If true, blend effects at 40% over original video
    
    Returns:
        Processing metadata and download info.
    """
    # Validate preset
    if preset not in PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}"
        )
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())[:8]
    
    # Create temp paths
    input_ext = Path(file.filename).suffix if file.filename else ".mp4"
    input_path = TEMP_DIR / f"{job_id}_input{input_ext}"
    output_path = TEMP_DIR / f"{job_id}_output.mp4"
    original_path = TEMP_DIR / f"{job_id}_original.mp4"
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Check video limits before processing
        limit_error = check_video_limits(str(input_path))
        if limit_error:
            input_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=limit_error["error"])
        
        # Process the video (also save re-encoded original for alternating playback)
        metadata = process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            preset=preset,
            overlay_mode=overlay_mode,
            original_output_path=str(original_path),
        )
        
        # Clean up input file
        input_path.unlink(missing_ok=True)
        
        # Check if original was saved successfully
        has_original = original_path.exists()
        
        return {
            "success": True,
            "job_id": job_id,
            "filename": file.filename,
            "preset": preset,
            "overlay_mode": overlay_mode,
            "metadata": metadata.to_dict(),
            "download_url": f"/download/{job_id}",
            "original_download_url": f"/download/{job_id}/original" if has_original else None,
        }
        
    except HTTPException:
        # Let HTTPExceptions propagate (validation errors, etc.)
        raise
    except Exception as e:
        # Clean up on error
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        original_path.unlink(missing_ok=True)
        
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{job_id}/original")
async def download_original_video(job_id: str):
    """Download the original video (for alternating playback comparison)."""
    original_path = TEMP_DIR / f"{job_id}_original.mp4"
    
    if not original_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Original video not found."
        )
    
    return FileResponse(
        path=original_path,
        media_type="video/mp4",
        filename=f"original_{job_id}.mp4",
    )


@router.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download a processed video."""
    output_path = TEMP_DIR / f"{job_id}_output.mp4"
    
    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video not found. It may have expired or the job ID is invalid."
        )
    
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"aftertrace_{job_id}.mp4",
    )


@router.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up temporary files for a job."""
    # Delete output file (always .mp4)
    output_path = TEMP_DIR / f"{job_id}_output.mp4"
    output_path.unlink(missing_ok=True)
    
    # Delete original file (for alternating playback)
    original_path = TEMP_DIR / f"{job_id}_original.mp4"
    original_path.unlink(missing_ok=True)
    
    # Delete input file (could be any video extension)
    # Use glob to find any input file with this job_id
    for input_file in TEMP_DIR.glob(f"{job_id}_input.*"):
        input_file.unlink(missing_ok=True)
    
    return {"status": "cleaned"}
