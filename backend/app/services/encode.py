"""
Video encoding helpers.

OpenCV's VideoWriter with the "mp4v" fourcc produces MPEG-4 Part 2, which is
large and not reliably playable in browsers (especially Safari). This module
re-encodes those intermediate files to web-optimized H.264 (yuv420p, faststart)
using the ffmpeg binary bundled with imageio-ffmpeg, so no system ffmpeg is
required.

The re-encode is best-effort: if ffmpeg is unavailable or fails, we fall back to
the original file so the pipeline still returns a valid (if heavier) video.
"""

import os
import shutil
import subprocess


def _ffmpeg_exe() -> str | None:
    """Locate an ffmpeg binary (bundled or system). Returns None if unavailable."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def encode_h264(
    src_path: str,
    dst_path: str,
    crf: int = 18,
    preset: str = "faster",
) -> bool:
    """
    Re-encode `src_path` to web-optimized H.264 at `dst_path`.

    Args:
        src_path: Intermediate video (e.g. OpenCV mp4v output).
        dst_path: Final output path (.mp4).
        crf: Constant Rate Factor (lower = higher quality/larger). 18-23 is a
             good visually-lossless-ish range; 20 is a strong default.
        preset: x264 speed/efficiency preset.

    Returns:
        True if the H.264 file was written, False if we fell back to a copy.
    """
    ffmpeg = _ffmpeg_exe()

    if ffmpeg is None:
        # No encoder available - fall back to the intermediate file as-is.
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copyfile(src_path, dst_path)
        return False

    # Encode to a temp file first, then atomically move into place. This avoids
    # leaving a half-written file at dst_path if ffmpeg is interrupted.
    tmp_dst = dst_path + ".enc.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",  # no audio track (OpenCV output is silent anyway)
        tmp_dst,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        if result.returncode == 0 and os.path.exists(tmp_dst) and os.path.getsize(tmp_dst) > 0:
            os.replace(tmp_dst, dst_path)
            return True
        else:
            print(f"[encode] ffmpeg failed (rc={result.returncode}): "
                  f"{result.stderr.decode('utf-8', 'ignore')[-500:]}")
    except Exception as e:
        print(f"[encode] ffmpeg error: {e}")
    finally:
        # Clean up temp file if it survived a failure.
        if os.path.exists(tmp_dst):
            try:
                os.unlink(tmp_dst)
            except OSError:
                pass

    # Fallback: keep the intermediate file as the final output.
    if os.path.abspath(src_path) != os.path.abspath(dst_path):
        try:
            shutil.copyfile(src_path, dst_path)
        except OSError:
            pass
    return False
