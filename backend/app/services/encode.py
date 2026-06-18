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


def transcode_for_processing(src_path: str, max_edge: int = 1080) -> tuple[str, bool]:
    """
    Normalize ANY uploaded clip into a clean H.264 mp4 that OpenCV can read.

    Phone camera-roll videos are usually HEVC/.mov, which the headless OpenCV
    build often cannot decode (=> "load failed"). ffmpeg decodes essentially
    anything, so we transcode up front and also downscale to `max_edge` here -
    that fixes 4K/HEVC uploads AND slashes memory/CPU (we then process 1080p,
    not 4K). Audio is preserved for beat/envelope analysis.

    Returns (path_to_use, did_transcode). On any failure, returns the original
    path so the pipeline still tries.
    """
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        return src_path, False

    dst = src_path + ".norm.mp4"
    # Fit within a max_edge x max_edge box, preserve aspect, force even dims.
    vf = (
        f"scale={max_edge}:{max_edge}:force_original_aspect_ratio=decrease,"
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    cmd = [
        ffmpeg, "-y",
        "-i", src_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        dst,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if result.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return dst, True
        print(f"[encode] transcode failed (rc={result.returncode}): "
              f"{result.stderr.decode('utf-8', 'ignore')[-400:]}")
    except Exception as e:
        print(f"[encode] transcode error: {e}")
    if os.path.exists(dst):
        try:
            os.unlink(dst)
        except OSError:
            pass
    return src_path, False


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
