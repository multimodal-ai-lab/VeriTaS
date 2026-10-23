"""Tests for turning media into what the models get."""

import io
import subprocess

import cv2
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image as PILImage

from eval.baselines.common.media import extract_video_frames


def _video(path, codec: str, size="320x240") -> str:
    """A 2-second test video: 10 frames per second."""
    subprocess.run([get_ffmpeg_exe(), "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=duration=2:size={size}:rate=10",
                    "-c:v", codec, *(["-cpu-used", "8"] if codec == "libaom-av1" else []), "-y", str(path)],
                   check=True)
    return str(path)


def test_frames_of_an_av1_video_which_opencv_cannot_decode(tmp_path):
    """YouTube and Instagram serve AV1, and so do some claims (Q3: one video)."""
    path = _video(tmp_path / "av1.mp4", "libaom-av1", size="640x480")
    assert not cv2.VideoCapture(path).read()[0]

    frames = extract_video_frames(path, 5, max_side=320)

    assert len(frames) == 5
    assert all(frame.mime_type == "image/jpeg" for frame in frames)
    assert PILImage.open(io.BytesIO(frames[0].data)).size == (320, 240)
    assert len({frame.data for frame in frames}) == 5  # Spread over the video


def test_frames_of_an_h264_video(tmp_path):
    frames = extract_video_frames(_video(tmp_path / "h264.mp4", "libx264"), 3)
    assert len(frames) == 3 and PILImage.open(io.BytesIO(frames[0].data)).size == (320, 240)
