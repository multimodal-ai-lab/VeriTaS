"""Media handling utilities for fact-checking baselines."""

import base64
import io
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ezmm import Item


# MIME type mapping for images
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


@dataclass(frozen=True)
class MediaLimits:
    """Which images and videos the model gets in a fact-check, and in what form.

    Every model call sends the whole conversation again, with the media of all pages fetched so
    far: the totals keep it within the APIs' request limits (Anthropic: 100 images per request,
    and at most 2000 px per side once there are more than 20) and the context window. Each
    field is a command line option of run_benchmark.py; its help is the field's metadata.
    """

    page_images: int = field(default=10, metadata={
        "help": "Images shown per fetched page, the first ones in page order"})
    page_videos: int = field(default=2, metadata={
        "help": "Videos shown per fetched page, the first ones in page order"})
    total_images: int = field(default=40, metadata={
        "help": "Images of fetched pages shown in a whole fact-check"})
    total_videos: int = field(default=4, metadata={
        "help": "Videos of fetched pages shown in a whole fact-check"})
    request_images: int | None = field(default=None, metadata={
        "help": "Images in one request at most, the claim's included and a video counting as "
                "video_frames images: page images and videos beyond them are not shown. For hosts "
                "that take fewer than the limits above add up to, e.g. 64 on OpenRouter's Near AI "
                "(None: no cap)"})
    video_frames: int = field(default=10, metadata={
        "help": "Frames per video for models without native video (all but Gemini)"})
    native_video_seconds: float = field(default=120.0, metadata={
        "help": "Page videos longer than this reach models with native video as frames too: native "
                "video costs tokens by the second. The claim's videos are always native"})
    page_image_side: int = field(default=1024, metadata={
        "help": "Longest side, in pixels, of the page images and of the frames of page videos"})
    claim_image_side: int = field(default=2000, metadata={
        "help": "Longest side, in pixels, of the claim's images and frames: larger ones are scaled "
                "down, the others stay as they are"})


def get_mime_type(path: str | Path) -> str:
    """Get MIME type for an image file."""
    suffix = Path(path).suffix.lower()
    return MIME_TYPES.get(suffix, "image/jpeg")


@dataclass
class Image:
    """An image to pass to the model."""

    mime_type: str
    data: bytes

    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


# An ezMM sequence's parts: text, and ezMM items (images, videos) in their places. A list, not
# a MultimodalSequence, which takes any "<image:7>" in its text (a page's, an error message's)
# for a reference to an item, and fails if there is no such item.
ItemSequence = list[str | Item]


def render(sequence: ItemSequence) -> str:
    """A sequence as text, each item replaced by its reference ("<image:40>"), as ezMM renders a
    MultimodalSequence."""
    return " ".join(part if isinstance(part, str) else part.reference for part in sequence)


def load_image(path: str | Path, max_side: int) -> Image:
    """Read an image file as JPEG, scaled down to `max_side` pixels on its longest side.
    Fetched pages carry images in any format and size; the APIs take a few formats of limited
    size. Raises if the image library can't decode it (SVG, a truncated file). Read from the
    file, not via ezMM, which keeps every image it decodes in memory."""
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        return _jpeg(image, max_side)


def _jpeg(image, max_side: int | None) -> Image:
    """A PIL image as JPEG, scaled down to `max_side` pixels on its longest side if given."""
    image = image.convert("RGB")
    if max_side:
        image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return Image("image/jpeg", buffer.getvalue())


def is_readable_image(path: str | Path, max_side: int) -> bool:
    """Whether load_image can read the image file, by reading it: the image library opens some
    files it can't decode, such as truncated ones."""
    try:
        load_image(path, max_side)
        return True
    except Exception:
        return False


def load_claim_image(path: str | Path, max_side: int) -> Image:
    """Read a claim's image file as it is, or, if it is larger than `max_side` pixels on its
    longest side, as with load_image. Re-encoding would blur the traces of editing a
    fact-checker looks for."""
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        if max(image.size) <= max_side:
            return Image(get_mime_type(path), Path(path).read_bytes())
    return load_image(path, max_side)


def parse_media_references(claim_text: str) -> tuple[str, list[dict]]:
    """
    Parse media references from claim text.

    Args:
        claim_text: Claim text potentially containing <image:ID> or <video:ID> tags.

    Returns:
        Tuple of (clean_text, media_refs) where media_refs is a list of dicts
        with keys: type, id, tag.
    """
    media_refs = []

    for match in re.finditer(r'<image:(\d+)>', claim_text):
        media_refs.append({
            "type": "image",
            "id": match.group(1),
            "tag": match.group(0)
        })

    for match in re.finditer(r'<video:(\d+)>', claim_text):
        media_refs.append({
            "type": "video",
            "id": match.group(1),
            "tag": match.group(0)
        })

    clean_text = re.sub(r'<(image|video):\d+>\s*', '', claim_text).strip()
    return clean_text, media_refs


def resolve_media_path(media_ref: dict, dataset_dir: Path) -> Path | None:
    """
    Resolve media reference to actual file path.

    Args:
        media_ref: Dict with type and id keys.
        dataset_dir: Base directory of the dataset.

    Returns:
        Path to the media file, or None if not found.
    """
    media_type = media_ref["type"]
    media_id = media_ref["id"]

    if media_type == "image":
        extensions = [".jpg", ".jpeg", ".png", ".webp"]
        subdir = "images"
    else:
        extensions = [".mp4", ".m4v", ".avi", ".mov"]
        subdir = "videos"

    for ext in extensions:
        path = dataset_dir / subdir / f"{media_id}{ext}"
        if path.exists():
            return path

    return None


def resolve_media_files(
    media_refs: list[dict],
    dataset_dir: Path,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Resolve media references to file paths.

    Args:
        media_refs: List of media reference dicts.
        dataset_dir: Base directory of the dataset.
        verbose: Whether to print status messages.

    Returns:
        Tuple of (image_paths, video_paths).
    """
    image_paths = []
    video_paths = []

    for ref in media_refs:
        path = resolve_media_path(ref, dataset_dir)
        if path is None:
            if verbose:
                print(f"    Warning: Media file not found for {ref['type']}:{ref['id']}")
            continue

        if ref["type"] == "image":
            image_paths.append(str(path))
            if verbose:
                print(f"    Found image: {path.name}")
        else:
            video_paths.append(str(path))
            if verbose:
                print(f"    Found video: {path.name} (will extract frames)")

    return image_paths, video_paths


def extract_video_frames(video_path: str, max_frames: int = MediaLimits.video_frames,
                         max_side: int | None = None) -> list[Image]:
    """
    Extract frames from a video as JPEG images.

    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to extract.
        max_side: If given, frames are scaled down to this many pixels on their longest side.

    Returns:
        The frames, sampled evenly across the video.
    """
    try:
        import cv2
    except ImportError:
        print("    Warning: opencv-python not installed, skipping video frame extraction")
        return []

    video = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        frames = []
        for i in range(max_frames):
            video.set(cv2.CAP_PROP_POS_FRAMES, int(i * total_frames / max_frames))
            success, frame = video.read()
            if not success and not frames:
                # OpenCV's FFmpeg build lacks some decoders, AV1's among them: FFmpeg itself has them
                fps = video.get(cv2.CAP_PROP_FPS)
                if fps <= 0:  # Unknown length: the first frame only
                    return _frames_with_ffmpeg(video_path, [0.0], max_side)
                seconds = total_frames / fps
                return _frames_with_ffmpeg(video_path, [i * seconds / max_frames for i in range(max_frames)],
                                           max_side)
            if success:
                height, width = frame.shape[:2]
                if max_side and max(height, width) > max_side:
                    scale = max_side / max(height, width)
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                       interpolation=cv2.INTER_AREA)
                encoded, jpeg = cv2.imencode(".jpg", frame)
                if encoded:
                    frames.append(Image("image/jpeg", jpeg.tobytes()))
        return frames
    finally:
        video.release()


def _frames_with_ffmpeg(video_path: str, seconds: list[float], max_side: int | None) -> list[Image]:
    """The frames at the given seconds of a video, as JPEG images, decoded by FFmpeg (the
    build imageio-ffmpeg ships, which ezMM needs too). Frames it can't decode are left out."""
    from imageio_ffmpeg import get_ffmpeg_exe
    from PIL import Image as PILImage

    frames = []
    for second in seconds:
        result = subprocess.run(
            [get_ffmpeg_exe(), "-loglevel", "error", "-ss", f"{second:.3f}", "-i", str(video_path),
             "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout:
            with PILImage.open(io.BytesIO(result.stdout)) as frame:
                frames.append(_jpeg(frame, max_side))
    return frames
