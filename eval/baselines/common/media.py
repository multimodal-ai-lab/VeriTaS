"""Media handling utilities for fact-checking baselines."""

import base64
import re
import tempfile
from pathlib import Path


# MIME type mapping for images
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def get_mime_type(path: str | Path) -> str:
    """Get MIME type for an image file."""
    suffix = Path(path).suffix.lower()
    return MIME_TYPES.get(suffix, "image/jpeg")


def encode_image_base64(image_path: str | Path) -> str | None:
    """
    Encode an image file to base64 data URI.

    Args:
        image_path: Path to image file.

    Returns:
        Base64 data URI string, or None if file doesn't exist.
    """
    path = Path(image_path)
    if not path.exists():
        return None

    mime_type = get_mime_type(path)

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


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


def extract_video_frames(video_path: str, max_frames: int = 5) -> list[str]:
    """
    Extract frames from a video and save as temporary images.

    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of paths to extracted frame images.
    """
    try:
        import cv2
    except ImportError:
        print("    Warning: opencv-python not installed, skipping video frame extraction")
        return []

    video = cv2.VideoCapture(video_path)
    frame_paths = []

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        video.release()
        return []

    # Sample frames evenly across the video
    frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            # Save frame to temp file
            fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            cv2.imwrite(temp_path, frame)
            frame_paths.append(temp_path)

    video.release()
    return frame_paths
