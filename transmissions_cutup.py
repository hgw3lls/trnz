"""
Generative video editor for the TRANSMISSIONS project.

This module analyses an audio track, derives visual edit instructions, and renders
three synchronized video channels representing different facets of the work.

Usage:
    python transmissions_cutup.py --track track01 [--mask]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import librosa
import moviepy.editor as mpy
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
METADATA_DIR = BASE_DIR / "metadata"
OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_SAMPLE_EVERY_SEC = 0.5

# Face-specific configuration guiding edit logic.
FACE_CONFIG: Dict[str, Dict[str, Any]] = {
    "A": {
        "beat_divisor": 1,
        "segment_duration": 0.5,
        "rms_high": 0.2,
        "centroid_high": 3500.0,
        "stutter_prob": 0.0,
        "black_frame_prob": 0.0,
    },
    "B": {
        "beat_divisor": 2,
        "segment_duration": 1.0,
        "rms_high": 0.15,
        "centroid_high": 2500.0,
        "stutter_prob": 0.05,
        "black_frame_prob": 0.02,
    },
    "C": {
        "beat_divisor": 1,
        "segment_duration": 0.25,
        "rms_high": 0.1,
        "centroid_high": 2200.0,
        "stutter_prob": 0.25,
        "black_frame_prob": 0.1,
    },
}


@dataclass
class ClipSegment:
    """Instruction to place a clip segment on a timeline."""

    clip_path: Path
    start_time: float
    duration: float


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not yet exist."""

    path.mkdir(parents=True, exist_ok=True)


def analyze_audio(audio_path: Path) -> Dict[str, Any]:
    """Analyze an audio file for tempo, beats, onsets, RMS, and spectral centroid.

    Args:
        audio_path: Path to the WAV or other audio file.

    Returns:
        A dictionary containing analysis results.
    """

    logging.info("Loading audio: %s", audio_path)
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop_length = 512

    logging.info("Estimating tempo and beat positions")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    logging.info("Detecting onsets")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    logging.info("Computing RMS and spectral centroid")
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    duration = librosa.get_duration(y=y, sr=sr)

    analysis = {
        "sample_rate": sr,
        "tempo": float(tempo),
        "beat_times": beat_times.tolist(),
        "onset_times": onset_times.tolist(),
        "timeline": {
            "times": times.tolist(),
            "rms": rms.tolist(),
            "spectral_centroid": spectral_centroid.tolist(),
        },
        "duration": float(duration),
    }

    ensure_dir(METADATA_DIR)
    out_path = METADATA_DIR / f"{audio_path.stem}_timeline.json"
    logging.info("Saving audio analysis to %s", out_path)
    out_path.write_text(json.dumps(analysis, indent=2))

    return analysis


def analyze_clip(path: Path, sample_every_sec: float = DEFAULT_SAMPLE_EVERY_SEC) -> Dict[str, Any]:
    """Analyze a single clip for brightness and motion.

    Args:
        path: Video file path.
        sample_every_sec: Interval between sampled frames for statistics.

    Returns:
        Dictionary containing brightness, motion, and tags.
    """

    logging.debug("Analyzing clip: %s", path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    sample_interval = max(1, int(sample_every_sec * fps))

    brightness_values: List[float] = []
    motion_values: List[float] = []
    previous_gray: np.ndarray | None = None

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray) / 255.0))

        if previous_gray is not None:
            diff = cv2.absdiff(gray, previous_gray)
            motion_values.append(float(np.mean(diff) / 255.0))
        previous_gray = gray

        frame_idx += sample_interval
        if frame_idx >= frame_count:
            break

    cap.release()

    avg_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
    avg_motion = float(np.mean(motion_values)) if motion_values else 0.0

    tags: List[str] = []
    if avg_brightness > 0.6:
        tags.append("bright")
    elif avg_brightness < 0.4:
        tags.append("dark")

    if avg_motion > 0.2:
        tags.append("high_motion")
    else:
        tags.append("low_motion")

    return {
        "path": str(path),
        "brightness": avg_brightness,
        "motion": avg_motion,
        "tags": tags,
    }


def analyze_folder(
    folder: Path, out_json: Path, sample_every_sec: float = DEFAULT_SAMPLE_EVERY_SEC
) -> List[Dict[str, Any]]:
    """Analyze all MP4 files in a folder and save metadata."""

    logging.info("Analyzing folder: %s", folder)
    clips: List[Dict[str, Any]] = []
    for clip_path in sorted(folder.glob("*.mp4")):
        clips.append(analyze_clip(clip_path, sample_every_sec=sample_every_sec))

    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(clips, indent=2))
    logging.info("Saved clip metadata to %s", out_json)
    return clips


def _load_or_analyze_audio(audio_path: Path) -> Dict[str, Any]:
    """Load cached audio analysis or run a new one."""

    analysis_path = METADATA_DIR / f"{audio_path.stem}_timeline.json"
    if analysis_path.exists():
        logging.info("Loading existing audio analysis from %s", analysis_path)
        return json.loads(analysis_path.read_text())
    return analyze_audio(audio_path)


def _load_or_analyze_clips(
    track: str, face: str, folder: Path, sample_every_sec: float = DEFAULT_SAMPLE_EVERY_SEC
) -> List[Dict[str, Any]]:
    """Load cached clip analysis or run a new one."""

    out_json = METADATA_DIR / f"{track}_{face}_clips.json"
    if out_json.exists():
        logging.info("Loading existing clip analysis from %s", out_json)
        return json.loads(out_json.read_text())
    return analyze_folder(folder, out_json, sample_every_sec=sample_every_sec)


def _select_clip_by_tags(
    clips_meta: Sequence[Dict[str, Any]], desired_tags: Sequence[str]
) -> Dict[str, Any]:
    """Select a clip whose tags include all desired tags, falling back to any clip."""

    matching = [clip for clip in clips_meta if all(tag in clip["tags"] for tag in desired_tags)]
    if matching:
        return random.choice(matching)
    return random.choice(list(clips_meta))


def build_face_timeline(
    audio_analysis: Dict[str, Any],
    clips_meta: Sequence[Dict[str, Any]],
    face_logic: Dict[str, Any],
    total_duration: float,
) -> List[ClipSegment]:
    """Construct a face timeline driven by audio features."""

    beat_divisor = max(1, int(face_logic.get("beat_divisor", 1)))
    segment_duration = float(face_logic.get("segment_duration", 0.5))

    beats = audio_analysis.get("beat_times", [])
    times = np.array(audio_analysis["timeline"]["times"])
    rms = np.array(audio_analysis["timeline"]["rms"])
    spectral_centroid = np.array(audio_analysis["timeline"]["spectral_centroid"])

    segments: List[ClipSegment] = []
    for beat_time in beats[::beat_divisor]:
        if beat_time >= total_duration:
            break

        idx = int(np.argmin(np.abs(times - beat_time)))
        rms_value = rms[idx]
        centroid_value = spectral_centroid[idx]

        desired_tags: List[str] = []
        if rms_value > face_logic.get("rms_high", 0.2):
            desired_tags.append("high_motion")
        else:
            desired_tags.append("low_motion")

        if centroid_value > face_logic.get("centroid_high", 3000.0):
            desired_tags.append("bright")
        else:
            desired_tags.append("dark")

        clip = _select_clip_by_tags(clips_meta, desired_tags)
        segments.append(
            ClipSegment(
                clip_path=Path(clip["path"]),
                start_time=float(beat_time),
                duration=segment_duration,
            )
        )

    return segments


def _maybe_add_black_frame(duration: float, resolution: tuple[int, int]) -> mpy.ColorClip:
    """Create a black frame clip of a given duration and resolution."""

    return mpy.ColorClip(size=resolution, color=(0, 0, 0)).set_duration(duration)


def render_face_video(
    segments: Sequence[ClipSegment],
    audio_path: Path,
    output_path: Path,
    fps: int = 25,
    face_logic: Dict[str, Any] | None = None,
) -> None:
    """Render a video for a face using the provided segments."""

    logging.info("Rendering video to %s", output_path)
    face_logic = face_logic or {}
    stutter_prob = float(face_logic.get("stutter_prob", 0.0))
    black_frame_prob = float(face_logic.get("black_frame_prob", 0.0))

    clips: List[mpy.VideoClip] = []
    for segment in segments:
        source_clip = mpy.VideoFileClip(str(segment.clip_path))
        if source_clip.duration <= 0:
            continue

        if random.random() < black_frame_prob:
            clips.append(_maybe_add_black_frame(segment.duration, source_clip.size))
            source_clip.close()
            continue

        start_max = max(0.0, source_clip.duration - segment.duration)
        start = random.uniform(0.0, start_max) if start_max > 0 else 0.0
        subclip = source_clip.subclip(start, min(source_clip.duration, start + segment.duration))
        if subclip.duration < segment.duration:
            subclip = subclip.fx(mpy.vfx.loop, duration=segment.duration)

        chosen_clips = [subclip.set_duration(segment.duration)]

        if random.random() < stutter_prob:
            repeats = random.randint(2, 4)
            stutter_duration = segment.duration / repeats
            chosen_clips = [subclip.set_duration(stutter_duration)] * repeats

        clips.extend(chosen_clips)

    if not clips:
        raise ValueError("No clips to render; ensure segments and videos are available.")

    final = mpy.concatenate_videoclips(clips, method="compose")
    audio = mpy.AudioFileClip(str(audio_path))
    final = final.set_audio(audio).set_fps(fps)

    ensure_dir(output_path.parent)
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        threads=2,
        verbose=False,
        logger=None,
    )

    final.close()
    audio.close()
    for clip in clips:
        clip.close()


def apply_triangle_mask(face_video_path: Path, mask_png_path: Path, output_path: Path) -> None:
    """Apply a triangular alpha mask to a rendered face video."""

    logging.info("Applying mask %s to %s", mask_png_path, face_video_path)
    video = mpy.VideoFileClip(str(face_video_path))
    mask = mpy.ImageClip(str(mask_png_path), ismask=True).resize(video.size).set_duration(video.duration)
    masked = video.set_mask(mask)

    ensure_dir(output_path.parent)
    masked.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=int(video.fps or 25),
        threads=2,
        verbose=False,
        logger=None,
    )

    masked.close()
    mask.close()
    video.close()


def _load_face_clips(track: str, face_key: str, sample_every_sec: float) -> List[Dict[str, Any]]:
    """Load metadata for clips belonging to a face."""

    folder = BASE_DIR / "videos" / f"{track}_face{face_key}"
    return _load_or_analyze_clips(
        track, f"face{face_key}", folder, sample_every_sec=sample_every_sec
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TRANSMISSIONS generative video editor")
    parser.add_argument("--track", required=True, help="Track name, e.g., track01")
    parser.add_argument("--mask", action="store_true", help="Apply triangle masks if available")
    parser.add_argument("--force-audio", action="store_true", help="Re-run audio analysis even if cached")
    parser.add_argument(
        "--force-clips", action="store_true", help="Re-run clip analysis even if cached"
    )
    parser.add_argument(
        "--sample-every",
        type=float,
        default=DEFAULT_SAMPLE_EVERY_SEC,
        help="Seconds between sampled frames during clip analysis",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    track = args.track
    audio_path = BASE_DIR / "audio" / f"{track}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if args.force_audio and (METADATA_DIR / f"{audio_path.stem}_timeline.json").exists():
        (METADATA_DIR / f"{audio_path.stem}_timeline.json").unlink()

    audio_analysis = _load_or_analyze_audio(audio_path)
    total_duration = float(audio_analysis.get("duration", 0.0))

    face_outputs: Dict[str, Path] = {}
    for face_key in ("A", "B", "C"):
        face_folder = BASE_DIR / "videos" / f"{track}_face{face_key}"
        if not face_folder.exists():
            logging.warning("Video folder missing for face %s: %s", face_key, face_folder)
            continue

        meta_path = METADATA_DIR / f"{track}_face{face_key}_clips.json"
        if args.force_clips and meta_path.exists():
            meta_path.unlink()

        clips_meta = _load_or_analyze_clips(
            track, f"face{face_key}", face_folder, sample_every_sec=args.sample_every
        )
        if not clips_meta:
            logging.warning("No analyzed clips for face %s", face_key)
            continue

        segments = build_face_timeline(
            audio_analysis=audio_analysis,
            clips_meta=clips_meta,
            face_logic=FACE_CONFIG.get(face_key, {}),
            total_duration=total_duration,
        )

        output_path = OUTPUT_DIR / f"{track}_face{face_key}.mp4"
        render_face_video(
            segments=segments,
            audio_path=audio_path,
            output_path=output_path,
            face_logic=FACE_CONFIG.get(face_key, {}),
        )
        face_outputs[face_key] = output_path

    if args.mask and face_outputs:
        for face_key, video_path in face_outputs.items():
            mask_path = BASE_DIR / "masks" / f"face{face_key}_mask.png"
            if not mask_path.exists():
                logging.warning("Mask not found for face %s: %s", face_key, mask_path)
                continue

            masked_output = OUTPUT_DIR / f"{audio_path.stem}_face{face_key}_masked.mp4"
            apply_triangle_mask(video_path, mask_path, masked_output)


if __name__ == "__main__":
    main()
