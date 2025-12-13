#!/usr/bin/env python
"""
TRANSMISSIONS + MUGEN bridge

Uses mugen's MusicVideoGenerator to auto-cut music videos from
per-face clip folders (A/B/C), with:

- Forced ffmpeg normalization of all source clips (constant FPS, libx264, yuv420p)
- Optional parallel normalization
- Per-face beat speed multipliers

Example:
    python transmissions_mugen.py \
        --track track01 \
        --audio audio/track01.wav \
        --faces A B C \
        --videos-root videos \
        --output-root output_mugen \
        --normalize \
        --target-fps 25
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from mugen import MusicVideoGenerator  # pip install -e . from mugen repo


# -------------------------
# Utility + normalization
# -------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def progress(iterable: Iterable, desc: str = "") -> Iterable:
    """Wrap an iterable with tqdm if available."""
    if tqdm is not None:
        return tqdm(iterable, desc=desc)
    return iterable


def is_video_corrupt(path: Path) -> bool:
    """Heuristic corruption check using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nw=1:nk=1",
        str(path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        logging.warning("ffprobe unavailable or failed for %s: %s", path, exc)
        return False

    if proc.returncode != 0:
        logging.warning("ffprobe error for %s: %s", path, proc.stderr.strip())
        return True

    out = proc.stdout.strip()
    if out in ("", "N/A", "0", ""):
        logging.warning("ffprobe suspicious nb_frames for %s: %r", path, out)
        return True

    return False




def ffprobe_decodes(path: Path) -> bool:
    """
    Returns True only if ffprobe can decode at several points in the file.
    Catches clips that decode at t=0 but fail later (MoviePy errors during middle/last frame tests).
    """
    # Try a few positions: start, 25%, 50%, 75%
    intervals = ["%+#1", "25%+#1", "50%+#1", "75%+#1"]

    for iv in intervals:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-read_intervals", iv,
            "-show_entries", "frame=pkt_pts_time",
            "-of", "json",
            str(path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as exc:
            logging.warning("ffprobe decode test failed for %s (%s): %s", path, iv, exc)
            return True  # don't block if we cannot test

        if proc.returncode != 0:
            return False
        if '"frames"' not in (proc.stdout or ""):
            return False

    return True
def normalize_video(input_path: Path, output_path: Path, target_fps: int = 25) -> Path:
    """
    FORCE re-encode a video to a safe, consistent encoding (Option D):

    - Constant frame rate (target_fps)
    - yuv420p pixel format
    - libx264 video codec
    - AAC audio
    """
    ensure_dir(output_path.parent)
    logging.info("Normalizing video %s -> %s", input_path, output_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-err_detect", "ignore_err",
        "-i", str(input_path),
        "-vf", f"fps={target_fps},format=yuv420p",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return output_path


@dataclass
class NormalizeResult:
    input: Path
    normalized: Path
    skipped: bool
    corrupt: bool


def _normalize_worker(args: tuple[Path, Path, int]) -> NormalizeResult:
    src, dst_folder, target_fps = args
    out = dst_folder / src.name

    if out.exists():
        # Even if it already exists, validate it decodes; if not, re-normalize.
        if ffprobe_decodes(out):
            return NormalizeResult(input=src, normalized=out, skipped=True, corrupt=False)
        else:
            logging.warning("Existing normalized file not decodable; re-normalizing: %s", out)
            try:
                out.unlink(missing_ok=True)
            except Exception:
                pass

    corrupt = is_video_corrupt(src)
    if corrupt:
        logging.warning("Video appears corrupt (forcing re-encode): %s", src)

    normalized = normalize_video(src, out, target_fps=target_fps)

    # Strict decode validation
    if not ffprobe_decodes(normalized):
        logging.warning("Normalized output still not decodable; quarantining: %s", normalized)
        try:
            normalized.unlink(missing_ok=True)
        except Exception:
            pass
        # Mark as corrupt so summary tells you something happened
        return NormalizeResult(input=src, normalized=normalized, skipped=False, corrupt=True)

    return NormalizeResult(input=src, normalized=normalized, skipped=False, corrupt=corrupt)


def normalize_folder(
    src_folder: Path,
    dst_folder: Path,
    target_fps: int = 25,
) -> List[Path]:
    """Normalize all .mp4 clips in a folder in parallel; return list of normalized paths."""
    ensure_dir(dst_folder)

    clips = sorted(src_folder.glob("*.mp4"))
    if not clips:
        logging.warning("No .mp4 files in %s", src_folder)
        return []

    cpu = max(1, cpu_count() - 1)
    logging.info(
        "Normalizing %d clips from %s using %d workers",
        len(clips), src_folder, cpu,
    )

    tasks: List[tuple[Path, Path, int]] = [(c, dst_folder, target_fps) for c in clips]
    results: List[NormalizeResult] = []

    with Pool(processes=cpu) as pool:
        for res in progress(
            pool.imap_unordered(_normalize_worker, tasks),
            desc=f"normalize:{src_folder.name}",
        ):
            results.append(res)

    corrupted = sum(1 for r in results if r.corrupt)
    skipped = sum(1 for r in results if r.skipped)
    logging.info(
        "Normalization summary %s: %d total, %d suspected corrupt, %d skipped (already normalized)",
        src_folder, len(results), corrupted, skipped,
    )

    usable = [r.normalized for r in results if r.normalized.exists()]
    return usable
# -------------------------
# Mugen-based generation
# -------------------------

FACE_SPEED_DEFAULTS: Dict[str, float] = {
    "A": 1.0,    # every beat
    "B": 0.5,    # every other beat
    "C": 2.0,    # twice as fast
}


def build_events(generator: MusicVideoGenerator, speed: float) -> object:
    """
    Build mugen events object from audio beats with a speed multiplier.

    We keep this tiny so you can later swap in onsets, grouped slices, etc.
    """
    beats = generator.audio.beats()
    beats.speed_multiply(speed)
    return beats


def generate_face_video(
    audio_path: Path,
    clip_paths: Sequence[Path],
    out_path: Path,
    beat_speed: float,
    fps: int,
) -> None:
    """
    Use mugen to create an audio-reactive video from a set of clips.
    """
    if not clip_paths:
        raise RuntimeError(f"No clips found for face output: {out_path}")

    logging.info("Generating face video %s using %d clips", out_path.name, len(clip_paths))

    # mugen constructor takes audio and list of video source paths
    generator = MusicVideoGenerator(str(audio_path), [str(p) for p in clip_paths])

    events = build_events(generator, speed=beat_speed)
    music_video = generator.generate_from_events(events)

    ensure_dir(out_path.parent)
    # mugen uses MoviePy under the hood; write_to_video_file accepts typical kwargs
    # Force an explicit fps so downstream players and mugen's MoviePy writer
    # do not infer mismatched values from source clips.
    music_video.write_to_video_file(str(out_path), fps=fps)


    # optionally, you can save the pickle for later tweaks:
    # music_video.save(str(out_path.with_suffix(".pickle")))


# -------------------------
# CLI + orchestration
# -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRANSMISSIONS audio-reactive editor using mugen"
    )
    parser.add_argument(
        "--track",
        required=True,
        help="Track identifier, e.g. track01",
    )
    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        help="Path to audio file (e.g. audio/track01.wav or .mp3)",
    )
    parser.add_argument(
        "--faces",
        nargs="+",
        default=["A", "B", "C"],
        help="Faces to generate (subset of A B C)",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=Path("videos"),
        help="Root folder containing track_faceX folders (default: videos)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_mugen"),
        help="Output root folder (default: output_mugen)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize all source clips with ffmpeg before passing to mugen",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=25,
        help="Target FPS for normalization and final output (default: 25)",
    )
    parser.add_argument(
        "--face-speed",
        nargs="*",
        metavar="FACE=SPEED",
        help="Override beat speed per face, e.g. A=1 B=0.5 C=2",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def parse_face_speeds(overrides: Sequence[str] | None) -> Dict[str, float]:
    speeds = dict(FACE_SPEED_DEFAULTS)
    if not overrides:
        return speeds

    for item in overrides:
        if "=" not in item:
            continue
        face, val = item.split("=", 1)
        face = face.strip().upper()
        try:
            speed = float(val)
        except ValueError:
            logging.warning("Invalid speed override %r, ignoring", item)
            continue
        speeds[face] = speed
    return speeds


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio not found: {args.audio}")

    face_speeds = parse_face_speeds(args.face_speed)

    for face in args.faces:
        face_key = face.upper()
        src_folder = args.videos_root / f"{args.track}_face{face_key}"
        if not src_folder.exists():
            logging.warning("Skipping face %s (missing folder %s)", face_key, src_folder)
            continue

        # where normalized clips go
        norm_folder = args.videos_root / "normalized" / f"{args.track}_face{face_key}"

        if args.normalize:
            clip_paths = normalize_folder(
                src_folder,
                norm_folder,
                target_fps=args.target_fps,
            )
        else:
            # either raw or pre-normalized; your choice
            clip_paths = sorted(src_folder.glob("*.mp4"))

        if not clip_paths:
            logging.warning("No clips for face %s after normalization", face_key)
            continue

        out_face_path = args.output_root / f"{args.track}_face{face_key}_mugen.mkv"
        beat_speed = face_speeds.get(face_key, 1.0)

        generate_face_video(
            audio_path=args.audio,
            clip_paths=clip_paths,
            out_path=out_face_path,
            beat_speed=beat_speed,
            fps=args.target_fps,
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()

