import sys
import types
import tempfile
from pathlib import Path
from unittest import mock, TestCase


# Provide a minimal fake mugen module so trans can import without the real dependency.
fake_mugen = types.ModuleType("mugen")
fake_mugen.MusicVideoGenerator = object
sys.modules.setdefault("mugen", fake_mugen)

import trans


class DummyBeats:
    def __init__(self):
        self.speeds = []

    def speed_multiply(self, speed: float) -> None:
        self.speeds.append(speed)


class DummyAudio:
    def __init__(self):
        self.beats_obj = DummyBeats()

    def beats(self) -> DummyBeats:
        return self.beats_obj


class DummyVideo:
    def __init__(self):
        self.calls = []

    def write_to_video_file(self, path: str, fps: int) -> None:
        self.calls.append({"path": path, "fps": fps})


class DummyGenerator:
    created = []

    def __init__(self, audio_path: str, clip_paths):
        self.audio = DummyAudio()
        self.audio_path = audio_path
        self.clip_paths = clip_paths
        DummyGenerator.created.append(self)

    def generate_from_events(self, events):
        self.events = events
        self.video = DummyVideo()
        return self.video


class GenerateFaceVideoTests(TestCase):
    def setUp(self):
        DummyGenerator.created.clear()

    def test_generate_face_video_passes_fps_and_speed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.mkv"
            clip_path = Path(tmpdir) / "clip1.mp4"
            clip_path.touch()

            with mock.patch.object(trans, "MusicVideoGenerator", DummyGenerator):
                trans.generate_face_video(
                    audio_path=Path("audio.wav"),
                    clip_paths=[clip_path],
                    out_path=out_path,
                    beat_speed=1.5,
                    fps=30,
                )

        self.assertTrue(DummyGenerator.created, "MusicVideoGenerator not constructed")
        generator = DummyGenerator.created[-1]

        self.assertEqual(generator.audio_path, "audio.wav")
        self.assertEqual(generator.clip_paths, [str(clip_path)])

        beats = generator.audio.beats_obj
        self.assertEqual(beats.speeds, [1.5])
        self.assertIs(generator.events, beats)

        self.assertEqual(generator.video.calls, [{"path": str(out_path), "fps": 30}])


if __name__ == "__main__":
    import unittest

    unittest.main()
