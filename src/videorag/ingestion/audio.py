import subprocess
from pathlib import Path


class AudioExtractionError(RuntimeError):
    pass


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """Extract mono PCM WAV audio from a video file using ffmpeg.

    16kHz mono matches what faster-whisper expects, avoiding an internal resample.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed for {video_path}: {result.stderr.strip()[-2000:]}"
        )
    return output_path
