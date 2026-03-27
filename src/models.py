"""Data models for transcripts and segments."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class TranscriptSegment:
    """A single segment of a transcript with timing information."""
    start: float      # seconds
    end: float        # seconds
    text: str

    def __str__(self) -> str:
        return self.text.strip()


@dataclass
class Transcript:
    """A complete transcript for a video."""
    video_id: str
    language: str
    source: Literal["innertube", "whisper"]
    segments: list[TranscriptSegment] = field(default_factory=list)
    created_at: str = ""  # ISO timestamp

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_text(self) -> str:
        """Convert transcript to plain text."""
        return " ".join(seg.text.strip() for seg in self.segments)

    def to_srt(self) -> str:
        """Convert transcript to SRT format."""
        lines = []
        for i, seg in enumerate(self.segments, start=1):
            start_t = self._seconds_to_srt_time(seg.start)
            end_t = self._seconds_to_srt_time(seg.end)
            lines.append(f"{i}\n{start_t} --> {end_t}\n{seg.text.strip()}\n")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Convert transcript to VTT format."""
        lines = ["WEBVTT", ""]
        for seg in self.segments:
            start_t = self._seconds_to_vtt_time(seg.start)
            end_t = self._seconds_to_vtt_time(seg.end)
            lines.append(f"{start_t} --> {end_t}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> dict:
        """Convert transcript to JSON-serializable dict."""
        return {
            "video_id": self.video_id,
            "language": self.language,
            "source": self.source,
            "created_at": self.created_at,
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text}
                for seg in self.segments
            ],
        }

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to VTT time format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
