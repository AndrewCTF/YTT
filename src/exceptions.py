"""Custom exceptions for the YouTube transcript fetcher."""


class YouTubeTranscriptError(Exception):
    """Base exception for transcript errors."""
    pass


class VideoUnavailable(YouTubeTranscriptError):
    """Raised when the video does not exist or is unavailable."""
    pass


class NoTranscriptFound(YouTubeTranscriptError):
    """Raised when no transcript is available for the video."""
    pass


class RateLimitError(YouTubeTranscriptError):
    """Raised when YouTube rate limits the request."""
    def __init__(self, message: str = "Rate limited by YouTube", retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ExtractionError(YouTubeTranscriptError):
    """Raised when data extraction from YouTube fails."""
    pass


class WhisperError(YouTubeTranscriptError):
    """Raised when Whisper transcription fails."""
    pass


class SearchError(YouTubeTranscriptError):
    """Raised when YouTube search fails."""
    pass
