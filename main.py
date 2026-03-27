"""Main library interface for YouTube transcript fetching."""

from .service import get_transcript, ServiceResult

__all__ = ["get_transcript", "ServiceResult"]