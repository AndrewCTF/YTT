"""Token bucket rate limiter for Innertube API calls."""

import asyncio
import time
from dataclasses import dataclass


@dataclass
class RateLimiter:
    """Token bucket rate limiter.

    Implements a token bucket algorithm to limit the rate of Innertube API calls.
    Tokens are added at a constant rate (RATE_LIMIT_RATE per second) up to the
    burst capacity (RATE_LIMIT_BURST). Each fetch consumes one token.

    Attributes:
        rate: Tokens added per second.
        burst: Maximum tokens (bucket size).
        tokens: Current number of available tokens.
        last_refill: Timestamp of last token refill.
    """

    rate: float = 0.5        # tokens per second
    burst: int = 5           # max tokens (bucket size)
    tokens: float = 5.0      # current tokens
    last_refill: float = None

    def __post_init__(self):
        self.last_refill = time.time()
        self.tokens = float(self.burst)

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens at the specified rate
        new_tokens = elapsed * self.rate
        self.tokens = min(self.burst, self.tokens + new_tokens)
        self.last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available and acquire it.

        This method will block until a token is available.
        Uses busy-waiting with sleep for async compatibility.
        """
        while True:
            self._refill()

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return

            # Wait a short time before checking again
            # Calculate wait time based on token deficit
            tokens_needed = 1.0 - self.tokens
            wait_time = tokens_needed / self.rate
            await asyncio.sleep(min(wait_time, 0.1))

    def get_wait_time(self) -> float:
        """Get estimated wait time until a token is available.

        Returns:
            Estimated seconds to wait for one token.
        """
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.rate


# Global rate limiter instance
rate_limiter = RateLimiter()