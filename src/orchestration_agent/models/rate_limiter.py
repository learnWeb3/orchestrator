"""External rate limiter integration.

`RateLimiterContext.check_and_consume` is called synchronously (without `await`) from
`Agent.run()`, so implementations must be synchronous — see spec deviation #5 in the
implementation plan. `RedisRateLimiter` therefore uses the plain (sync) `redis` client,
not `redis.asyncio`.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class RateLimiterContext(ABC):
    """Interface for external, bucket-based rate limiter integration."""

    @abstractmethod
    def check_and_consume(self, tokens: int) -> bool:
        """Check if `tokens` can be consumed and consume them if allowed."""

    @abstractmethod
    def get_current_bucket(self) -> Dict[str, Any]:
        """Get current rate limiter state."""

    @abstractmethod
    def reset(self) -> None:
        """Reset rate limiter state."""

    def release(self, tokens: int) -> None:
        """Give back over-reserved tokens once actual usage is known.

        Supports the reserve-then-reconcile pattern (spec section 10): a
        caller reserves an upper-bound estimate before dispatching a provider
        call, then reconciles the reservation down to actual usage once the
        response returns. Concrete (not abstract) with a no-op default so
        existing implementations don't break; override to support exact
        reconciliation.
        """


class RedisRateLimiter(RateLimiterContext):
    """Redis-backed sliding-window-ish (fixed window) token bucket rate limiter.

    Tracks tokens consumed in the current minute and current hour using
    `INCRBY` + `EXPIRE` on two keys. Not perfectly precise at window boundaries
    (fixed-window, not sliding), which is an accepted trade-off for a dev/example
    implementation per the spec.
    """

    def __init__(
        self,
        redis_client: Any,
        tokens_per_minute: int = 10_000,
        tokens_per_hour: int = 1_000_000,
        key_prefix: str = "rate_limit:agent:",
    ) -> None:
        self.redis = redis_client
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_hour = tokens_per_hour
        self.key_prefix = key_prefix

    def _minute_key(self) -> str:
        return f"{self.key_prefix}minute"

    def _hour_key(self) -> str:
        return f"{self.key_prefix}hour"

    def check_and_consume(self, tokens: int) -> bool:
        minute_key = self._minute_key()
        hour_key = self._hour_key()

        minute_tokens = int(self.redis.get(minute_key) or 0)
        hour_tokens = int(self.redis.get(hour_key) or 0)

        if minute_tokens + tokens > self.tokens_per_minute:
            return False
        if hour_tokens + tokens > self.tokens_per_hour:
            return False

        pipe = self.redis.pipeline()
        pipe.incrby(minute_key, tokens)
        pipe.expire(minute_key, 60)
        pipe.incrby(hour_key, tokens)
        pipe.expire(hour_key, 3600)
        pipe.execute()

        return True

    def get_current_bucket(self) -> Dict[str, Any]:
        return {
            "minute_tokens": int(self.redis.get(self._minute_key()) or 0),
            "minute_limit": self.tokens_per_minute,
            "hour_tokens": int(self.redis.get(self._hour_key()) or 0),
            "hour_limit": self.tokens_per_hour,
        }

    def reset(self) -> None:
        self.redis.delete(self._minute_key(), self._hour_key())

    def release(self, tokens: int) -> None:
        if tokens <= 0:
            return
        minute_key = self._minute_key()
        hour_key = self._hour_key()

        # Best-effort, non-atomic (matches the fixed-window trade-off already
        # accepted above); clamp at 0 so a release can't push a bucket negative.
        minute_tokens = int(self.redis.get(minute_key) or 0)
        hour_tokens = int(self.redis.get(hour_key) or 0)

        pipe = self.redis.pipeline()
        pipe.set(minute_key, max(minute_tokens - tokens, 0), keepttl=True)
        pipe.set(hour_key, max(hour_tokens - tokens, 0), keepttl=True)
        pipe.execute()
