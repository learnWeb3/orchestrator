"""Exponential backoff with jitter for step-level retries."""

import random


def default_exponential_backoff(attempt: int) -> float:
    """Exponential backoff: 2^attempt seconds, plus up to 10% jitter.

    Attempt 0: ~1s, attempt 1: ~2s, attempt 2: ~4s, ...
    """
    base_delay = 2**attempt
    jitter = random.uniform(0, base_delay * 0.1)
    return base_delay + jitter
