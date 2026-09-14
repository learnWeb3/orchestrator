import uuid

import pytest

from orchestration_agent.models.rate_limiter import RedisRateLimiter


@pytest.fixture
def rate_limiter(redis_available):
    if not redis_available:
        pytest.skip("redis not reachable on localhost:6379 (run `docker compose up -d`)")

    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url("redis://localhost:6379/0")
    prefix = f"rate_limit:test:{uuid.uuid4().hex}:"
    limiter = RedisRateLimiter(
        redis_client=client, tokens_per_minute=100, tokens_per_hour=1000, key_prefix=prefix
    )
    try:
        yield limiter
    finally:
        limiter.reset()
        client.close()


def test_check_and_consume_allows_within_budget(rate_limiter):
    assert rate_limiter.check_and_consume(40) is True
    bucket = rate_limiter.get_current_bucket()
    assert bucket["minute_tokens"] == 40
    assert bucket["hour_tokens"] == 40


def test_check_and_consume_blocks_over_minute_budget(rate_limiter):
    assert rate_limiter.check_and_consume(60) is True
    assert rate_limiter.check_and_consume(60) is False  # would exceed 100/min

    bucket = rate_limiter.get_current_bucket()
    assert bucket["minute_tokens"] == 60  # the rejected call did not consume tokens


def test_reset_clears_buckets(rate_limiter):
    rate_limiter.check_and_consume(10)
    rate_limiter.reset()

    bucket = rate_limiter.get_current_bucket()
    assert bucket["minute_tokens"] == 0
    assert bucket["hour_tokens"] == 0
