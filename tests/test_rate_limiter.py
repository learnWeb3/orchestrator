import uuid

import pytest

from orchestration_agent.models.rate_limiter import RedisRateLimiter


class _FakePipeline:
    """Minimal fake of a redis pipeline: batches ops, applies them on execute()."""

    def __init__(self, store: dict):
        self._store = store
        self._ops = []

    def incrby(self, key, amount):
        self._ops.append(("incrby", key, amount))
        return self

    def expire(self, key, seconds):
        return self

    def set(self, key, value, keepttl=False):
        self._ops.append(("set", key, value))
        return self

    def execute(self):
        for op, key, value in self._ops:
            if op == "incrby":
                self._store[key] = int(self._store.get(key, 0)) + value
            else:
                self._store[key] = value
        self._ops = []


class _FakeRedis:
    """Minimal in-process fake covering the redis-py surface RedisRateLimiter uses,
    so its reserve/release arithmetic can be unit-tested without a real redis
    instance (see the docker-backed `rate_limiter` fixture below for integration
    coverage)."""

    def __init__(self):
        self.store: dict = {}

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)

    def pipeline(self):
        return _FakePipeline(self.store)


def test_release_reconciles_reservation_without_real_redis():
    limiter = RedisRateLimiter(redis_client=_FakeRedis(), tokens_per_minute=1000, tokens_per_hour=10_000)

    assert limiter.check_and_consume(50) is True
    limiter.release(20)

    bucket = limiter.get_current_bucket()
    assert bucket["minute_tokens"] == 30
    assert bucket["hour_tokens"] == 30


def test_release_clamps_at_zero_without_real_redis():
    limiter = RedisRateLimiter(redis_client=_FakeRedis(), tokens_per_minute=1000, tokens_per_hour=10_000)

    assert limiter.check_and_consume(10) is True
    limiter.release(999)

    bucket = limiter.get_current_bucket()
    assert bucket["minute_tokens"] == 0
    assert bucket["hour_tokens"] == 0


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


def test_release_gives_back_over_reserved_tokens(rate_limiter):
    assert rate_limiter.check_and_consume(40) is True
    rate_limiter.release(15)

    bucket = rate_limiter.get_current_bucket()
    assert bucket["minute_tokens"] == 25
    assert bucket["hour_tokens"] == 25


def test_release_clamps_at_zero(rate_limiter):
    assert rate_limiter.check_and_consume(10) is True
    rate_limiter.release(100)

    bucket = rate_limiter.get_current_bucket()
    assert bucket["minute_tokens"] == 0
    assert bucket["hour_tokens"] == 0


def test_reset_clears_buckets(rate_limiter):
    rate_limiter.check_and_consume(10)
    rate_limiter.reset()

    bucket = rate_limiter.get_current_bucket()
    assert bucket["minute_tokens"] == 0
    assert bucket["hour_tokens"] == 0
