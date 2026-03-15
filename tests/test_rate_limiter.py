import asyncio
import pytest
from api_gateway.rate_limiter import RateLimiter, TokenBucket


def test_token_bucket_allows_within_capacity():
    bucket = TokenBucket(capacity=5.0, refill_rate=1.0)
    for _ in range(5):
        assert bucket.consume() is True


def test_token_bucket_blocks_when_empty():
    bucket = TokenBucket(capacity=2.0, refill_rate=0.1)
    bucket.consume()
    bucket.consume()
    assert bucket.consume() is False


def test_token_bucket_refills_over_time():
    import time
    bucket = TokenBucket(capacity=1.0, refill_rate=10.0)
    bucket.consume()
    assert bucket.consume() is False
    time.sleep(0.15)
    assert bucket.consume() is True


@pytest.mark.asyncio
async def test_rate_limiter_allows_normal_traffic():
    limiter = RateLimiter(capacity=5.0, refill_rate=10.0)
    results = [await limiter.is_allowed("client-1") for _ in range(5)]
    assert all(results)


@pytest.mark.asyncio
async def test_rate_limiter_blocks_burst():
    limiter = RateLimiter(capacity=3.0, refill_rate=0.1)
    results = [await limiter.is_allowed("client-x") for _ in range(6)]
    assert results[:3] == [True, True, True]
    assert False in results[3:]


@pytest.mark.asyncio
async def test_rate_limiter_cleanup_preserves_defaultdict():
    limiter = RateLimiter(capacity=5.0, refill_rate=5.0)
    await limiter.is_allowed("client-keep")
    await limiter.cleanup()
    # After cleanup, new key must still auto-create (no KeyError)
    result = await limiter.is_allowed("brand-new-client")
    assert result is True
