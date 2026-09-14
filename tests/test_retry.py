from orchestration_agent.utils.retry import default_exponential_backoff


def test_backoff_grows_exponentially():
    # Jitter adds up to 10%, so compare against the base value with headroom.
    assert 1.0 <= default_exponential_backoff(0) <= 1.1
    assert 2.0 <= default_exponential_backoff(1) <= 2.2
    assert 4.0 <= default_exponential_backoff(2) <= 4.4


def test_backoff_is_monotonically_non_decreasing_in_expectation():
    values = [default_exponential_backoff(a) for a in range(5)]
    bases = [2**a for a in range(5)]
    for value, base in zip(values, bases):
        assert base <= value <= base * 1.1
