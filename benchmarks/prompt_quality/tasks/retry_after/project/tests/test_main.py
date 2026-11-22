from datetime import datetime, timezone

from main import parse_retry_after


def test_numeric_seconds_is_returned_as_float():
    value = parse_retry_after({"Retry-After": "2"})
    assert isinstance(value, float)
    assert value == 2.0


def test_header_lookup_is_case_insensitive():
    assert parse_retry_after({"retry-after": "4"}) == 4.0


def test_http_date_uses_injected_now():
    headers = {"Retry-After": "Wed, 21 Oct 2015 07:28:10 GMT"}
    now = datetime(2015, 10, 21, 7, 28, 0, tzinfo=timezone.utc)
    assert parse_retry_after(headers, now=now) == 10.0


def test_past_date_is_clamped_to_zero():
    headers = {"Retry-After": "Wed, 21 Oct 2015 07:27:00 GMT"}
    now = datetime(2015, 10, 21, 7, 28, 0, tzinfo=timezone.utc)
    assert parse_retry_after(headers, now=now) == 0.0


def test_missing_or_invalid_value_uses_default():
    assert parse_retry_after({}, default=2.5) == 2.5
    assert parse_retry_after({"Retry-After": "soon"}, default=2.5) == 2.5
