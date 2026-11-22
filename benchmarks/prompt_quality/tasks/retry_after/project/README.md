# HTTP client helpers

Public helpers live in `main.py`. Header lookup is case-insensitive. `Retry-After` accepts either delay seconds or an HTTP date. Return seconds as a float, clamp past dates to zero, and use the function's `default` for missing or invalid values. The optional `now` argument supports deterministic tests.
