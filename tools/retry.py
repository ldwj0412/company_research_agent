import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from observability import classify_error


T = TypeVar("T")


_RETRYABLE_CATEGORIES = {"rate_limit", "timeout", "tool_error", "unknown_error"}
_NON_RETRYABLE_CATEGORIES = {"auth_error", "parse_error"}


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    value: T | None
    error: Exception | None
    attempts: int
    retried: bool
    error_category: str

    def format_failure(self, label: str) -> str:
        error_text = str(self.error or "unknown error")
        return (
            f"Error fetching {label} after {self.attempts} attempt"
            f"{'s' if self.attempts != 1 else ''} "
            f"({self.error_category or 'unknown_error'}): {error_text}"
        )


def retry_transient(
    operation: Callable[[], T],
    max_attempts: int = 2,
    sleep_seconds: float = 0.5,
) -> RetryResult[T]:
    """Retry transient tool failures, but fail fast for auth/config errors."""
    attempts = 0
    last_error: Exception | None = None
    last_category = ""

    while attempts < max_attempts:
        attempts += 1
        try:
            return RetryResult(
                value=operation(),
                error=None,
                attempts=attempts,
                retried=attempts > 1,
                error_category="",
            )
        except Exception as exc:
            last_error = exc
            last_category = classify_error(str(exc))
            if last_category in _NON_RETRYABLE_CATEGORIES:
                break
            if last_category not in _RETRYABLE_CATEGORIES:
                break
            if attempts >= max_attempts:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return RetryResult(
        value=None,
        error=last_error,
        attempts=attempts,
        retried=attempts > 1,
        error_category=last_category,
    )
