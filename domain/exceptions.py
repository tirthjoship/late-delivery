"""Domain exception hierarchy.

Use custom exception classes for domain errors; fail fast
with clear error messages.
"""


class DomainError(Exception):
    """Base exception for domain errors."""

    pass
