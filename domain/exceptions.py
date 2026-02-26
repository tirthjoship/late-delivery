"""Domain exception hierarchy.

Steering: use custom exception classes for domain errors; fail fast
with clear error messages.
"""


class DomainError(Exception):
    """Base exception for domain errors."""

    pass


class InvalidForecastHorizonError(DomainError):
    """Raised when forecast horizon is invalid (e.g. non-positive)."""

    pass


class InvalidShippingModeError(DomainError):
    """Raised when shipping mode is not recognized."""

    pass
