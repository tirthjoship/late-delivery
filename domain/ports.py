"""Abstract interfaces (protocols) for Retail Supply Chain.

Adapters implement these ports; domain and application depend only
on these abstractions.
"""

from typing import Protocol

from .models import Order, Product


class SalesDataRepository(Protocol):
    """Port: load historical sales/order data.

    Implemented by adapters (e.g. CSV, database) to provide orders
    and optionally products without the domain depending on I/O.
    """

    def get_orders(self) -> list[Order]:
        """Return all orders (and their line items) from the source.

        Returns:
            List of domain Order entities. Order items are grouped
            by Order Id. Leakage columns (e.g. real shipping days,
            delivery status) must not be used.
        """
        ...

    def get_products(self) -> list[Product]:
        """Return distinct products from the source.

        Returns:
            List of domain Product entities.
        """
        ...
