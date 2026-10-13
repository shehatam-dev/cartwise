"""A per-session shopping cart. Business rules live here, not in the prompt."""

from __future__ import annotations

from .catalog import Product

MAX_QTY_PER_ITEM = 5


class CartError(ValueError):
    pass


class Cart:
    def __init__(self) -> None:
        self._items: dict[str, tuple[Product, int]] = {}

    def add(self, product: Product, quantity: int) -> None:
        if quantity < 1:
            raise CartError("Quantity must be at least 1.")
        current = self._items.get(product.id, (product, 0))[1]
        new_qty = current + quantity
        if new_qty > MAX_QTY_PER_ITEM:
            raise CartError(f"Limit is {MAX_QTY_PER_ITEM} per item; the cart already has {current}.")
        if new_qty > product.stock:
            raise CartError(f"Only {product.stock} in stock.")
        self._items[product.id] = (product, new_qty)

    def snapshot(self) -> dict:
        lines = [{"id": p.id, "name": p.name, "quantity": q, "unit_price": p.price,
                  "line_total": round(p.price * q, 2)} for p, q in self._items.values()]
        return {"items": lines, "subtotal": round(sum(l["line_total"] for l in lines), 2)}
