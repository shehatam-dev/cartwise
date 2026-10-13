"""Product catalog with keyword search and filters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

from .text import tokenize


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    brand: str
    category: str
    price: float
    rating: float
    stock: int
    description: str
    attributes: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Fields the model sees in search results (no long description)."""
        return {"id": self.id, "name": self.name, "brand": self.brand, "category": self.category,
                "price": self.price, "rating": self.rating, "in_stock": self.stock > 0}

    def to_dict(self) -> dict:
        return asdict(self)

    def _search_text(self) -> str:
        attrs = " ".join(f"{k} {v}" for k, v in self.attributes.items())
        return f"{self.name} {self.brand} {self.category} {self.description} {attrs}".replace("_", " ")


class Catalog:
    def __init__(self, products: list[Product]):
        self._products = {p.id: p for p in products}
        self._tokens = {p.id: set(tokenize(p._search_text())) for p in products}
        self.categories = sorted({p.category for p in products})

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "Catalog":
        if path is None:
            raw = resources.files("cartwise").joinpath("data/catalog.json").read_text()
        else:
            raw = Path(path).read_text()
        return cls([Product(**row) for row in json.loads(raw)])

    def get(self, product_id: str) -> Product | None:
        return self._products.get(product_id)

    def search(self, query: str = "", *, category: str | None = None, max_price: float | None = None,
               min_rating: float | None = None, limit: int = 5) -> list[Product]:
        q = set(tokenize(query))
        scored = []
        for pid, product in self._products.items():
            if category and product.category != category:
                continue
            if max_price is not None and product.price > max_price:
                continue
            if min_rating is not None and product.rating < min_rating:
                continue
            score = len(q & self._tokens[pid])
            if q and score == 0:
                continue
            scored.append((score, product.rating, product))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [p for _, _, p in scored[:limit]]
