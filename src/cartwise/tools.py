"""Tools the agent can call, with JSON-schema specs and argument validation.

The model only ever proposes calls. Validation, business rules and error handling happen here,
and every failure comes back as a structured `{"error": ...}` the model can recover from.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cart import Cart, CartError
from .catalog import Catalog
from .retrieval import Retriever

log = logging.getLogger("cartwise")

_JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}


class ToolArgumentError(ValueError):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def validate_args(schema: dict, args: dict) -> dict:
    """Minimal JSON-schema check: required keys, unknown keys, types, enums and bounds."""
    props = schema.get("properties", {})
    unknown = set(args) - set(props)
    if unknown:
        raise ToolArgumentError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise ToolArgumentError(f"Missing required argument(s): {', '.join(missing)}")
    out = {}
    for key, value in args.items():
        rule = props[key]
        expected = _JSON_TYPES[rule["type"]]
        if rule["type"] == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)  # models often send 2.0 for 2
        if isinstance(value, bool) and rule["type"] in ("number", "integer"):
            raise ToolArgumentError(f"'{key}' must be a {rule['type']}")
        if not isinstance(value, expected):
            raise ToolArgumentError(f"'{key}' must be a {rule['type']}")
        if "enum" in rule and value not in rule["enum"]:
            raise ToolArgumentError(f"'{key}' must be one of: {', '.join(map(str, rule['enum']))}")
        if "minimum" in rule and value < rule["minimum"]:
            raise ToolArgumentError(f"'{key}' must be at least {rule['minimum']}")
        if "maximum" in rule and value > rule["maximum"]:
            raise ToolArgumentError(f"'{key}' must be at most {rule['maximum']}")
        if rule["type"] == "array" and "items" in rule:
            item_type = _JSON_TYPES[rule["items"]["type"]]
            if not all(isinstance(v, item_type) and not isinstance(v, bool) for v in value):
                raise ToolArgumentError(f"'{key}' must be a list of {rule['items']['type']}s")
        out[key] = value
    return out


class Toolbox:
    def __init__(self, catalog: Catalog, policies: Retriever, cart: Cart | None = None):
        self.catalog, self.policies = catalog, policies
        self.cart = cart or Cart()
        self._tools = {t.name: t for t in self._build()}

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def call(self, name: str, args: dict[str, Any] | None) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return tool.fn(**validate_args(tool.parameters, args or {}))
        except (ToolArgumentError, CartError) as exc:
            return {"error": str(exc)}
        except Exception as exc:  # a bug in one tool must not end the conversation
            log.exception("Tool '%s' failed", name)
            return {"error": f"Tool '{name}' failed ({type(exc).__name__}). Try a different request."}

    # --- tool implementations -------------------------------------------------------------

    def _search_products(self, query: str, category: str | None = None, max_price: float | None = None,
                         min_rating: float | None = None, limit: int = 5) -> dict:
        found = self.catalog.search(query, category=category, max_price=max_price, min_rating=min_rating, limit=limit)
        return {"results": [p.summary() for p in found], "count": len(found)}

    def _get_product(self, product_id: str) -> dict:
        product = self.catalog.get(product_id)
        return {"product": product.to_dict()} if product else {"error": f"No product with id '{product_id}'."}

    def _check_stock(self, product_id: str) -> dict:
        product = self.catalog.get(product_id)
        if not product:
            return {"error": f"No product with id '{product_id}'."}
        return {"id": product.id, "name": product.name, "in_stock": product.stock > 0, "quantity_available": product.stock}

    def _compare_products(self, product_ids: list) -> dict:
        product_ids = list(dict.fromkeys(product_ids))  # comparing a product with itself is not a comparison
        if not 2 <= len(product_ids) <= 4:
            return {"error": "Compare between 2 and 4 products."}
        products, missing = [], []
        for pid in product_ids:
            product = self.catalog.get(pid)
            (products if product else missing).append(product or pid)
        if missing:
            return {"error": f"Unknown product id(s): {', '.join(missing)}"}
        return {"products": [{**p.summary(), "attributes": p.attributes} for p in products]}

    def _search_policies(self, query: str, k: int = 3) -> dict:
        hits = self.policies.search(query, k=k)
        return {"passages": [{"source": h.chunk.source, "text": h.chunk.text, "score": h.score} for h in hits]}

    def _add_to_cart(self, product_id: str, quantity: int = 1) -> dict:
        product = self.catalog.get(product_id)
        if not product:
            return {"error": f"No product with id '{product_id}'."}
        self.cart.add(product, quantity)
        return {"added": {"id": product.id, "name": product.name, "quantity": quantity}, "cart": self.cart.snapshot()}

    def _view_cart(self) -> dict:
        return {"cart": self.cart.snapshot()}

    def _build(self) -> list[Tool]:
        pid = {"type": "string", "description": "Product id from search results, e.g. 'aud-100'."}
        return [
            Tool("search_products", "Search the catalog by keywords with optional filters.", {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords, e.g. 'noise cancelling headphones'."},
                    "category": {"type": "string", "enum": self.catalog.categories},
                    "max_price": {"type": "number", "minimum": 0},
                    "min_rating": {"type": "number", "minimum": 0, "maximum": 5},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            }, self._search_products),
            Tool("get_product", "Get full details for one product.",
                 {"type": "object", "properties": {"product_id": pid}, "required": ["product_id"]}, self._get_product),
            Tool("check_stock", "Check whether a product is in stock and how many are available.",
                 {"type": "object", "properties": {"product_id": pid}, "required": ["product_id"]}, self._check_stock),
            Tool("compare_products", "Compare 2 to 4 products side by side.", {
                "type": "object",
                "properties": {"product_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["product_ids"],
            }, self._compare_products),
            Tool("search_policies", "Search store policies: returns, shipping, warranty and price match.", {
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["query"],
            }, self._search_policies),
            Tool("add_to_cart", "Add a product to the shopper's cart.", {
                "type": "object",
                "properties": {"product_id": pid, "quantity": {"type": "integer", "minimum": 1, "maximum": 10}},
                "required": ["product_id"],
            }, self._add_to_cart),
            Tool("view_cart", "Show the cart contents and subtotal.",
                 {"type": "object", "properties": {}}, self._view_cart),
        ]
