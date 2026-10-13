"""A deterministic, rule-based stand-in for an LLM.

It speaks the same protocol as a real provider (it proposes tool calls, then answers from the tool
results), so the agent loop, guardrails, tracing and evals can be exercised offline and in CI
without an API key. It is not meant to be clever; swap in `gemini` for real conversations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..guardrails import REDACTED, find_injection
from ..text import money, tokenize
from .base import LLMResponse, Message, ToolCall

_CHECKOUT = re.compile(r"\b(check ?out|pay(ment)?|credit card|card number|place (the|my|an|this) order)\b", re.I)
_DISCOUNT = re.compile(r"\b(discount|coupon|promo code|\d+\s?% off|for free|make it free)\b", re.I)
_POLICY = re.compile(r"\b(return|refund|exchang|ship|deliver|warrant|price ?match|restocking)\w*", re.I)
_COMPARE = re.compile(r"\b(compare|versus|vs\.?)\b", re.I)
_ADD = re.compile(r"\badd\b.*\bcart\b|\bbuy\b", re.I)
_STOCK = re.compile(r"\b(in stock|available|availability)\b", re.I)
_DETAILS = re.compile(r"\b(tell me about|more about|details|describe|specs)\b", re.I)
_CART = re.compile(r"\b(what'?s in my cart|view (my )?cart|show (me )?(my )?cart)\b", re.I)
_MAX_PRICE = re.compile(r"\b(?:under|below|less than|up to|max(?:imum)?)\s*\$?(\d+(?:\.\d+)?)", re.I)
_MIN_RATING = re.compile(r"\b(?:rated\s*)?(\d(?:\.\d)?)\+?\s*stars?\b|\brated (?:at least )?(\d(?:\.\d)?)", re.I)
_QTY = re.compile(r"\badd\s+(\d+)\b|\b(\d+)\s*(?:x|units?|pairs?|of)\b", re.I)

# Checked in order: accessory nouns first, so "laptop backpack" is a backpack.
_CATEGORY_WORDS = [
    ("backpacks", ("backpack", "bag", "sling")),
    ("headphones", ("headphone", "earbud", "bud", "headset")),
    ("laptops", ("laptop", "chromebook", "notebook")),
    ("coffee makers", ("coffee", "espresso", "kettle", "brewer")),
    ("running shoes", ("shoe", "sneaker", "runner", "running")),
]


@dataclass
class _Intent:
    kind: str  # refuse_checkout | refuse_discount | refuse_injection | policy | compare | add | stock | details | cart | search | greet


def classify(text: str) -> _Intent:
    if find_injection(text):
        return _Intent("refuse_injection")
    if _CHECKOUT.search(text):
        return _Intent("refuse_checkout")
    if _DISCOUNT.search(text):
        return _Intent("refuse_discount")
    for kind, pattern in (("cart", _CART), ("policy", _POLICY), ("compare", _COMPARE), ("add", _ADD),
                          ("stock", _STOCK), ("details", _DETAILS)):
        if pattern.search(text):
            return _Intent(kind)
    return _Intent("search" if tokenize(text) else "greet")


def _category(text: str) -> str | None:
    tokens = set(tokenize(text))
    for category, words in _CATEGORY_WORDS:
        if tokens & set(words):
            return category
    return None


def _search_args(text: str, limit: int = 3) -> dict:
    args: dict = {"query": text, "limit": limit}
    if (category := _category(text)):
        args["category"] = category
    if (m := _MAX_PRICE.search(text)):
        args["max_price"] = float(m.group(1))
    if (m := _MIN_RATING.search(text)):
        args["min_rating"] = float(m.group(1) or m.group(2))
    return args


def _compare_parts(text: str) -> list[str]:
    body = re.split(_COMPARE, text, maxsplit=1)[-1]
    parts = re.split(r"\s+(?:and|vs\.?|versus|with|or)\s+|,", body, flags=re.I)
    return [p.strip(" ?.!") for p in parts if tokenize(p)][:4]


class ScriptedLLM:
    provider = "scripted"
    model = "rules-v1"

    def generate(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        last_user = max(i for i, m in enumerate(messages) if m.role == "user")
        text = messages[last_user].content
        results: dict[str, list[dict]] = {}
        for m in messages[last_user + 1:]:
            if m.role == "tool":
                results.setdefault(m.name, []).append(json.loads(m.content))
        intent = classify(text)
        step = self._plan(intent, text, results)
        if isinstance(step, list):
            return LLMResponse(tool_calls=step, input_tokens=len(text.split()))
        return LLMResponse(text=step, input_tokens=len(text.split()), output_tokens=len(step.split()))

    # Returns tool calls to make next, or the final answer text.
    def _plan(self, intent: _Intent, text: str, results: dict) -> list[ToolCall] | str:
        k = intent.kind
        if k == "refuse_injection":
            return ("I can't help with that request. I'm here to help you shop: finding products, "
                    "checking stock and answering questions about our policies.")
        if k == "refuse_checkout":
            return ("I can't take payment or place orders in this chat. You can review your cart and "
                    "check out securely on the cart page.")
        if k == "refuse_discount":
            return ("I can't change prices or apply discounts. I can help you find options within your "
                    "budget, or check whether our price-match policy applies.")
        if k == "greet":
            return "Hi! I can help you find products, compare them, check stock and answer policy questions."
        if k == "cart":
            if "view_cart" not in results:
                return [ToolCall("view_cart", {})]
            return _cart_text(results["view_cart"][-1]["cart"])
        if k == "policy":
            if "search_policies" not in results:
                return [ToolCall("search_policies", {"query": text, "k": 2})]
            return _policy_text(results["search_policies"][-1])
        if k == "compare":
            if "search_products" not in results:
                return [ToolCall("search_products", {"query": p, "limit": 1}) for p in _compare_parts(text)]
            ids = [r["results"][0]["id"] for r in results["search_products"] if r.get("results")]
            if "compare_products" not in results and len(set(ids)) >= 2:
                return [ToolCall("compare_products", {"product_ids": list(dict.fromkeys(ids))})]
            if "compare_products" in results:
                return _compare_text(results["compare_products"][-1])
            return "I could only find one of those products. Could you check the names?"
        if k in ("add", "stock", "details"):
            if "search_products" not in results:
                return [ToolCall("search_products", {"query": text, "limit": 1})]
            hits = results["search_products"][-1].get("results", [])
            if not hits:
                return "I couldn't find that product. Could you give me the name or a few more details?"
            pid = hits[0]["id"]
            follow = {"add": "add_to_cart", "stock": "check_stock", "details": "get_product"}[k]
            if follow not in results:
                args = {"product_id": pid}
                if k == "add":
                    m = _QTY.search(text)
                    args["quantity"] = int(m.group(1) or m.group(2)) if m else 1
                return [ToolCall(follow, args)]
            out = results[follow][-1]
            if "error" in out:
                return f"I couldn't do that: {out['error']}"
            if k == "add":
                a, cart = out["added"], out["cart"]
                return f"Added {a['quantity']} × {a['name']} to your cart. Cart subtotal: {money(cart['subtotal'])}."
            if k == "stock":
                return (f"{out['name']} is in stock ({out['quantity_available']} available)." if out["in_stock"]
                        else f"{out['name']} is currently out of stock.")
            return _details_text(out["product"])
        # plain search
        if "search_products" not in results:
            return [ToolCall("search_products", _search_args(text))]
        return _search_text(results["search_products"][-1])


def _clean(text: str) -> str:
    return " ".join(text.replace(REDACTED, "").split())


def _search_text(out: dict) -> str:
    if "error" in out:
        return f"I couldn't run that search: {out['error']}"
    if not out["results"]:
        return "I couldn't find matching products. Try a broader search or a different budget."
    lines = [f"- {p['name']} ({p['id']}): {money(p['price'])}, rated {p['rating']}/5"
             + ("" if p["in_stock"] else ", currently out of stock") for p in out["results"]]
    return "Here are the best matches:\n" + "\n".join(lines)


def _policy_text(out: dict) -> str:
    passages = out.get("passages", [])
    if not passages:
        return "I couldn't find a policy that covers that. Our support team can help with specific cases."
    # Use the runner-up too when it scores close to the top hit (e.g. standard vs. opened-item returns).
    chosen = [p for p in passages[:2] if p["score"] >= 0.8 * passages[0]["score"]]
    body = " ".join(_clean(p["text"]) for p in chosen)
    sources = "; ".join('{}, "{}"'.format(*p["source"].split("#", 1)) for p in chosen)
    return f"{body} (Source: {sources})"


def _compare_text(out: dict) -> str:
    if "error" in out:
        return f"I couldn't compare those: {out['error']}"
    products = out["products"]
    lines = []
    for p in products:
        attrs = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in p["attributes"].items())
        lines.append(f"- {p['name']}: {money(p['price'])}, rated {p['rating']}/5 ({attrs})")
    cheapest = min(products, key=lambda p: p["price"])
    best = max(products, key=lambda p: p["rating"])
    return ("Here's how they compare:\n" + "\n".join(lines)
            + f"\nThe {cheapest['name']} costs less; the {best['name']} has the higher rating.")


def _details_text(p: dict) -> str:
    stock = "in stock" if p["stock"] > 0 else "out of stock"
    return f"{p['name']} by {p['brand']}: {money(p['price'])}, rated {p['rating']}/5, {stock}. {_clean(p['description'])}"


def _cart_text(cart: dict) -> str:
    if not cart["items"]:
        return "Your cart is empty."
    lines = [f"- {i['quantity']} × {i['name']}: {money(i['line_total'])}" for i in cart["items"]]
    return "Your cart:\n" + "\n".join(lines) + f"\nSubtotal: {money(cart['subtotal'])}"
