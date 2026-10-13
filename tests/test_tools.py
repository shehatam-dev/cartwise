from cartwise.retrieval import BM25Retriever


def test_search_filters_by_price_and_category(toolbox):
    out = toolbox.call("search_products", {"query": "headphones", "category": "headphones", "max_price": 100})
    assert out["count"] > 0
    assert all(p["price"] <= 100 and p["category"] == "headphones" for p in out["results"])


def test_unknown_tool_returns_error(toolbox):
    assert toolbox.call("delete_database", {}) == {"error": "Unknown tool 'delete_database'."}


def test_argument_validation(toolbox):
    assert "Missing required" in toolbox.call("get_product", {})["error"]
    assert "Unknown argument" in toolbox.call("get_product", {"product_id": "aud-100", "x": 1})["error"]
    assert "must be a number" in toolbox.call("search_products", {"query": "x", "max_price": "cheap"})["error"]
    assert "must be one of" in toolbox.call("search_products", {"query": "x", "category": "cars"})["error"]


def test_integer_like_floats_are_accepted(toolbox):
    out = toolbox.call("add_to_cart", {"product_id": "bag-300", "quantity": 2.0})
    assert out["added"]["quantity"] == 2


def test_cart_enforces_limits_and_stock(toolbox):
    assert "Limit is 5" in toolbox.call("add_to_cart", {"product_id": "cof-300", "quantity": 6})["error"]
    assert "Only 3 in stock" in toolbox.call("add_to_cart", {"product_id": "shoe-300", "quantity": 4})["error"]
    assert "out of stock" not in str(toolbox.call("view_cart", {}))  # failed adds leave the cart empty
    assert toolbox.call("view_cart", {})["cart"]["subtotal"] == 0


def test_cart_subtotal(toolbox):
    toolbox.call("add_to_cart", {"product_id": "bag-300", "quantity": 2})
    toolbox.call("add_to_cart", {"product_id": "cof-300", "quantity": 1})
    assert toolbox.call("view_cart", {})["cart"]["subtotal"] == 109.0


def test_compare_needs_two_known_products(toolbox):
    assert "between 2 and 4" in toolbox.call("compare_products", {"product_ids": ["aud-100"]})["error"]
    assert "Unknown product" in toolbox.call("compare_products", {"product_ids": ["aud-100", "nope"]})["error"]


def test_policy_retrieval_finds_the_right_section():
    hits = BM25Retriever.from_policies().search("express shipping cost", k=1)
    assert hits[0].chunk.source == "shipping.md#Express shipping"


def test_policy_retrieval_returns_nothing_for_unrelated_query():
    assert BM25Retriever.from_policies().search("quantum chromodynamics") == []
