import argparse
import asyncio
import json
import os
import sqlite3
from types import SimpleNamespace

from dotenv import load_dotenv


# -------------------------
# 1. LLM client (Groq, not xmlrpc.client)
# -------------------------
# The original crash was:
#   AttributeError: module 'xmlrpc.client' has no attribute 'chat'
# that happens if `client` is bound to stdlib xmlrpc.client instead of
# the Groq/OpenAI-compatible chat SDK.

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def _build_live_client():
    from groq import AsyncGroq

    return AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


def _tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _completion(message, total_tokens=32):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            total_tokens=total_tokens,
            prompt_tokens=max(1, total_tokens // 2),
            completion_tokens=max(1, total_tokens - total_tokens // 2),
        ),
    )


class MockChatCompletions:
    """Deterministic stand-in so the eval suite can run without an API key."""

    def __init__(self):
        self._turn = 0

    async def create(self, **kwargs):
        messages = kwargs.get("messages") or []
        user_text = ""
        for msg in messages:
            content = msg["content"] if isinstance(msg, dict) else getattr(msg, "content", None)
            role = msg["role"] if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "user" and content:
                user_text = content
                break

        has_tool_result = any(
            (m["role"] if isinstance(m, dict) else getattr(m, "role", None)) == "tool"
            for m in messages
        )

        lowered = user_text.lower()
        self._turn += 1

        if not has_tool_result:
            if "order" in lowered:
                return _completion(
                    SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[_tool_call("get_order_details", {"order_id": 102})],
                    )
                )
            if "customer" in lowered:
                return _completion(
                    SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[_tool_call("get_customer_details", {"customer_id": 2})],
                    )
                )
            if "refund" in lowered or "restock" in lowered or "policy" in lowered:
                return _completion(
                    SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[_tool_call("search_policy", {"query": user_text})],
                    )
                )

        if "order" in lowered:
            content = (
                "Order 102 belongs to customer 2 (Bob Jones). "
                "Current status: Shipped. Amount: $75.00."
            )
        elif "customer" in lowered:
            content = "Customer 2 is Bob Jones (bob@example.com)."
        elif "restock" in lowered:
            content = "Electronics items are subject to a 15% restocking fee if opened."
        elif "refund" in lowered:
            content = (
                "Refunds are permitted within 30 days of purchase for Delivered or "
                "Shipped orders. Refunds over $100 need manager approval."
            )
        else:
            content = "I can help with order status, customer lookup, and refund policy."

        return _completion(
            SimpleNamespace(role="assistant", content=content, tool_calls=None)
        )


class MockClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=MockChatCompletions())


_use_mock = not os.getenv("GROQ_API_KEY")
client = MockClient() if _use_mock else _build_live_client()


# -------------------------
# 2. SQLite database
# -------------------------

db = sqlite3.connect(":memory:")

db.executescript(
    "CREATE TABLE customers (id INT PRIMARY KEY, name TEXT, email TEXT);"
    "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, amount REAL, status TEXT);"
    "CREATE TABLE refunds (id INT PRIMARY KEY, order_id INT, amount REAL, approved INT);"
    "INSERT INTO customers VALUES (1, 'Alice Smith', 'alice@example.com');"
    "INSERT INTO customers VALUES (2, 'Bob Jones', 'bob@example.com');"
    "INSERT INTO orders VALUES (101, 1, 150.00, 'Delivered');"
    "INSERT INTO orders VALUES (102, 2, 75.00, 'Shipped');"
)


# -------------------------
# 3. Policy documents
# -------------------------

POLICIES = [
    {
        "text": "Refunds are permitted within 30 days of purchase for orders in 'Delivered' or 'Shipped' status.",
        "keywords": ["refund", "days", "limit", "period"],
    },
    {
        "text": "Refunds exceeding $100 require manager approval. Orders under $100 can be auto-approved.",
        "keywords": ["refund", "amount", "limit", "approval", "manager"],
    },
    {
        "text": "Electronics items are subject to a 15% restocking fee if opened.",
        "keywords": ["restocking", "fee", "electronics", "opened"],
    },
]


# -------------------------
# 4. Tools
# -------------------------

def get_customer_details(customer_id):
    print(f"   [API GET /api/customers/{customer_id}] Fetching customer details...")
    cursor = db.cursor()
    cursor.execute("SELECT id, name, email FROM customers WHERE id = ?", (customer_id,))
    row = cursor.fetchone()
    if row:
        return json.dumps({"id": row[0], "name": row[1], "email": row[2]})
    return json.dumps({"error": "Customer not found."})


def get_order_details(order_id):
    print(f"   [API GET /api/orders/{order_id}] Fetching order details...")
    cursor = db.cursor()
    cursor.execute(
        "SELECT id, customer_id, amount, status FROM orders WHERE id = ?",
        (order_id,),
    )
    row = cursor.fetchone()
    if row:
        return json.dumps(
            {
                "id": row[0],
                "customer_id": row[1],
                "amount": row[2],
                "status": row[3],
            }
        )
    return json.dumps({"error": "Order not found."})


def search_policy(query):
    print(f"   [RAG EXECUTE] Query: {query}")
    words = query.lower().split()
    results = []
    for policy in POLICIES:
        score = sum(1 for w in words if any(w in kw for kw in policy["keywords"]))
        if score > 0:
            results.append((score, policy["text"]))
    results.sort(reverse=True)
    extracted = [r[1] for r in results]
    return json.dumps(extracted if extracted else ["No matching policies found."])


def execute_tool(name, args):
    if name == "get_customer_details":
        return get_customer_details(int(args["customer_id"]))
    if name == "get_order_details":
        return get_order_details(int(args["order_id"]))
    if name == "search_policy":
        return search_policy(args.get("query", ""))
    return json.dumps({"error": f"Unknown tool: {name}"})


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_details",
            "description": "Retrieve customer profile information by customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The unique ID of the customer.",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Retrieve order status, amount, and owner details by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The unique order ID.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search customer refund policies using a keyword string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords for refund limits and restocking fees.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


SYSTEM_PROMPT = (
    "You are a customer support agent. Use tools to look up orders, customers, "
    "and refund policies. Never invent order status. Never write raw SQL."
)


def _message_to_dict(message):
    if isinstance(message, dict):
        return message
    payload = {"role": getattr(message, "role", "assistant")}
    if getattr(message, "content", None) is not None:
        payload["content"] = message.content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
        payload["content"] = payload.get("content") or None
    return payload


# -------------------------
# 5. Agent loop
# -------------------------

async def run_agent(user_query, max_turns=5):
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    tokens = 0

    for _ in range(max_turns):
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=tools,
        )
        usage = getattr(resp, "usage", None)
        if usage and getattr(usage, "total_tokens", None):
            tokens += usage.total_tokens

        msg = resp.choices[0].message
        history.append(_message_to_dict(msg))

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            break

        for call in tool_calls:
            args = json.loads(call.function.arguments or "{}")
            outcome = execute_tool(call.function.name, args)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": outcome,
                }
            )

    return history, tokens


def final_text(history):
    for message in reversed(history):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
    return ""


def tools_used(history):
    names = []
    for message in history:
        for call in message.get("tool_calls") or []:
            names.append(call["function"]["name"])
    return names


# -------------------------
# 6. Evaluation cases
# -------------------------

TESTS = [
    {
        "name": "Order Status Check",
        "query": "Check status of order 102.",
        "expect_tools": ["get_order_details"],
        "expect_keywords": ["102", "Shipped"],
    },
    {
        "name": "Customer Lookup",
        "query": "Look up customer 2.",
        "expect_tools": ["get_customer_details"],
        "expect_keywords": ["Bob"],
    },
    {
        "name": "Refund Policy",
        "query": "What is the refund period and approval limit?",
        "expect_tools": ["search_policy"],
        "expect_keywords": ["30 days"],
    },
    {
        "name": "Restocking Fee",
        "query": "Is there a restocking fee for electronics?",
        "expect_tools": ["search_policy"],
        "expect_keywords": ["15%"],
    },
]


def score_case(test, history):
    answer = final_text(history)
    used = tools_used(history)
    missing_tools = [name for name in test["expect_tools"] if name not in used]
    missing_keywords = [
        kw for kw in test["expect_keywords"] if kw.lower() not in answer.lower()
    ]
    passed = not missing_tools and not missing_keywords
    return {
        "passed": passed,
        "answer": answer,
        "tools": used,
        "missing_tools": missing_tools,
        "missing_keywords": missing_keywords,
    }


async def run_eval_suite(limit=None):
    print("--- Starting Systematic Agent Evaluation Suite ---")
    if _use_mock:
        print("(Using mock LLM client; set GROQ_API_KEY for live Groq runs.)")
    print()

    selected = TESTS if limit is None else TESTS[:limit]
    results = []

    for index, test in enumerate(selected, start=1):
        print(f"[Test Case {index}: {test['name']}] Query: {test['query']!r}")
        history, tokens = await run_agent(test["query"])
        outcome = score_case(test, history)
        outcome.update({"name": test["name"], "tokens": tokens, "query": test["query"]})
        results.append(outcome)

        status = "PASS" if outcome["passed"] else "FAIL"
        print(f"  Tools: {outcome['tools'] or ['(none)']}")
        print(f"  Tokens: {tokens}")
        print(f"  Answer: {outcome['answer']}")
        if outcome["missing_tools"]:
            print(f"  Missing tools: {outcome['missing_tools']}")
        if outcome["missing_keywords"]:
            print(f"  Missing keywords: {outcome['missing_keywords']}")
        print(f"  Result: {status}")
        print()

    passed = sum(1 for item in results if item["passed"])
    print(f"--- Suite complete: {passed}/{len(results)} passed ---")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the customer-support agent.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_eval_suite(limit=args.limit))
