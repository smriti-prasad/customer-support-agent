import asyncio
import json
import os

from dotenv import load_dotenv
from groq import AsyncGroq

from agent import (
    get_customer_details,
    get_order_details,
    search_policy,
    tools,
)


# Groq chat client. Do not bind this name to xmlrpc.client — that module has
# no `.chat` attribute and is what caused:
#   AttributeError: module 'xmlrpc.client' has no attribute 'chat'
load_dotenv()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY") or "missing-groq-key")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are a customer support agent. Use tools to look up orders, customers, "
    "and refund policies. Never invent order status. Never write raw SQL."
)


def execute_tool(name, args):
    if name == "get_customer_details":
        return get_customer_details(int(args["customer_id"]))
    if name == "get_order_details":
        return get_order_details(int(args["order_id"]))
    if name == "search_policy":
        return search_policy(args.get("query", ""))
    return json.dumps({"error": f"Unknown tool: {name}"})


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


async def run_agent(user_query, max_turns=5):
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    tokens = 0

    for _ in range(max_turns):
        resp = await groq_client.chat.completions.create(
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
        "expect_keywords": ["30"],
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
    return {
        "passed": not missing_tools and not missing_keywords,
        "answer": answer,
        "tools": used,
        "missing_tools": missing_tools,
        "missing_keywords": missing_keywords,
    }


def assert_llm_client_is_groq():
    """Fail fast with the original error shape if client is still xmlrpc."""
    chat = getattr(groq_client, "chat", None)
    if chat is None:
        raise AttributeError(
            f"{type(groq_client).__module__!r} object {type(groq_client).__name__!r} "
            "has no attribute 'chat' (expected groq.AsyncGroq, not xmlrpc.client)"
        )
    if not hasattr(chat, "completions"):
        raise AttributeError("LLM client.chat has no attribute 'completions'")


async def run_eval_suite():
    print("--- Starting Systematic Agent Evaluation Suite ---")
    print(f"LLM client: {type(groq_client).__module__}.{type(groq_client).__name__}")
    assert_llm_client_is_groq()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Refusing to run the suite against xmlrpc "
            "or a mock. Export GROQ_API_KEY and retry."
        )
    print()

    results = []
    for index, test in enumerate(TESTS, start=1):
        print(f"[Test Case {index}: {test['name']}] Query: {test['query']!r}")
        history, tokens = await run_agent(test["query"])
        outcome = score_case(test, history)
        outcome.update({"name": test["name"], "tokens": tokens})
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


if __name__ == "__main__":
    asyncio.run(run_eval_suite())
