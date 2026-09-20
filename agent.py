import json
from pyexpat.errors import messages
import sqlite3
import asyncio
import os

from groq import AsyncGroq
from dotenv import load_dotenv


# -------------------------
# 1. Setup Groq
# -------------------------

load_dotenv()

client = AsyncGroq(
    api_key=os.getenv("GROQ_API_KEY")
)


# -------------------------
# 2. Initialize SQLite database
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
# 3. Mock policy documents
# -------------------------

POLICIES = [
    {
        "text": "Refunds are permitted within 30 days of purchase for orders in 'Delivered' or 'Shipped' status.",
        "keywords": ["refund", "days", "limit", "period"]
    },
    {
        "text": "Refunds exceeding $100 require manager approval. Orders under $100 can be auto-approved.",
        "keywords": ["refund", "amount", "limit", "approval", "manager"]
    },
    {
        "text": "Electronics items are subject to a 15% restocking fee if opened.",
        "keywords": ["restocking", "fee", "electronics", "opened"]
    }
]


# -------------------------
# 4. Database tools
# -------------------------

def get_customer_details(customer_id):

    print(
        f"   [API GET /api/customers/{customer_id}] "
        "Fetching customer details from database..."
    )

    cursor = db.cursor()

    cursor.execute(
        "SELECT id, name, email FROM customers WHERE id = ?",
        (customer_id,)
    )

    row = cursor.fetchone()

    if row:
        return json.dumps({
            "id": row[0],
            "name": row[1],
            "email": row[2]
        })

    return json.dumps({
        "error": "Customer not found."
    })


def get_order_details(order_id):

    print(
        f"   [API GET /api/orders/{order_id}] "
        "Fetching order details from database..."
    )

    cursor = db.cursor()

    cursor.execute(
        "SELECT id, customer_id, amount, status FROM orders WHERE id = ?",
        (order_id,)
    )

    row = cursor.fetchone()

    if row:
        return json.dumps({
            "id": row[0],
            "customer_id": row[1],
            "amount": row[2],
            "status": row[3]
        })

    return json.dumps({
        "error": "Order not found."
    })


# -------------------------
# 5. RAG tool
# -------------------------

def search_policy(query):

    print(f"   [RAG EXECUTE] Query: {query}")

    words = query.lower().split()

    results = []

    for policy in POLICIES:

        score = sum(
            1
            for w in words
            if any(w in kw for kw in policy["keywords"])
        )

        if score > 0:
            results.append(
                (score, policy["text"])
            )

    # Highest relevance score first
    results.sort(reverse=True)

    # Remove the scores and keep only policy text
    extracted = [
        r[1]
        for r in results
    ]

    return json.dumps(
        extracted
        if extracted
        else ["No matching policies found."]
    )


# -------------------------
# 6. Tool schemas
# -------------------------

tools = [

    {
        "type": "function",
        "function": {
            "name": "get_customer_details",

            "description":
                "Retrieve customer profile information by customer ID.",

            "parameters": {
                "type": "object",

                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description":
                            "The unique ID of the customer."
                    }
                },

                "required": ["customer_id"]
            }
        }
    },


    {
        "type": "function",
        "function": {
            "name": "get_order_details",

            "description":
                "Retrieve order status, amount, and owner details by order ID.",

            "parameters": {
                "type": "object",

                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description":
                            "The unique order ID."
                    }
                },

                "required": ["order_id"]
            }
        }
    },


    {
        "type": "function",
        "function": {
            "name": "search_policy",

            "description":
                "Search customer refund policies using a keyword string.",

            "parameters": {
                "type": "object",

                "properties": {
                    "query": {
                        "type": "string",
                        "description":
                            "The keyword search term for locating refund limits and restocking fee rules."
                    }
                },

                "required": ["query"]
            }
        }
    }
]

# state router function 

async def run_state_graph(user_query):
    print(f"USER: {user_query}")

    #STEP 1: TRIAGE NODE
    triage_prompt = ("You are a triage node. Classify the user query into exactly one of three categories:\n"
        "DATABASE (if it requires looking up customer orders or transaction history via API)\n"
        "POLICY (if it asks about refund guidelines, restocking fees, or approval rules)\n"
        "GENERAL (for other questions)\n"
        "Respond with exactly one word: DATABASE, POLICY, or GENERAL.")

    triage_resp = await client.chat.completions.create(
    
                model="openai/gpt-oss-120b",
    
                messages=[
                    {"role": "system", "content": triage_prompt},
                    {"role": "user", "content": user_query}
                ],
                tools=tools
            )
    category = triage_resp.choices[0].message.content.strip().upper()
    print(f"  [STATE: Triage Classifier] -> {category}")

    #route execution to specialists
    messages = [{"role":"user", "content": user_query}]

    if "DATABASE" in category:
        print(" [STATE: DB API Routing] -> Routing to database tools...")

        db_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_customer_details",
                    "description": "Retrieve customer profile information by customer ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "integer", "description": "The unique ID of the customer."}
                        },
                        "required": ["customer_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_order_details",
                    "description": "Retrieve order status, amount, and owner details by order ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "integer", "description": "The unique order ID."}
                        },
                        "required": ["order_id"]
                    }
                }
            }
        ]

        resp = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages = [
                {"role": "system", "content": "You are a database access assistant. Retrieve customer or order details by calling get_customer_details or get_order_details. Never attempt to write raw database queries."},
                {"role": "user", "content": user_query}
            ],
            tools=db_tools
        )

        msg = resp.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                if name == "get_customer_details":
                    outcome = get_customer_details(int(args["customer_id"]))
                elif name == "get_order_details":
                    outcome = get_order_details(int(args["order_id"]))
                else:
                    outcome = "Error: Tool not found."

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": outcome
                })

    elif "POLICY" in category:
        print(" [STATE: RAG Routing] -> Routing to policy retrieval tools...")

        rag_tools =[{
            "type": "function",
            "function" : {
                "name": "search_policy",
                "description": "Search policy guidelines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The keyword search term for locating refund limits and restocking fee rules."}
                    },
                    "required": ["query"]
                }
            }
        }]

        resp = await client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages = [
                        {"role": "system", "content": "You are a policy search assistant. Search policies using search_policy."},
                        {"role": "user", "content": user_query}
                    ],
                    tools=rag_tools
                ) 

        msg = resp.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            call = msg.tool_calls[0]
            args = json.loads(call.function.arguments)
            outcome = search_policy(args.get("query"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome})

    else:
        print(" [STATE: GENERAL Routing] -> Routing to general response generation...")

        resp = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages = [
                {"role": "system", "content": "You are a general customer support assistant. Provide helpful responses to user inquiries."},
                {"role": "user", "content": user_query}
            ]
        )

        msg = resp.choices[0].message
        messages.append(msg)
        return messages

    #step 3: compiler node
    print(" [STATE: Compiler Node] -> Compiling final response...")

    resp2 = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages =[
            {"role":"system", "content": "You are a support agent. Compile the user request and retrieved tools context into a final answer."}
        ] + messages
    )
    messages.append(resp2.choices[0].message)
    return messages

def pretty_print_run(messages):
    print("\n" + "=" * 50)
    print("AGENT RUN")
    print("=" * 50)

    for message in messages:

        # User message
        if isinstance(message, dict) and message["role"] == "user":
            print("\nUSER:")
            print(message["content"])

        # Tool result
        elif isinstance(message, dict) and message["role"] == "tool":
            print("\nTOOL RESULT:")
            print(message["content"])

        # Assistant message
        elif hasattr(message, "role") and message.role == "assistant":

            # LLM requested a tool
            if message.tool_calls:
                for call in message.tool_calls:
                    print("\nLLM → TOOL:")
                    print(f"Tool: {call.function.name}")
                    print(f"Arguments: {call.function.arguments}")

            # LLM produced final answer
            elif message.content:
                print("\nFINAL ANSWER:")
                print(message.content)

    print("\n" + "=" * 50)

# Run DB path
async def main():
    print("--- RUN 1: Relational Query ---")
    results1 = await run_state_graph("Check status of order 102.")
    pretty_print_run(results1)
    print()

    # Run Policy path
    print("--- RUN 2: Policy Query ---")
    results2 = await run_state_graph("Is there a restocking fee for items?")
    pretty_print_run(results2)

asyncio.run(main())