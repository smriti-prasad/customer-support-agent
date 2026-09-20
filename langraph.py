import json
import sqlite3
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables and initialize Groq client
load_dotenv()
client = AsyncGroq()

# 1. Initialize SQLite database
db = sqlite3.connect(":memory:")
db.executescript(
    "CREATE TABLE customers (id INT, name TEXT);"
    "CREATE TABLE orders (id INT, customer_id INT, amount REAL, status TEXT);"
    "INSERT INTO customers VALUES (1, 'Alice Smith');"
    "INSERT INTO orders VALUES (101, 1, 150.00, 'Delivered');"
    "INSERT INTO orders VALUES (102, 1, 50.00, 'Shipped');"
)

# 2. Mock vector policy lookup
POLICIES = [
    {"text": "Refunds are permitted within 30 days of purchase.", "keywords": ["refund", "days"]},
    {"text": "Refunds exceeding $100 require manager approval.", "keywords": ["refund", "limit", "manager"]}
]


# Tools (REST APIs)
def get_order_details(order_id):
    print(f"  [API GET /api/orders/{order_id}] Fetching order status...")
    cursor = db.cursor()
    cursor.execute("SELECT id, customer_id, amount, status FROM orders WHERE id = ?", (order_id,))
    row = cursor.fetchone()
    if row:
        return json.dumps({"id": row[0], "customer_id": row[1], "amount": row[2], "status": row[3]})
    return json.dumps({"error": "Order not found."})

def search_policy(query):
    print(f"  [RAG GATE] Searching for: {query}")
    words = query.lower().split()
    matches = [p["text"] for p in POLICIES if any(kw in words for kw in p["keywords"])]
    return json.dumps(matches if matches else ["No matching policies found."])

# 3. Lightweight StateGraph Simulator
class StateGraph:
    def __init__(self, state_schema):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}
        self.entry_point = None

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, start, end):
        self.edges[start] = end

    def add_conditional_edges(self, start, condition_func, routing_map):
        self.conditional_edges[start] = (condition_func, routing_map)

    def set_entry_point(self, name):
        self.entry_point = name

    def compile(self):
        return CompiledGraph(self)

class CompiledGraph:
    def __init__(self, graph):
        self.graph = graph

    async def invoke(self, initial_state):
        state = initial_state.copy()
        current = self.graph.entry_point
        print(f"--- [LangGraph] Starting execution at: {current} ---")
        
        # Run state machine loop
        limit = 10
        step = 0
        while current and current != "END" and step < limit:
            step += 1
            print(f"  [Node: {current}] Executing...")
            node_func = self.graph.nodes[current]
            update = await node_func(state)
            state.update(update)
            
            # Routing logic
            if current in self.graph.edges:
                current = self.graph.edges[current]
            elif current in self.graph.conditional_edges:
                cond_func, r_map = self.graph.conditional_edges[current]
                decision = cond_func(state)
                current = r_map.get(decision)
                print(f"  [Conditional Edge Router] Decision '{decision}' -> Next: {current}")
            else:
                current = None
        return state

async def triage_node(state):
    triage_prompt = (
        "You are a triage node. Classify the user query into exactly one of three categories:\n"
        "DATABASE (if it requires looking up customer orders or transaction history via API)\n"
        "POLICY (if it asks about refund guidelines or restocking fees)\n"
        "Respond with exactly one word: DATABASE or POLICY."
    )

    resp = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": triage_prompt},
            {"role": "user", "content": state["query"]}
        ]
    )

    category = resp.choices[0].message.content.strip().upper()
    print(f"    [Triage Classifier Output] -> {category}")
    return {"category": category}

def route_triage(state):
    return state["category"]

async def db_api_agent_node(state):
    resp = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a database access assistant. Retrieve order details by calling get_order_details. Never attempt to write raw database queries."},
            {"role": "user", "content": state["query"]}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Retrieve order status, amount, and owner details by order ID.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "integer"}},
                    "required": ["order_id"]
                }
            }
        }]
    )
    msg = resp.choices[0].message
    retrieved = []
    if msg.tool_calls:
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            retrieved.append(get_order_details(int(args.get("order_id"))))
    return {"retrieved_context": retrieved}

async def rag_agent_node(state):
    resp = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a policy search assistant. Search policies using search_policy."},
            {"role": "user", "content": state["query"]}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "search_policy",
                "description": "Search refund policy.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }]
    )
    msg = resp.choices[0].message
    retrieved = []
    if msg.tool_calls:
        call = msg.tool_calls[0]
        args = json.loads(call.function.arguments)
        retrieved.append(search_policy(args.get("query")))
    return {"retrieved_context": retrieved}

async def compiler_node(state):
    context_str = "\n".join(state.get("retrieved_context", []))
    resp = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": f"Compile the user request and retrieved tools context into a final answer. Context:\n{context_str}"},
            {"role": "user", "content": state["query"]}
        ]
    )
    return {"response": resp.choices[0].message.content}

# 5. Build and Compile Graph
builder = StateGraph(state_schema=dict)

builder.add_node("triage", triage_node)
builder.add_node("db_api_agent", db_api_agent_node)
builder.add_node("rag_agent", rag_agent_node)
builder.add_node("compiler", compiler_node)

builder.set_entry_point("triage")

builder.add_conditional_edges(
    "triage",
    route_triage,
    {
        "DATABASE": "db_api_agent",
        "POLICY": "rag_agent"
    }
)
builder.add_edge("db_api_agent", "compiler")
builder.add_edge("rag_agent", "compiler")
builder.add_edge("compiler", "END")

graph = builder.compile()

async def main():
    # Execute Graph
    initial = {"query": "Check status of order 102."}
    final_state = await graph.invoke(initial)
    print(f"Final Answer: {final_state.get('response')}")

if __name__ == "__main__":
    asyncio.run(main())



