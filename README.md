# customer-support-agent

Barebones customer-support agent with Groq tool calling, an in-memory order database, and a policy RAG stub.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
export GROQ_API_KEY=your_key
```

## Run

```bash
python agent.py
python agent-eval.py
python -m pytest -q
```

`agent-eval.py` talks to Groq via `groq_client.chat.completions.create`. Do not import `xmlrpc.client` as `client` — that stdlib module has no `chat` attribute.
