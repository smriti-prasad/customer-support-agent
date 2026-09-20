# customer-support-agent

Barebones customer-support agent with Groq tool calling, an in-memory order database, and a policy RAG stub.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional live LLM runs:

```bash
export GROQ_API_KEY=your_key
```

If `GROQ_API_KEY` is unset, `agent-eval.py` uses a mock chat client so the suite still runs.

## Run

```bash
python agent.py
python agent-eval.py
python -m pytest -q
```
