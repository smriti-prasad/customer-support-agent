"""Guard against the xmlrpc.client shadowing bug."""

import ast
import importlib.util
import pathlib
import xmlrpc.client as xmlrpc_client

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_xmlrpc_client_has_no_chat():
    assert not hasattr(xmlrpc_client, "chat")
    try:
        xmlrpc_client.chat
        raised = False
    except AttributeError as exc:
        raised = True
        assert "chat" in str(exc)
    assert raised


def test_eval_source_does_not_import_xmlrpc_as_client():
    source = (ROOT / "agent-eval.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("xmlrpc"):
            for alias in node.names:
                assert (alias.asname or alias.name) != "client"
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[-1]
                assert not (alias.name.startswith("xmlrpc") and bound == "client")


def test_eval_and_agent_clients_expose_chat():
    agent = _load(ROOT / "agent.py", "agent_under_test")
    eval_mod = _load(ROOT / "agent-eval.py", "agent_eval_under_test")
    for client in (agent.groq_client, eval_mod.groq_client):
        assert type(client).__module__.startswith("groq")
        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")
        assert hasattr(client.chat.completions, "create")


def test_run_agent_uses_groq_not_xmlrpc():
    source = (ROOT / "agent-eval.py").read_text(encoding="utf-8")
    assert "await groq_client.chat.completions.create" in source
    assert "from xmlrpc import client" not in source
    assert "import xmlrpc.client as client" not in source
