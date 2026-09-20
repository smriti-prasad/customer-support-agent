"""Guard against the xmlrpc.client shadowing bug in agent-eval.py."""

import ast
import importlib.util
import pathlib
import xmlrpc.client as xmlrpc_client

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "agent-eval.py"


def test_xmlrpc_client_has_no_chat():
    """Reproduce the original AttributeError shape."""
    assert not hasattr(xmlrpc_client, "chat")
    try:
        xmlrpc_client.chat.completions.create
        raised = False
    except AttributeError as exc:
        raised = True
        assert "chat" in str(exc)
    assert raised


def test_agent_eval_does_not_bind_xmlrpc_client():
    source = EVAL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[-1]
                assert not (
                    alias.name.startswith("xmlrpc") and bound == "client"
                ), "client must not be imported from xmlrpc"
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("xmlrpc"):
            for alias in node.names:
                bound = alias.asname or alias.name
                assert bound != "client", "client must not be imported from xmlrpc"


def test_loaded_client_exposes_chat_completions():
    spec = importlib.util.spec_from_file_location("agent_eval_under_test", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module.client, "chat")
    assert hasattr(module.client.chat, "completions")
    assert hasattr(module.client.chat.completions, "create")
    assert module.client.__class__.__name__ != "ModuleType"
