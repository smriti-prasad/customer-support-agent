import asyncio
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "agent-eval.py"


def _load_eval():
    spec = importlib.util.spec_from_file_location("agent_eval_under_test", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_order_status_check_case():
    module = _load_eval()
    results = asyncio.run(module.run_eval_suite(limit=1))
    assert results[0]["name"] == "Order Status Check"
    assert results[0]["passed"]
    assert "get_order_details" in results[0]["tools"]
    assert "Shipped" in results[0]["answer"]
