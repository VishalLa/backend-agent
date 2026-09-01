from __future__ import annotations

from agent.eval import EvalCase, EvalHarness
from config import Config


def test_eval_case_defaults_are_local_and_reasoning_focused():
    cfg = Config()
    case = EvalCase(
        name="routing",
        agent_key="backend",
        prompt="Inspect the auth route and tell me what to change.",
        expected_keywords=("auth", "route"),
    )

    assert case.provider == "local"
    assert case.expected_keywords == ("auth", "route")
    assert cfg.provider == "local"


def test_harness_scores_response_against_keywords_and_latency():
    harness = EvalHarness(Config())

    result = harness.score_response(
        content="The auth route is the right place to validate the session token.",
        expected_keywords=("auth", "route", "session"),
        latency_seconds=0.35,
    )

    assert result.passed is True
    assert 0.0 <= result.score <= 1.0
    assert result.latency_seconds == 0.35


def test_build_default_suite_includes_all_agent_modes():
    suite = EvalHarness(Config()).build_default_suite()

    assert {case.agent_key for case in suite} == {"backend", "ml", "git", "algorithms"}
    assert len(suite) == 4


def test_harness_runs_cases_and_reports_invocation_failures(monkeypatch):
    class FakeModel:
        def invoke(self, _messages):
            return type("Response", (), {"content": "auth route reset"})()

    class FakeChatModel:
        def __init__(self, _config):
            pass

        def get_llm(self):
            return FakeModel()

    monkeypatch.setattr("agent.eval.ChatModel", FakeChatModel)
    case = EvalCase(
        name="backend", agent_key="backend", prompt="test",
        expected_keywords=("auth", "route", "reset"),
    )

    result = EvalHarness(Config()).run([case])

    assert result["passed"] == 1
    assert result["invocation_failures"] == 0
