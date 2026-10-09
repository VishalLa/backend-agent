from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent import AgentRunner
from config import Config


class TestLocalDefaultConfig:
    def test_provider_defaults_to_local(self):
        cfg = Config()
        assert cfg.provider == "local"

    def test_local_model_names_are_set(self):
        cfg = Config()
        assert cfg.ollama_summary_model == "phi4-mini-reasoning"
        assert cfg.backend_model_name == "MFDoom/deepseek-coder-v2-tool-calling:16b"


class TestDispatcher:
    def test_implementation_and_commit_request_routes_to_backend_without_model(self):
        runner = AgentRunner(Config())
        prompt = (
            "The test suite is failing. Find the root cause and fix it, then add a "
            "DELETE /tasks/{task_id} endpoint, add success and 404 tests, run the "
            "full test suite, and commit the changes."
        )

        with patch.object(runner.dispatcher, "_classify_with_model") as classify:
            assert runner.route_task(prompt) == "backend"
            classify.assert_not_called()

    def test_git_only_request_routes_to_git_without_model(self):
        runner = AgentRunner(Config())

        with patch.object(runner.dispatcher, "_classify_with_model") as classify:
            assert runner.route_task("Create a release branch and push it to origin") == "git"
            classify.assert_not_called()

    def test_ml_request_routes_to_ml_without_model(self):
        runner = AgentRunner(Config())

        with patch.object(runner.dispatcher, "_classify_with_model") as classify:
            assert runner.route_task("Train a PyTorch model on this dataset") == "ml"
            classify.assert_not_called()

    def test_algorithm_request_routes_to_algorithms_without_model(self):
        runner = AgentRunner(Config())

        with patch.object(runner.dispatcher, "_classify_with_model") as classify:
            assert runner.route_task("Analyze binary search complexity") == "algorithms"
            classify.assert_not_called()

    def test_ambiguous_request_uses_model_classifier(self):
        runner = AgentRunner(Config())

        with patch.object(runner.dispatcher, "_classify_with_model", return_value="ml") as classify:
            assert runner.route_task("Which approach should I use here?") == "ml"
            classify.assert_called_once()

    def test_phase_5_ollama_tuning_vars_load_from_environment(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "5m")
        monkeypatch.setenv("OLLAMA_FLASH_ATTENTION", "false")
        monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q4_0")
        monkeypatch.setenv("OLLAMA_NUM_THREADS", "8")

        cfg = Config.from_env()

        assert cfg.ollama_keep_alive == "5m"
        assert cfg.ollama_flash_attention is False
        assert cfg.ollama_kv_cache_type == "q4_0"
        assert cfg.ollama_num_thread == 8

    def test_phase_6_tool_calling_model_is_default_for_local_env(self):
        cfg = Config.from_env()

        assert cfg.ollama_model == "MFDoom/deepseek-coder-v2-tool-calling:16b"
        assert cfg.backend_model_name == "MFDoom/deepseek-coder-v2-tool-calling:16b"
        assert cfg.ml_model_name == "MFDoom/deepseek-coder-v2-tool-calling:16b"
        assert cfg.git_model_name == "MFDoom/deepseek-coder-v2-tool-calling:16b"
        assert cfg.algo_model_name == "MFDoom/deepseek-coder-v2-tool-calling:16b"

    def test_phase_7_prompt_requires_direct_tool_calls(self):
        from pathlib import Path

        prompt = Path("agent/graphs/md/backend_agent.md").read_text(encoding="utf-8")

        assert "call the tool" in prompt.lower()
        assert "don't narrate" in prompt.lower()
