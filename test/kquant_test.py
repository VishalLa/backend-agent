"""
Phase 8: K-quant upgrade tests.

Verifies that K-quant configuration is properly supported in config.py
and can be enabled via OLLAMA_USE_KQUANT environment variable.

K-quant allocates bits more intelligently per layer than Q4_0, yielding
better accuracy at the same size and speed.
"""
from __future__ import annotations

from config import Config


class TestPhase8KQuantSupport:
    """Verify Phase 8 K-quant model selection works correctly."""

    def test_kquant_flag_is_false_by_default(self):
        """Phase 8: K-quant is opt-in, not default."""
        cfg = Config()
        assert cfg.ollama_use_kquant is False

    def test_kquant_model_constant_is_defined(self):
        """Phase 8: K-quant model name is configured."""
        from config import DEFAULT_OLLAMA_KQUANT_MODEL
        assert DEFAULT_OLLAMA_KQUANT_MODEL == "deepseek-coder-v2:16b-lite-instruct-q4_K_M"

    def test_kquant_enabled_via_environment(self, monkeypatch):
        """Phase 8: OLLAMA_USE_KQUANT=true enables K-quant for all task models."""
        monkeypatch.setenv("OLLAMA_USE_KQUANT", "true")
        cfg = Config.from_env()

        assert cfg.ollama_use_kquant is True
        # K-quant becomes the default model for all tasks when enabled
        assert cfg.backend_model_name == "deepseek-coder-v2:16b-lite-instruct-q4_K_M"
        assert cfg.ml_model_name == "deepseek-coder-v2:16b-lite-instruct-q4_K_M"
        assert cfg.git_model_name == "deepseek-coder-v2:16b-lite-instruct-q4_K_M"
        assert cfg.algo_model_name == "deepseek-coder-v2:16b-lite-instruct-q4_K_M"

    def test_direct_config_kquant_flag_selects_kquant_model(self):
        cfg = Config(ollama_use_kquant=True)
        assert cfg.get_model_for_task("backend") == "deepseek-coder-v2:16b-lite-instruct-q4_K_M"

    def test_phase7_model_is_default_when_kquant_disabled(self, monkeypatch):
        """Phase 8: K-quant disabled (default) keeps Phase 7 tool-calling model."""
        monkeypatch.setenv("OLLAMA_USE_KQUANT", "false")
        cfg = Config.from_env()

        assert cfg.ollama_use_kquant is False
        # Phase 7 tool-calling model remains default
        assert cfg.backend_model_name == "MFDoom/deepseek-coder-v2-tool-calling:16b"
        assert cfg.ml_model_name == "MFDoom/deepseek-coder-v2-tool-calling:16b"
        assert cfg.git_model_name == "MFDoom/deepseek-coder-v2-tool-calling:16b"
        assert cfg.algo_model_name == "MFDoom/deepseek-coder-v2-tool-calling:16b"

    def test_kquant_respects_various_truthy_values(self, monkeypatch):
        """Phase 8: OLLAMA_USE_KQUANT accepts multiple truthy formats."""
        for truthy in ("1", "true", "yes", "on"):
            monkeypatch.setenv("OLLAMA_USE_KQUANT", truthy)
            cfg = Config.from_env()
            assert cfg.ollama_use_kquant is True, f"Failed for value: {truthy}"

    def test_kquant_respects_falsy_values(self, monkeypatch):
        """Phase 8: OLLAMA_USE_KQUANT accepts falsy formats."""
        for falsy in ("0", "false", "no", "off"):
            monkeypatch.setenv("OLLAMA_USE_KQUANT", falsy)
            cfg = Config.from_env()
            assert cfg.ollama_use_kquant is False, f"Failed for value: {falsy}"

    def test_kquant_summary_model_unchanged(self, monkeypatch):
        """Phase 8: Summary model is not affected by K-quant flag."""
        monkeypatch.setenv("OLLAMA_USE_KQUANT", "true")
        cfg = Config.from_env()

        # Summary model is always the cheap reasoning model, independent of K-quant
        assert cfg.ollama_summary_model == "phi4-mini-reasoning"

    def test_kquant_with_api_provider_ignored(self, monkeypatch):
        """Phase 8: K-quant only applies in local provider mode."""
        # When provider='api', K-quant flag should not affect model selection
        monkeypatch.setenv("AGENT_PROVIDER", "api")
        monkeypatch.setenv("SAMBANOVA_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OLLAMA_USE_KQUANT", "true")

        cfg = Config.from_env()

        # In API mode, local K-quant setting is irrelevant
        assert cfg.provider == "api"
        assert cfg.ollama_use_kquant is True  # Flag is set but not used
        # Models are API defaults, not K-quant
        assert cfg.backend_model_name == "gpt-oss-120b"
