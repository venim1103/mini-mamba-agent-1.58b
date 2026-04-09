import importlib
import os
from unittest import mock


def _reload_context_config(env):
    with mock.patch.dict(os.environ, env, clear=True):
        import context_config

        return importlib.reload(context_config)


class TestResolveContextLength:
    def test_default_context_when_env_missing(self):
        cfg = _reload_context_config({})
        assert cfg.CONTEXT_LENGTH == 16_384
        with mock.patch.dict(os.environ, {}, clear=True):
            assert cfg.resolve_context_length() == 16_384

    def test_env_override_applies(self):
        cfg = _reload_context_config({"AGENT_CONTEXT_LENGTH": "32768"})
        assert cfg.CONTEXT_LENGTH == 32_768
        with mock.patch.dict(os.environ, {"AGENT_CONTEXT_LENGTH": "32768"}, clear=True):
            assert cfg.resolve_context_length() == 32_768

    def test_invalid_env_value_falls_back_to_default(self):
        cfg = _reload_context_config({"AGENT_CONTEXT_LENGTH": "not-a-number"})
        with mock.patch.dict(os.environ, {"AGENT_CONTEXT_LENGTH": "not-a-number"}, clear=True):
            assert cfg.resolve_context_length() == 16_384

    def test_non_positive_env_value_falls_back_to_default(self):
        cfg = _reload_context_config({"AGENT_CONTEXT_LENGTH": "0"})
        with mock.patch.dict(os.environ, {"AGENT_CONTEXT_LENGTH": "0"}, clear=True):
            assert cfg.resolve_context_length() == 16_384
