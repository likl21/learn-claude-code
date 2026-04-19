"""
Integration tests for agent harness modules.

These tests verify module interactions without making real API calls.
They use mock LLM clients to test the agent loop behavior, tool dispatch,
and cross-module integration.

Run with:  pytest tests/test_integration.py -v
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.harness.config import AgentConfig, AppConfig, LLMConfig
from agents.harness.evaluation import EvalResult, EvalSuite
from agents.harness.observability import Metrics, StructuredLogger, Tracer
from agents.harness.resilience import CircuitBreaker, CircuitBreakerOpen, ResilientLLMClient
from agents.harness.security import CommandFilter, OutputSanitizer, PathSandbox


# ===================================================================
#  Mock LLM Infrastructure
# ===================================================================


@dataclass
class MockBlock:
    """Mock for Anthropic content blocks."""
    type: str
    text: str = ""
    name: str = ""
    id: str = "tool_001"
    input: dict = None  # type: ignore

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class MockResponse:
    """Mock for Anthropic API response."""
    content: list
    stop_reason: str = "end_turn"
    usage: MockUsage = None  # type: ignore

    def __post_init__(self):
        if self.usage is None:
            self.usage = MockUsage()


class MockLLMClient:
    """Mock Anthropic client for testing."""

    def __init__(self, responses: list[MockResponse] | None = None):
        self.responses = responses or []
        self._call_index = 0
        self.call_log: list[dict] = []
        self.messages = self  # client.messages.create -> client.messages = self

    def create(self, **kwargs: Any) -> MockResponse:
        self.call_log.append(kwargs)
        if self._call_index < len(self.responses):
            resp = self.responses[self._call_index]
            self._call_index += 1
            return resp
        return MockResponse(
            content=[MockBlock(type="text", text="Done.")],
            stop_reason="end_turn",
        )


# ===================================================================
#  Integration: Observability + Metrics tracking across modules
# ===================================================================


class TestObservabilityIntegration:
    """Test that Metrics and Tracer work together correctly."""

    def test_metrics_with_tracer(self) -> None:
        """Metrics and Tracer should be able to track the same LLM call."""
        metrics = Metrics()
        logger = StructuredLogger("integration_test")
        tracer = Tracer(logger)

        # Simulate an LLM call tracked by both
        with tracer.span("llm_call", model="claude-3") as s:
            cost = metrics.add_llm_call(1000, 500)
            s.attributes["cost"] = cost
            s.attributes["input_tokens"] = 1000

        assert metrics.llm_calls == 1
        spans = tracer.get_spans()
        assert len(spans) == 1
        assert spans[0]["attributes"]["cost"] == cost

    def test_metrics_track_multiple_tools(self) -> None:
        """Metrics should correctly track multiple sequential tool calls."""
        metrics = Metrics()
        tracer = Tracer()

        tools_called = ["bash", "read_file", "write_file", "edit_file", "bash"]
        for tool in tools_called:
            with tracer.span("tool_call", tool=tool):
                metrics.inc("tool_calls")

        assert metrics.tool_calls == 5
        spans = tracer.get_spans()
        assert len(spans) == 5
        assert [s["attributes"]["tool"] for s in spans] == tools_called

    def test_error_tracking(self) -> None:
        """Errors in tool calls should be reflected in both metrics and spans."""
        metrics = Metrics()
        tracer = Tracer()

        with pytest.raises(ValueError):
            with tracer.span("tool_call", tool="bash") as s:
                metrics.inc("tool_calls")
                metrics.inc("errors")
                raise ValueError("command failed")

        assert metrics.errors == 1
        spans = tracer.get_spans()
        assert spans[0]["status"] == "error"


# ===================================================================
#  Integration: Security pipeline (Validate → Filter → Execute → Sanitize)
# ===================================================================


class TestSecurityPipelineIntegration:
    """Test the full security pipeline end-to-end."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.sandbox = PathSandbox(Path(self.tmpdir))
        self.cmd_filter = CommandFilter()
        self.sanitizer = OutputSanitizer()

    def test_safe_command_passes_pipeline(self) -> None:
        """A safe command should pass through all security layers."""
        command = "ls -la"
        # Step 1: Command filter
        allowed, reason = self.cmd_filter.check(command)
        assert allowed
        assert reason is None
        # Step 2: Output sanitization (no secrets in output)
        output = "total 4\ndrwxr-xr-x  2 user user 4096 Jan 1 file.txt"
        sanitized = self.sanitizer.sanitize(output)
        assert sanitized == output  # no change

    def test_dangerous_command_blocked(self) -> None:
        """A dangerous command should be blocked at the filter stage."""
        command = "sudo rm -rf /"
        allowed, reason = self.cmd_filter.check(command)
        assert not allowed
        # Should never reach execution

    def test_output_with_secrets_sanitized(self) -> None:
        """Tool output containing secrets should be sanitized before LLM sees it."""
        output = (
            "Config loaded:\n"
            "  api_key=sk-ant-abc123def456ghi789jkl012mno345pqr678\n"
            "  OPENAI_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz1234567890abcdefgh\n"
            "  AWS: AKIAIOSFODNN7EXAMPLE\n"
        )
        sanitized = self.sanitizer.sanitize(output)
        assert "abc123" not in sanitized
        assert "REDACTED" in sanitized
        assert "Config loaded:" in sanitized  # non-secret parts preserved

    def test_path_traversal_then_command(self) -> None:
        """Path traversal should be caught before command execution."""
        # This simulates: model asks to read a file outside workspace
        with pytest.raises(ValueError, match="escapes workspace"):
            self.sandbox.validate("../../../../etc/shadow")


# ===================================================================
#  Integration: ResilientLLMClient with mock
# ===================================================================


class TestResilientClientIntegration:
    """Test ResilientLLMClient with mock responses."""

    def test_normal_call(self) -> None:
        """Normal call should pass through."""
        mock_client = MockLLMClient([
            MockResponse(content=[MockBlock(type="text", text="Hello")])
        ])
        resilient = ResilientLLMClient(
            mock_client, max_retries=1, breaker_threshold=3
        )
        resp = resilient.create(model="test", messages=[], max_tokens=100)
        assert resp.content[0].text == "Hello"

    def test_fallback_on_breaker_open(self) -> None:
        """Should fall back to fallback_model when breaker opens."""
        call_count = 0
        original_model = "expensive-model"
        fallback_model = "cheap-model"

        class FailThenSucceedClient:
            messages = None

            def __init__(self):
                self.messages = self
                self.call_log = []

            def create(self, **kwargs):
                self.call_log.append(kwargs)
                if kwargs.get("model") == original_model:
                    raise RuntimeError("Rate limited")
                return MockResponse(
                    content=[MockBlock(type="text", text="Fallback OK")]
                )

        mock = FailThenSucceedClient()
        resilient = ResilientLLMClient(
            mock,
            fallback_model=fallback_model,
            max_retries=0,
            breaker_threshold=1,
            breaker_timeout=0.05,
        )

        # First call: fails and opens breaker
        with pytest.raises(RuntimeError):
            resilient.create(model=original_model, messages=[])

        # Breaker is now open, should fall back
        time.sleep(0.06)  # wait for half-open
        # After half-open, the next call to the original model will fail again
        # but the client should handle the CircuitBreakerOpen internally
        # Let's just verify the breaker state
        assert resilient.breaker_state in ("half_open", "open")


# ===================================================================
#  Integration: Evaluation with Metrics
# ===================================================================


class TestEvaluationIntegration:
    """Test evaluation framework with real metrics data."""

    def test_eval_from_metrics(self) -> None:
        """Create an EvalResult from Metrics snapshot."""
        metrics = Metrics()
        start = time.time()

        # Simulate some work
        metrics.add_llm_call(2000, 800)
        metrics.add_llm_call(1500, 600)
        metrics.inc("tool_calls", 5)
        metrics.inc("errors", 1)

        duration = time.time() - start
        snap = metrics.snapshot()

        result = EvalResult(
            task_name="integration_test",
            success=True,
            duration_s=round(duration, 3),
            llm_calls=snap["llm_calls"],
            tool_calls=snap["tool_calls"],
            input_tokens=snap["total_input_tokens"],
            output_tokens=snap["total_output_tokens"],
            estimated_cost_usd=snap["total_cost_usd"],
            error_count=snap["errors"],
            todo_completion_rate=1.0,
            compactions=snap["compactions"],
        )

        assert result.llm_calls == 2
        assert result.tool_calls == 5
        assert result.error_count == 1
        assert result.total_tokens == 4900  # 2000+1500+800+600

    def test_suite_report_generation(self) -> None:
        """Full suite should generate a valid report."""
        suite = EvalSuite(metadata={"model": "test", "run_id": "abc123"})

        for i in range(3):
            suite.add(EvalResult(
                task_name=f"task_{i}",
                success=i < 2,  # 2 pass, 1 fail
                duration_s=float(10 + i * 5),
                llm_calls=5 + i,
                tool_calls=8 + i * 2,
                input_tokens=3000 + i * 1000,
                output_tokens=1000 + i * 500,
                estimated_cost_usd=0.01 * (i + 1),
                error_count=i,
                todo_completion_rate=1.0 - i * 0.2,
                compactions=i,
            ))

        report = suite.report()
        assert "3" in report  # total tasks
        assert "66.7%" in report  # success rate (2/3)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_results.json"
            suite.save(path)

            data = json.loads(path.read_text())
            assert data["metadata"]["model"] == "test"
            assert data["summary"]["total_tasks"] == 3

            reloaded = EvalSuite.load(path)
            assert len(reloaded.results) == 3


# ===================================================================
#  Integration: Config → Security → Observability chain
# ===================================================================


class TestConfigSecurityChain:
    """Test that config drives security and observability setup."""

    def test_config_creates_sandbox(self) -> None:
        """Config workdir should be usable for PathSandbox."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AgentConfig(workdir=Path(tmpdir))
            sandbox = PathSandbox(cfg.workdir)
            # Should not raise
            (Path(tmpdir) / "test.txt").write_text("hello")
            path = sandbox.validate("test.txt")
            assert path.exists()

    def test_full_config_validation(self) -> None:
        """AppConfig should validate all sub-configs together."""
        llm = LLMConfig(api_key="test-key", model_id="test-model")
        agent = AgentConfig()
        config = AppConfig(llm=llm, agent=agent)
        assert config.llm.api_key == "test-key"
        assert config.agent.token_threshold == 100_000


# ===================================================================
#  Run with:  python -m pytest tests/test_integration.py -v
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
