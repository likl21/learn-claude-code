"""
Observability layer for agent harness.

Three components, zero external dependencies:
1. StructuredLogger — JSON Lines event logging
2. Metrics         — Thread-safe counters for LLM calls, tokens, cost
3. Tracer          — Lightweight span-based tracing (OpenTelemetry-compatible concepts)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional


# ── Helpers ────────────────────────────────────────────────

def _uuid_short() -> str:
    return str(uuid.uuid4())[:8]


def _safe_serialize(obj: Any) -> Any:
    """Make objects JSON-serializable."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return str(obj)
    return obj


# ── Structured Logger ──────────────────────────────────────

class StructuredLogger:
    """
    JSON Lines logger.  Each line is a self-contained event dict.

    Output targets:
      - File   : all events at `level` or above
      - Console: WARNING and above (coloured)
    """

    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
    ):
        self.name = name
        self.logger = logging.getLogger(f"harness.{name}")
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Avoid duplicate handlers on re-init
        if not self.logger.handlers:
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_dir / f"{name}.jsonl")
                fh.setLevel(level)
                fh.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)

    def event(
        self,
        event_type: str,
        level: int = logging.INFO,
        **data: Any,
    ) -> dict:
        """Emit a structured event.  Returns the event dict for testing."""
        record = {
            "ts": time.time(),
            "event": event_type,
            "logger": self.name,
            "thread": threading.current_thread().name,
            **data,
        }
        self.logger.log(level, json.dumps(record, default=_safe_serialize))
        return record

    def debug(self, event_type: str, **data: Any) -> dict:
        return self.event(event_type, logging.DEBUG, **data)

    def info(self, event_type: str, **data: Any) -> dict:
        return self.event(event_type, logging.INFO, **data)

    def warning(self, event_type: str, **data: Any) -> dict:
        return self.event(event_type, logging.WARNING, **data)

    def error(self, event_type: str, **data: Any) -> dict:
        return self.event(event_type, logging.ERROR, **data)


# ── Metrics Collector ──────────────────────────────────────

# Default pricing per 1M tokens (USD).  Override via Metrics.set_pricing().
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514":  {"input": 3.0,  "output": 15.0},
    "claude-opus-4-20250514":    {"input": 15.0, "output": 75.0},
    "claude-haiku-3-20250307":   {"input": 0.25, "output": 1.25},
}


@dataclass
class Metrics:
    """
    In-process metrics collector.  All operations are thread-safe.

    Tracks: LLM calls, tool calls, tokens, estimated cost, errors,
            compactions, subagent spawns.
    """

    llm_calls: int = 0
    tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    errors: int = 0
    compactions: int = 0
    subagent_spawns: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _pricing: dict[str, dict[str, float]] = field(
        default_factory=lambda: dict(_DEFAULT_PRICING), repr=False,
    )

    # ── Mutators ───────────────────────────────────────────

    def inc(self, name: str, amount: int = 1) -> None:
        """Increment a counter by name."""
        with self._lock:
            current = getattr(self, name, None)
            if current is None:
                raise AttributeError(f"Unknown metric: {name}")
            setattr(self, name, current + amount)

    def add_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-sonnet-4-20250514",
    ) -> float:
        """Record an LLM call.  Returns estimated cost (USD)."""
        prices = self._pricing.get(model, {"input": 3.0, "output": 15.0})
        cost = (
            input_tokens * prices["input"]
            + output_tokens * prices["output"]
        ) / 1_000_000
        with self._lock:
            self.llm_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost
        return cost

    def set_pricing(self, model: str, input_per_m: float, output_per_m: float) -> None:
        """Override pricing for a model."""
        with self._lock:
            self._pricing[model] = {"input": input_per_m, "output": output_per_m}

    # ── Readers ────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe snapshot of all public counters."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cost_usd": self.total_cost_usd,
                "errors": self.errors,
                "compactions": self.compactions,
                "subagent_spawns": self.subagent_spawns,
            }

    def report(self) -> str:
        """One-line human-readable summary."""
        s = self.snapshot()
        return (
            f"LLM calls: {s['llm_calls']} | "
            f"Tool calls: {s['tool_calls']} | "
            f"Tokens: {s['total_input_tokens']}in/{s['total_output_tokens']}out | "
            f"Est. cost: ${s['total_cost_usd']:.4f} | "
            f"Errors: {s['errors']} | "
            f"Compactions: {s['compactions']}"
        )

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self.llm_calls = 0
            self.tool_calls = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cost_usd = 0.0
            self.errors = 0
            self.compactions = 0
            self.subagent_spawns = 0


# ── Tracing (Span-based) ──────────────────────────────────

@dataclass
class Span:
    """
    A single trace span.

    Concepts align with OpenTelemetry:
      - trace_id groups all spans in one agent session
      - span_id is unique per operation
      - parent_id links nested spans
    """

    name: str
    trace_id: str
    span_id: str = field(default_factory=_uuid_short)
    parent_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """
    Lightweight span-based tracer.

    Usage::

        tracer = Tracer(logger)
        with tracer.span("llm_call", model="claude-3") as s:
            response = client.messages.create(...)
            s.attributes["tokens"] = response.usage.input_tokens
    """

    def __init__(self, logger: Optional[StructuredLogger] = None):
        self.logger = logger
        self._trace_id = _uuid_short()
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    @contextmanager
    def span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        **attrs: Any,
    ) -> Generator[Span, None, None]:
        s = Span(
            name=name,
            trace_id=self._trace_id,
            parent_id=parent_id,
            start_time=time.time(),
            attributes=dict(attrs),
        )
        try:
            yield s
        except Exception as e:
            s.status = "error"
            s.attributes["error"] = str(e)
            raise
        finally:
            s.end_time = time.time()
            with self._lock:
                self._spans.append(s)
            if self.logger:
                self.logger.event(
                    "span.end",
                    name=s.name,
                    duration_ms=round(s.duration_ms(), 2),
                    trace_id=s.trace_id,
                    span_id=s.span_id,
                    parent_id=s.parent_id,
                    status=s.status,
                    **{k: v for k, v in s.attributes.items() if k != "error"},
                    **({"error": s.attributes["error"]} if "error" in s.attributes else {}),
                )

    def get_spans(self) -> list[dict[str, Any]]:
        """Return all recorded spans as dicts."""
        with self._lock:
            return [asdict(s) for s in self._spans]

    def reset(self) -> None:
        """Clear spans and generate new trace_id."""
        with self._lock:
            self._spans.clear()
            self._trace_id = _uuid_short()
