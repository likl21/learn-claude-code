"""
Agent behavior evaluation framework.

Runs benchmark tasks and measures multi-dimensional efficiency metrics:
1. Task completion rate
2. Token efficiency (tokens per LLM call)
3. Tool call efficiency (redundant calls)
4. Error rate
5. Compression ratio
6. Plan adherence (todo completion rate)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class EvalResult:
    """Result of a single evaluation task."""

    task_name: str
    success: bool
    duration_s: float
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    error_count: int
    todo_completion_rate: float  # 0.0 – 1.0
    compactions: int
    notes: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_llm_call(self) -> float:
        if self.llm_calls == 0:
            return 0.0
        return self.total_tokens / self.llm_calls

    @property
    def tool_error_rate(self) -> float:
        if self.tool_calls == 0:
            return 0.0
        return self.error_count / self.tool_calls

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["tokens_per_llm_call"] = round(self.tokens_per_llm_call, 1)
        d["tool_error_rate"] = round(self.tool_error_rate, 4)
        return d


@dataclass
class EvalSuite:
    """
    Collection of evaluation results with aggregate reporting.

    Usage::

        suite = EvalSuite()
        suite.add(EvalResult(task_name="file_crud", success=True, ...))
        suite.add(EvalResult(task_name="multi_step", success=True, ...))
        print(suite.report())
        suite.save(Path("benchmarks/results/latest.json"))
    """

    results: list[EvalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, result: EvalResult) -> None:
        self.results.append(result)

    def summary(self) -> dict[str, Any]:
        """Compute aggregate metrics across all results."""
        if not self.results:
            return {}

        n = len(self.results)
        successes = sum(r.success for r in self.results)

        return {
            "total_tasks": n,
            "success_rate": round(successes / n, 4),
            "avg_duration_s": round(
                sum(r.duration_s for r in self.results) / n, 1
            ),
            "avg_llm_calls": round(
                sum(r.llm_calls for r in self.results) / n, 1
            ),
            "avg_tool_calls": round(
                sum(r.tool_calls for r in self.results) / n, 1
            ),
            "total_tokens": sum(r.total_tokens for r in self.results),
            "total_cost_usd": round(
                sum(r.estimated_cost_usd for r in self.results), 6
            ),
            "avg_tokens_per_call": round(
                sum(r.tokens_per_llm_call for r in self.results) / n, 0
            ),
            "avg_tool_error_rate": round(
                sum(r.tool_error_rate for r in self.results) / n, 4
            ),
            "avg_todo_completion": round(
                sum(r.todo_completion_rate for r in self.results) / n, 4
            ),
            "total_compactions": sum(r.compactions for r in self.results),
            "total_errors": sum(r.error_count for r in self.results),
        }

    def report(self) -> str:
        """Generate a human-readable evaluation report."""
        s = self.summary()
        if not s:
            return "No evaluation results."

        lines = [
            "=" * 60,
            "AGENT EVALUATION REPORT",
            "=" * 60,
            f"Tasks evaluated:       {s['total_tasks']}",
            f"Success rate:          {s['success_rate']:.1%}",
            f"Avg duration:          {s['avg_duration_s']:.1f}s",
            f"Avg LLM calls/task:    {s['avg_llm_calls']:.1f}",
            f"Avg tool calls/task:   {s['avg_tool_calls']:.1f}",
            f"Avg tokens/LLM call:   {s['avg_tokens_per_call']:.0f}",
            f"Total tokens:          {s['total_tokens']:,}",
            f"Avg tool error rate:   {s['avg_tool_error_rate']:.1%}",
            f"Avg todo completion:   {s['avg_todo_completion']:.1%}",
            f"Total estimated cost:  ${s['total_cost_usd']:.4f}",
            f"Total compactions:     {s['total_compactions']}",
            f"Total errors:          {s['total_errors']}",
            "=" * 60,
        ]

        # Per-task breakdown
        if len(self.results) > 1:
            lines.append("")
            lines.append("Per-task breakdown:")
            lines.append("-" * 60)
            for r in self.results:
                status = "PASS" if r.success else "FAIL"
                lines.append(
                    f"  [{status}] {r.task_name}: "
                    f"{r.llm_calls} calls, "
                    f"{r.total_tokens:,} tokens, "
                    f"${r.estimated_cost_usd:.4f}, "
                    f"{r.duration_s:.1f}s"
                )
            lines.append("-" * 60)

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save results to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": time.time(),
            "metadata": self.metadata,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "EvalSuite":
        """Load results from a JSON file."""
        data = json.loads(path.read_text())
        suite = cls(metadata=data.get("metadata", {}))
        for r_dict in data.get("results", []):
            # Remove computed fields
            r_dict.pop("total_tokens", None)
            r_dict.pop("tokens_per_llm_call", None)
            r_dict.pop("tool_error_rate", None)
            suite.add(EvalResult(**r_dict))
        return suite
