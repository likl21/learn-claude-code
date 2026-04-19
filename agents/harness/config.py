"""
Typed configuration with validation.

Replaces scattered ``os.environ[]`` calls and hardcoded constants
with immutable, validated configuration objects.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    """
    LLM provider configuration.  Immutable after creation.

    Supports: Anthropic, MiniMax, GLM, Kimi, DeepSeek
    (any Anthropic-compatible API via ``base_url``).
    """

    api_key: str
    model_id: str
    base_url: Optional[str] = None
    max_tokens: int = 8000
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        if not self.model_id:
            raise ValueError("MODEL_ID is required")
        if self.max_tokens < 1 or self.max_tokens > 128_000:
            raise ValueError(
                f"max_tokens must be 1–128000, got {self.max_tokens}"
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"temperature must be 0.0–2.0, got {self.temperature}"
            )


@dataclass(frozen=True)
class AgentConfig:
    """Agent behavior tuning knobs."""

    workdir: Path = field(default_factory=Path.cwd)
    token_threshold: int = 100_000
    keep_recent_results: int = 3
    poll_interval: int = 5
    idle_timeout: int = 60
    max_subagent_turns: int = 30
    max_tool_output: int = 50_000
    subprocess_timeout: int = 120
    dangerous_commands: tuple[str, ...] = (
        "rm -rf /", "sudo", "shutdown", "reboot", "> /dev/",
    )

    def __post_init__(self) -> None:
        if self.token_threshold < 10_000:
            raise ValueError(
                f"token_threshold must be >= 10000, got {self.token_threshold}"
            )
        if self.keep_recent_results < 1:
            raise ValueError("keep_recent_results must be >= 1")
        if self.poll_interval < 1:
            raise ValueError("poll_interval must be >= 1")
        if self.idle_timeout < self.poll_interval:
            raise ValueError("idle_timeout must be >= poll_interval")
        if self.max_subagent_turns < 1:
            raise ValueError("max_subagent_turns must be >= 1")


@dataclass(frozen=True)
class AppConfig:
    """
    Top-level configuration.  Aggregates all sub-configs.

    Directory fields are created lazily (not at config time)
    to avoid side effects during validation.
    """

    llm: LLMConfig
    agent: AgentConfig = field(default_factory=AgentConfig)
    log_dir: Path = field(
        default_factory=lambda: Path.cwd() / ".logs"
    )
    tasks_dir: Path = field(
        default_factory=lambda: Path.cwd() / ".tasks"
    )
    team_dir: Path = field(
        default_factory=lambda: Path.cwd() / ".team"
    )
    skills_dir: Path = field(
        default_factory=lambda: Path.cwd() / "skills"
    )
    transcript_dir: Path = field(
        default_factory=lambda: Path.cwd() / ".transcripts"
    )

    def ensure_dirs(self) -> None:
        """Create all working directories if they don't exist."""
        for d in (
            self.log_dir,
            self.tasks_dir,
            self.team_dir,
            self.transcript_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    """
    Load and validate configuration from environment variables.

    Required env vars:
        ANTHROPIC_API_KEY
        MODEL_ID

    Optional env vars:
        ANTHROPIC_BASE_URL, MAX_TOKENS, TEMPERATURE,
        TOKEN_THRESHOLD, POLL_INTERVAL, IDLE_TIMEOUT
    """
    from dotenv import load_dotenv
    load_dotenv(override=True)

    try:
        llm = LLMConfig(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model_id=os.environ.get("MODEL_ID", ""),
            base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
            max_tokens=int(os.environ.get("MAX_TOKENS", "8000")),
            temperature=float(os.environ.get("TEMPERATURE", "0.0")),
        )

        agent_kwargs: dict = {}
        if os.environ.get("TOKEN_THRESHOLD"):
            agent_kwargs["token_threshold"] = int(os.environ["TOKEN_THRESHOLD"])
        if os.environ.get("POLL_INTERVAL"):
            agent_kwargs["poll_interval"] = int(os.environ["POLL_INTERVAL"])
        if os.environ.get("IDLE_TIMEOUT"):
            agent_kwargs["idle_timeout"] = int(os.environ["IDLE_TIMEOUT"])

        agent = AgentConfig(**agent_kwargs)
        return AppConfig(llm=llm, agent=agent)

    except (ValueError, KeyError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
