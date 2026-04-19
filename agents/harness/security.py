"""
Security hardening for agent harness.

Four layers:
1. PathSandbox      — Symlink-aware path validation
2. CommandFilter    — Regex-based command blocking (not substring matching)
3. OutputSanitizer  — Mask secrets in tool output before sending to LLM
4. InputValidator   — Validate tool inputs
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ── Path Sandbox (hardened) ────────────────────────────────

class PathSandbox:
    """
    Hardened path sandbox.

    Improvements over the basic ``safe_path()`` in teaching sessions:
    1. Resolves symlinks *before* boundary check
    2. Blocks access to dotfiles with sensitive patterns
    3. Blocks common credential/secret file paths
    """

    BLOCKED_PATTERNS: list[str] = [
        r"\.env($|\.)",         # .env, .env.local, .env.production
        r"\.git/config$",       # git credentials
        r"id_rsa",              # SSH private keys
        r"id_ed25519",          # SSH private keys
        r"\.ssh/",              # SSH directory
        r"credentials",         # generic credentials
        r"\.aws/",              # AWS credentials
        r"\.kube/config",       # Kubernetes config
        r"\.npmrc$",            # npm tokens
        r"\.pypirc$",          # PyPI tokens
    ]

    def __init__(self, workdir: Path):
        self.workdir = workdir.resolve()
        self._blocked_re = [
            re.compile(p, re.IGNORECASE) for p in self.BLOCKED_PATTERNS
        ]

    def validate(self, path_str: str) -> Path:
        """
        Validate and resolve a path.

        Returns the resolved absolute ``Path`` if safe.
        Raises ``ValueError`` if the path is outside the workspace,
        follows a symlink escaping the workspace, or matches a
        sensitive file pattern.
        """
        # Resolve relative to workdir
        path = (self.workdir / path_str).resolve()

        # Check 1: within workdir after resolution
        if not path.is_relative_to(self.workdir):
            raise ValueError(f"Path escapes workspace: {path_str}")

        # Check 2: symlink target also within workdir
        if path.is_symlink():
            real = Path(path).resolve()
            if not real.is_relative_to(self.workdir):
                raise ValueError(
                    f"Symlink target escapes workspace: {path_str} -> {real}"
                )

        # Check 3: not a sensitive file pattern
        relative = str(path.relative_to(self.workdir))
        for pattern in self._blocked_re:
            if pattern.search(relative):
                raise ValueError(f"Access to sensitive path blocked: {path_str}")

        return path


# ── Command Filter (regex-based) ──────────────────────────

class CommandFilter:
    """
    Regex-based command filtering with two severity levels.

    - BLOCK: hard-blocked, command will not execute
    - WARN:  allowed but logged as suspicious

    Improvements over substring matching:
    - ``rm -r -f /`` (flag reordering) is caught
    - ``curl ... | sh`` (pipe to shell) is caught
    - Fork bombs are caught
    """

    BLOCK_PATTERNS: list[str] = [
        r"\brm\s+(-[a-zA-Z]*\s+)*-?r\s*f?\s+/\s*$",  # rm -rf /
        r"\brm\s+(-[a-zA-Z]*\s+)*-?f\s*r?\s+/\s*$",   # rm -fr /
        r"\bsudo\b",                                      # any sudo
        r"\bshutdown\b",                                   # system shutdown
        r"\breboot\b",                                     # system reboot
        r">\s*/dev/",                                      # write to /dev/
        r"\bmkfs\b",                                       # format filesystem
        r"\bdd\s+.*of=/dev/",                              # dd to device
        r"\bcurl\b.*\|\s*\b(ba)?sh\b",                    # curl | sh
        r"\bwget\b.*\|\s*\b(ba)?sh\b",                    # wget | sh
        r":\(\)\s*\{.*\}\s*;\s*:",                         # fork bomb
        r"\bchmod\s+[0-7]*s",                              # setuid
        r"\bnc\s+-[a-z]*l",                                # netcat listener
    ]

    WARN_PATTERNS: list[str] = [
        r"\bchmod\s+777\b",                # overly permissive
        r"\bgit\s+push\s+.*--force\b",     # force push
        r"\bgit\s+push\s+.*-f\b",          # force push short
        r"\bdocker\s+rm\b",                # docker remove
        r"\bkill\s+-9\b",                  # SIGKILL
        r"\bpkill\b",                      # process kill
    ]

    def __init__(self) -> None:
        self._block = [re.compile(p, re.IGNORECASE) for p in self.BLOCK_PATTERNS]
        self._warn = [re.compile(p, re.IGNORECASE) for p in self.WARN_PATTERNS]

    def check(self, command: str) -> tuple[bool, Optional[str]]:
        """
        Check a command for safety.

        Returns:
            (allowed, reason)
            - allowed=False, reason=pattern  → hard block
            - allowed=True,  reason=pattern  → warn
            - allowed=True,  reason=None     → clean
        """
        for pattern in self._block:
            if pattern.search(command):
                return False, f"Blocked: {pattern.pattern}"
        for pattern in self._warn:
            if pattern.search(command):
                return True, f"Warning: {pattern.pattern}"
        return True, None


# ── Output Sanitizer ──────────────────────────────────────

class OutputSanitizer:
    """
    Mask sensitive data in tool output before sending to LLM.

    Detects and redacts:
    - API keys (Anthropic, OpenAI, AWS, GitHub)
    - Generic key=value secrets
    - Bearer tokens
    """

    PATTERNS: list[tuple[str, str]] = [
        # Specific patterns first (before generic key=value catch-all)
        # Anthropic key
        (r"sk-ant-[a-zA-Z0-9\-]{20,}", "sk-ant-***REDACTED***"),
        # GitHub token
        (r"ghp_[a-zA-Z0-9]{36}", "ghp_***REDACTED***"),
        (r"github_pat_[a-zA-Z0-9_]{20,}", "github_pat_***REDACTED***"),
        # OpenAI key
        (r"sk-[a-zA-Z0-9]{20,}", "sk-***REDACTED***"),
        # AWS key
        (r"AKIA[A-Z0-9]{16}", "AKIA***REDACTED***"),
        # Bearer tokens
        (r"Bearer\s+[a-zA-Z0-9\-_.]+", "Bearer ***REDACTED***"),
        # Generic key=value secrets (last — catch-all)
        (
            r"(?i)(api[_-]?key|token|secret|password|passwd|authorization)"
            r"\s*[:=]\s*['\"]?\S{8,}['\"]?",
            r"\1=***REDACTED***",
        ),
    ]

    def __init__(self) -> None:
        self._compiled = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.PATTERNS
        ]

    def sanitize(self, text: str) -> str:
        """Return text with sensitive values masked."""
        for pattern, replacement in self._compiled:
            text = pattern.sub(replacement, text)
        return text


# ── Input Validator ───────────────────────────────────────

class InputValidator:
    """Validate tool inputs before execution."""

    MAX_CONTENT_LENGTH = 500_000  # 500KB max for write_file
    MAX_COMMAND_LENGTH = 10_000   # 10KB max for bash commands

    @staticmethod
    def validate_command(command: str) -> str:
        """Validate a bash command string."""
        if not command or not command.strip():
            raise ValueError("Empty command")
        if len(command) > InputValidator.MAX_COMMAND_LENGTH:
            raise ValueError(
                f"Command too long: {len(command)} > {InputValidator.MAX_COMMAND_LENGTH}"
            )
        return command.strip()

    @staticmethod
    def validate_file_content(content: str) -> str:
        """Validate content for write_file."""
        if len(content) > InputValidator.MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Content too large: {len(content)} > {InputValidator.MAX_CONTENT_LENGTH}"
            )
        return content

    @staticmethod
    def validate_path(path_str: str) -> str:
        """Basic path string validation (before sandbox check)."""
        if not path_str or not path_str.strip():
            raise ValueError("Empty path")
        if "\x00" in path_str:
            raise ValueError("Null byte in path")
        if len(path_str) > 4096:
            raise ValueError("Path too long")
        return path_str.strip()
