"""Shared, env-overridable locations for the Inkling MLX script suite.

The suite serves both family members, so nothing is pinned to one build root.
Point it at a different sweep with environment variables:

    export INKLING_OUT=/Users/david/llm/inkling-small-out
    export INKLING_SRC=/Users/david/llm/Inkling-Small-src
    export INKLING_PREFIX=Inkling-Small
    python3 scripts/profile_experts_mm.py

Defaults reproduce the original 975B Inkling paths, so existing invocations are
unchanged.
"""

import os

OUT = os.path.expanduser(os.environ.get("INKLING_OUT", "/Users/david/llm/inkling-mlx-out"))
SRC = os.path.expanduser(os.environ.get("INKLING_SRC", "/Users/david/llm/Inkling-src"))
PREFIX = os.environ.get("INKLING_PREFIX", "Inkling")


def build(name: str) -> str:
    """Path to a converted build: build("4bit") -> $OUT/$PREFIX-4bit."""
    return os.path.join(OUT, f"{PREFIX}-{name}")


def asset(*parts: str) -> str:
    """Path to a calibration/eval artifact under the build root."""
    return os.path.join(OUT, *parts)
