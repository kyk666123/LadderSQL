"""Minimal stand-in for `agentlightning`, so that test-time inference needs **no training framework**.

Why this exists
---------------
`original_agent/ReAct_Agent.py` holds two classes:

* `Agent`      — the multi-turn ReAct loop actually used at inference. Pure LangGraph +
                 LangChain + an OpenAI-compatible client. Needs nothing from agentlightning.
* `LitAgent`   — the *training* wrapper (`class LitAgent(agl.LitAgent[...])`), used only by the
                 RL trainer. Never instantiated during inference.

Because `LitAgent` subclasses `agl.LitAgent`, merely *importing* the module used to require the
whole `agentlightning` package (which in turn pulls in opentelemetry, and the trainer stack pulls
verl / ray / torch). That is a lot of environment risk for code paths that never run here.

This module installs a tiny stub under the name `agentlightning` **before** `original_agent` is
imported. Only the two things touched at import time are provided:

* `setup_logging()`            — called at module level in ReAct_Agent.py
* `LitAgent`                   — used as a base class (and subscripted: `agl.LitAgent[Dict]`)

Type annotations such as `resources: agl.NamedResources` are *not* evaluated, because
`ReAct_Agent.py` starts with `from __future__ import annotations` (PEP 563), so they stay strings.
`NamedResources` / `Rollout` / `LLM` / `emit_annotation` are still provided for safety.

If the real `agentlightning` is importable, it is used and this stub does nothing.
"""
from __future__ import annotations

import sys
import types


def install() -> str:
    """Make `import agentlightning` work. Returns "real" or "stub"."""
    try:
        import agentlightning  # noqa: F401
        return "real"
    except Exception:            # noqa: BLE001  (missing package or missing opentelemetry etc.)
        pass

    mod = types.ModuleType("agentlightning")

    def setup_logging(*args, **kwargs):      # no-op; the entry points configure logging themselves
        return None

    class _Subscriptable:
        """Base class that tolerates `Base[SomeType]` (PEP 560) without typing machinery."""

        def __class_getitem__(cls, item):
            return cls

    class LitAgent(_Subscriptable):
        """Placeholder for the training-side wrapper. Never used at inference time."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "LitAgent is the training-time wrapper and is not available in the "
                "inference-only package (agentlightning is not installed)."
            )

    class LLM:                                # noqa: D401 - annotation-only placeholder
        pass

    class Rollout:
        pass

    NamedResources = dict

    def emit_annotation(*args, **kwargs):
        return None

    mod.setup_logging = setup_logging
    mod.LitAgent = LitAgent
    mod.LLM = LLM
    mod.Rollout = Rollout
    mod.NamedResources = NamedResources
    mod.emit_annotation = emit_annotation
    mod.__version__ = "stub"
    sys.modules["agentlightning"] = mod
    return "stub"
