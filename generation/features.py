"""Independent rollout flags for architecture experiments."""

from __future__ import annotations

import os


DEFAULTS = {
    "two_resource_router": True,
    "support_evidence_judge": True,
    "typed_planner": True,
    "parallel_retrieval": True,
    "coverage_retry": True,
    "metadata_routing": True,
    "cross_resource_computation": True,
    "parent_child_expansion": False,
}


def feature_enabled(name: str, *, environ=None) -> bool:
    if name not in DEFAULTS:
        raise ValueError(f"unknown architecture feature {name!r}")
    values = os.environ if environ is None else environ
    key = f"SDGE_FEATURE_{name.upper()}"
    raw = values.get(key)
    if raw is None:
        if environ is None:
            # Import lazily to avoid coupling provider-only use cases to the
            # retrieval configuration module at import time.
            try:
                from retrieval.utils import load_config

                configured = load_config().get("architecture_features", {}).get(name)
            except (OSError, RuntimeError, ValueError):
                configured = None
            if isinstance(configured, bool):
                return configured
        return DEFAULTS[name]
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean value")
