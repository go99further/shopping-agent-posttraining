"""Deterministic public-state process verification for shopping trajectories.

This module is deliberately independent from veRL and from Reward v3 internals.
It only consumes the observation and tool call visible to the actor.  The frozen
Environment v2.1 Reward v3 remains the terminal objective; these signals are
training-side diagnostics and optional potential-based credit assignment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence

from shopping_grpo.environment.actions import (
    action_reject_reason,
    clickable_buttons,
    product_ids,
)
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

PROCESS_VERIFIER_SCHEMA = "shopping-process-verifier-v1"
EXACT_PUBLIC_OBSERVATION_SCHEMA = "shopping-exact-public-observation-v1"
PROCESS_CHECK_NAMES = (
    "legal_action",
    "state_changed",
    "novel_evidence",
    "candidate_opened",
    "option_selection_progress",
    "information_evidence_opened",
    "purchase_ready_before_action",
    "terminal_action",
)
PROCESS_FAILURE_NAMES = (
    "illegal_action",
    "no_progress_action",
    "repeated_no_progress",
    "premature_purchase",
)

_FIELD = re.compile(r"(?:^|\n)([a-z_]+):\s*([^\n]*)")
_PAGE_NUMBER = re.compile(r"(?:^|\n)Page\s+(\d+)\s+of\s+\d+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_STRUCTURED_FIELDS = {
    "page_type",
    "query",
    "normalized_query",
    "page",
    "asin",
    "subpage",
    "selected_options",
    "available_options",
}
_EVIDENCE_TOOLS = {
    "search_products",
    "open_product",
    "select_option",
    "view_description",
    "view_features",
    "view_reviews",
    "view_attributes",
    "next_page",
    "prev_page",
}
_REQUIRED_ARGUMENT_NAMES = {
    tool["function"]["name"]: set(
        tool["function"]["parameters"].get("required", [])
    )
    for tool in SHOP_TOOL_SCHEMAS
}


def _normalized_text(value: object) -> str:
    return _WHITESPACE.sub(" ", str(value or "")).strip()


def _json_object(raw: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def public_state_features(observation: object) -> dict[str, object]:
    """Extract a bounded canonical state from actor-visible observation text."""
    text = str(observation or "")
    fields = {
        match.group(1): _normalized_text(match.group(2))
        for match in _FIELD.finditer(text)
        if match.group(1) in _STRUCTURED_FIELDS
    }
    page_type = _normalized_text(fields.get("page_type", "generic")).casefold()
    page_match = _PAGE_NUMBER.search(text)
    page = fields.get("page") or (page_match.group(1) if page_match else "")
    if page_type in {"product_detail", "information_subpage"}:
        selected_options = _json_object(fields.get("selected_options", ""))
        available_options = _json_object(fields.get("available_options", ""))
    else:
        selected_options = {}
        available_options = {}
    return {
        "page_type": page_type,
        "query": fields.get("normalized_query") or fields.get("query") or "",
        "page": page,
        "asin": fields.get("asin", ""),
        "subpage": fields.get("subpage", ""),
        "selected_options": selected_options,
        "available_option_names": sorted(str(key) for key in available_options),
        "product_ids": product_ids(text),
        "buttons": sorted(
            _normalized_text(item).casefold() for item in clickable_buttons(text)
        ),
    }


def canonical_public_anchor(
    observation: object,
    *,
    mode: str = "structured",
    features: Mapping[str, object] | None = None,
) -> str:
    """Return a stable public-state anchor without retaining raw observations."""
    if mode == "structured":
        payload = json.dumps(
            features if features is not None else public_state_features(observation),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    elif mode == "exact":
        payload = _normalized_text(observation)
    else:
        raise ValueError("anchor mode must be 'structured' or 'exact'")
    return f"{mode}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def exact_public_observation_hash(observation: object) -> str:
    """Hash exact actor-visible UTF-8 bytes without fuzzy or semantic projection."""
    payload = str(observation or "").encode("utf-8")
    return f"exact:{hashlib.sha256(payload).hexdigest()}"


def canonical_action_key(tool_name: object, parameters: object) -> str:
    """Canonicalize a public tool decision for repeated/contrastive checks."""
    safe_parameters = parameters if isinstance(parameters, Mapping) else {}
    payload = json.dumps(
        {"tool": str(tool_name), "parameters": safe_parameters},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def public_state_checks(features: Mapping[str, object]) -> dict[str, bool]:
    """Expose named, auditable state checks used by the process potential."""
    page_type = str(features.get("page_type") or "generic")
    selected = features.get("selected_options")
    selected_names = set(selected) if isinstance(selected, Mapping) else set()
    option_names = {str(value) for value in features.get("available_option_names") or []}
    candidate_open = page_type in {"product_detail", "information_subpage"} and bool(
        features.get("asin")
    )
    options_complete = candidate_open and option_names.issubset(selected_names)
    return {
        "search_results_visible": page_type == "search_results",
        "search_query_visible": bool(features.get("query")),
        "result_candidates_visible": bool(features.get("product_ids")),
        "candidate_open": candidate_open,
        "options_complete": options_complete,
        "buy_action_visible": "buy now" in (features.get("buttons") or []),
        "information_evidence_visible": page_type == "information_subpage",
        "purchase_ready": (
            page_type == "product_detail"
            and options_complete
            and "buy now" in (features.get("buttons") or [])
        ),
    }


def public_state_potential(observation: object) -> float:
    """Compute bounded deterministic progress from public navigation state only."""
    return public_feature_potential(public_state_features(observation))


def public_feature_potential(features: Mapping[str, object]) -> float:
    """Compute the potential from named public checks and option completion."""
    checks = public_state_checks(features)
    page_type = str(features.get("page_type") or "generic")
    if page_type == "search_results":
        value = (
            0.20
            + 0.05 * checks["search_query_visible"]
            + 0.05 * checks["result_candidates_visible"]
        )
    elif page_type == "product_detail":
        option_names = list(features.get("available_option_names") or [])
        selected = features.get("selected_options")
        selected_count = len(selected) if isinstance(selected, Mapping) else 0
        option_ratio = selected_count / max(len(option_names), 1) if option_names else 0.0
        value = 0.55 + 0.15 * min(option_ratio, 1.0) + 0.05 * checks["buy_action_visible"]
    elif page_type == "information_subpage":
        value = 0.65
    else:
        value = 0.0
    return min(max(float(value), 0.0), 1.0)


def potential_delta_reward(
    before_potential: float,
    after_potential: float,
    *,
    gamma: float = 1.0,
    clip: float = 0.25,
) -> float:
    """Compute clipped potential shaping from precomputed finite potentials."""
    gamma = float(gamma)
    clip = float(clip)
    before_potential = float(before_potential)
    after_potential = float(after_potential)
    if not 0.0 < gamma <= 1.0 or not math.isfinite(gamma):
        raise ValueError("potential gamma must be finite and in (0, 1]")
    if clip < 0.0 or not math.isfinite(clip):
        raise ValueError("potential clip must be finite and non-negative")
    if not math.isfinite(before_potential) or not math.isfinite(after_potential):
        raise ValueError("public-state potentials must be finite")
    delta = gamma * after_potential - before_potential
    return min(max(delta, -clip), clip)


def potential_turn_reward(
    before: object,
    after: object,
    *,
    gamma: float = 1.0,
    clip: float = 0.25,
) -> float:
    """Return clipped potential shaping; repeated states cannot yield a gain."""
    return potential_delta_reward(
        public_state_potential(before),
        public_state_potential(after),
        gamma=gamma,
        clip=clip,
    )


def verify_public_transition(
    before_observation: object,
    after_observation: object,
    *,
    tool_name: object,
    parameters: object = None,
    prior_decisions: Sequence[tuple[str, str]] = (),
    anchor_mode: str = "structured",
    gamma: float = 1.0,
    clip: float = 0.25,
    terminal: bool = False,
    infrastructure_invalid: bool = False,
) -> dict[str, object]:
    """Verify one transition without task goals, Reward v3 fields or an LLM judge."""
    parameters = parameters if isinstance(parameters, Mapping) else {}
    tool_name = str(tool_name)
    before_text = str(before_observation or "")
    after_text = str(after_observation or before_text)
    before_features = public_state_features(before_text)
    after_features = public_state_features(after_text)
    before_anchor = canonical_public_anchor(
        before_text,
        mode=anchor_mode,
        features=before_features,
    )
    after_anchor = canonical_public_anchor(
        after_text,
        mode=anchor_mode,
        features=after_features,
    )
    action_key = canonical_action_key(tool_name, parameters)
    guard_reason = action_reject_reason(tool_name, dict(parameters), before_text)
    missing_arguments = sorted(
        _REQUIRED_ARGUMENT_NAMES.get(tool_name, set()) - set(parameters)
    )
    if guard_reason is None and missing_arguments:
        guard_reason = "schema_missing_arguments:" + ",".join(missing_arguments)
    legal_action = guard_reason is None
    before_checks = public_state_checks(before_features)
    after_checks = public_state_checks(after_features)
    before_potential = public_feature_potential(before_features)
    after_potential = public_feature_potential(after_features)
    state_changed = before_anchor != after_anchor

    before_selected = before_features.get("selected_options")
    after_selected = after_features.get("selected_options")
    before_selected_count = len(before_selected) if isinstance(before_selected, Mapping) else 0
    after_selected_count = len(after_selected) if isinstance(after_selected, Mapping) else 0
    candidate_opened = (
        tool_name == "open_product"
        and before_features.get("page_type") == "search_results"
        and after_features.get("page_type") == "product_detail"
        and bool(after_features.get("asin"))
    )
    option_progress = (
        tool_name == "select_option" and after_selected_count > before_selected_count
    )
    information_opened = (
        tool_name.startswith("view_")
        and after_features.get("page_type") == "information_subpage"
        and state_changed
    )
    search_evidence_changed = (
        after_features.get("page_type") == "search_results"
        and any(
            before_features.get(name) != after_features.get(name)
            for name in ("query", "page", "product_ids")
        )
    )
    novel_evidence = legal_action and (
        candidate_opened
        or option_progress
        or information_opened
        or search_evidence_changed
        or (tool_name in _EVIDENCE_TOOLS and after_potential > before_potential)
    )
    repeated_decision = (before_anchor, action_key) in set(prior_decisions)
    no_progress_action = (
        legal_action
        and not terminal
        and tool_name not in {"finish_without_purchase"}
        and not state_changed
        and not novel_evidence
    )
    repeated_no_progress = no_progress_action and repeated_decision
    premature_purchase = (
        legal_action and tool_name == "buy_now" and not before_checks["purchase_ready"]
    )
    checks = {
        "legal_action": legal_action,
        "state_changed": state_changed,
        "novel_evidence": novel_evidence,
        "candidate_opened": candidate_opened,
        "option_selection_progress": option_progress,
        "information_evidence_opened": information_opened,
        "purchase_ready_before_action": before_checks["purchase_ready"],
        "terminal_action": tool_name in {"buy_now", "finish_without_purchase"},
    }
    failures = {
        "illegal_action": not legal_action,
        "no_progress_action": no_progress_action,
        "repeated_no_progress": repeated_no_progress,
        "premature_purchase": premature_purchase,
    }
    process_reward = 0.0
    if not terminal and not infrastructure_invalid:
        process_reward = potential_delta_reward(
            before_potential,
            after_potential,
            gamma=gamma,
            clip=clip,
        )
    return {
        "schema_version": PROCESS_VERIFIER_SCHEMA,
        "checks": checks,
        "failures": failures,
        "guard_reason": guard_reason,
        "before_state_checks": before_checks,
        "after_state_checks": after_checks,
        "before_potential": before_potential,
        "after_potential": after_potential,
        "process_reward": process_reward,
    }


def first_process_failure_step(turns: Sequence[Mapping[str, object]]) -> int | None:
    """Return the zero-based first verifier failure step, if any."""
    for index, turn in enumerate(turns):
        verifier = turn.get("verifier")
        failures = verifier.get("failures") if isinstance(verifier, Mapping) else None
        if isinstance(failures, Mapping) and any(
            bool(failures.get(name)) for name in PROCESS_FAILURE_NAMES
        ):
            return int(turn.get("step_index", index))
    return None
