"""
training_session.py - Runtime state for JSONL-driven roleplay sessions.

Each live session gets one fixed roleplay case selected from the JSONL
candidate pool. The model's next_state is stored here and fed into the next
turn prompt.
"""

from __future__ import annotations

import random
import threading
from typing import Any

try:
    from .case_loader import find_cases, get_case
except ImportError:
    from case_loader import find_cases, get_case

_LOCK = threading.Lock()
_SESSIONS: dict[str, dict[str, Any]] = {}
_ACTIVE_CASE_BY_COMBO: dict[tuple[str, str, str | None], str] = {}
_LAST_RANDOM_CASE_BY_COMBO: dict[tuple[str, str, str | None], str] = {}


def _initial_state(case: dict[str, Any]) -> str:
    state_machine = case.get("state_machine", {})
    initial = state_machine.get("initial_state")
    if initial:
        return str(initial)
    states = state_machine.get("states", {})
    if states:
        return str(next(iter(states)))
    return "guarded"


def _valid_states(case: dict[str, Any]) -> set[str]:
    states = case.get("state_machine", {}).get("states", {})
    return {str(key) for key in states} or {_initial_state(case)}


def _state_config(case: dict[str, Any], state: str | None) -> dict[str, Any]:
    states = case.get("state_machine", {}).get("states", {})
    config = states.get(str(state or ""), {})
    return config if isinstance(config, dict) else {}


def _allowed_next_states(case: dict[str, Any], current_state: str) -> set[str]:
    transitions = case.get("state_machine", {}).get("transitions", [])
    allowed = {current_state}
    for transition in transitions:
        if str(transition.get("from") or "") == current_state:
            target = str(transition.get("to") or "").strip()
            if target:
                allowed.add(target)
    if len(allowed) == 1 and not transitions:
        allowed.update(_valid_states(case))
    return allowed


def _with_terminal_flags(context: dict[str, Any], case: dict[str, Any]) -> None:
    current_state = str(context.get("current_state") or "")
    config = _state_config(case, current_state)
    is_terminal = bool(config.get("is_terminal"))
    context["training_complete"] = is_terminal
    context["is_success"] = bool(config.get("is_success")) if is_terminal else False
    context["is_failure"] = bool(config.get("is_failure")) if is_terminal else False
    if is_terminal:
        context["final_state"] = current_state


def _is_matching_case(case: dict[str, Any], training_type: str, difficulty: str) -> bool:
    candidates = find_cases(training_type, difficulty)
    candidate_ids = {item.get("case_id") for item in candidates}
    return case.get("case_id") in candidate_ids


def _combo_key(training_type: str, difficulty: str, business_line: str | None = None) -> tuple[str, str, str | None]:
    return (str(training_type or ""), str(difficulty or ""), business_line)


def set_active_case_for_combo(
    training_type: str,
    difficulty: str,
    case_id: str | None,
    business_line: str | None = None,
) -> None:
    """Remember the dashboard-selected case so voice sessions use the same customer."""
    if not case_id:
        return
    with _LOCK:
        _ACTIVE_CASE_BY_COMBO[_combo_key(training_type, difficulty, business_line)] = str(case_id)


def get_active_case_for_combo(
    training_type: str,
    difficulty: str,
    business_line: str | None = None,
) -> str | None:
    with _LOCK:
        return _ACTIVE_CASE_BY_COMBO.get(_combo_key(training_type, difficulty, business_line))


def _choose_random_case(candidates: list[dict[str, Any]], key: tuple[str, str, str | None]) -> dict[str, Any] | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        case = candidates[0]
    else:
        last_case_id = _LAST_RANDOM_CASE_BY_COMBO.get(key)
        pool = [case for case in candidates if case.get("case_id") != last_case_id]
        case = random.choice(pool or candidates)
    if case and case.get("case_id"):
        _LAST_RANDOM_CASE_BY_COMBO[key] = str(case.get("case_id"))
    return case


def get_or_create_session_context(
    session_id: str,
    training_type: str,
    difficulty: str,
    business_line: str | None = None,
    preferred_case_id: str | None = None,
    use_active_case: bool = False,
) -> dict[str, Any]:
    """Return a stable case/state context for this session.

    If the session changes stage or difficulty, a new case is selected.
    """
    candidates = find_cases(training_type, difficulty, business_line)
    key = _combo_key(training_type, difficulty, business_line)

    with _LOCK:
        active_case_id = preferred_case_id or (_ACTIVE_CASE_BY_COMBO.get(key) if use_active_case else None)
        existing = _SESSIONS.get(session_id)
        if (
            existing
            and existing.get("training_type") == training_type
            and existing.get("difficulty") == difficulty
            and existing.get("business_line") == business_line
            and (not active_case_id or existing.get("case_id") == active_case_id)
        ):
            return dict(existing)

        case = None
        if active_case_id:
            preferred = get_case(active_case_id)
            if preferred and _is_matching_case(preferred, training_type, difficulty):
                case = preferred
        if case is None and candidates:
            case = _choose_random_case(candidates, key)

        context = {
            "session_id": session_id,
            "training_type": training_type,
            "difficulty": difficulty,
            "business_line": business_line,
            "candidate_count": len(candidates),
            "case_id": case.get("case_id") if case else None,
            "current_state": _initial_state(case) if case else "",
            "turn_count": 0,
        }
        if case:
            _with_terminal_flags(context, case)
            if case.get("case_id"):
                _ACTIVE_CASE_BY_COMBO[key] = str(case.get("case_id"))
        _SESSIONS[session_id] = context
        return dict(context)


def get_session_context(session_id: str) -> dict[str, Any] | None:
    with _LOCK:
        context = _SESSIONS.get(session_id)
        return dict(context) if context else None


def get_session_case(session_id: str) -> dict[str, Any] | None:
    context = get_session_context(session_id)
    if not context:
        return None
    return get_case(context.get("case_id"))


def advance_session_state(
    session_id: str,
    next_state: str | None,
    triggered_events: list[str] | None = None,
) -> dict[str, Any] | None:
    """Update current_state when next_state is valid for the fixed case.

    The model can keep the current state or move along an explicit transition.
    Invalid states or illegal jumps are recorded but do not mutate the session.
    """
    with _LOCK:
        context = _SESSIONS.get(session_id)
        if not context:
            return None
        case = get_case(context.get("case_id"))
        if not case:
            return dict(context)

        current_state = str(context.get("current_state") or _initial_state(case))
        normalized = str(next_state or "").strip()
        valid_states = _valid_states(case)
        allowed = _allowed_next_states(case, current_state)
        current_config = _state_config(case, current_state)

        validation = {
            "requested_state": normalized,
            "from_state": current_state,
            "allowed_states": sorted(allowed),
            "is_valid": True,
            "reason": "unchanged",
        }

        applied_state = current_state
        if current_config.get("is_terminal"):
            validation.update({"is_valid": normalized in ("", current_state), "reason": "terminal_state_locked"})
        elif not normalized:
            validation["reason"] = "empty_next_state"
        elif normalized not in valid_states:
            validation.update({"is_valid": False, "reason": "unknown_state"})
        elif normalized not in allowed:
            validation.update({"is_valid": False, "reason": "illegal_transition"})
        else:
            applied_state = normalized
            validation["reason"] = "valid_transition" if normalized != current_state else "unchanged"

        context["previous_state"] = current_state
        context["current_state"] = applied_state
        context["turn_count"] = int(context.get("turn_count") or 0) + 1
        context["last_triggered_events"] = triggered_events or []
        validation["applied_state"] = applied_state
        context["state_validation"] = validation
        _with_terminal_flags(context, case)
        return dict(context)


def reset_session_context(session_id: str) -> None:
    with _LOCK:
        _SESSIONS.pop(session_id, None)
