"""Shared JSON parsing utilities for adapter examples."""

import json
import re

_THINK_TAGS = ("think", "thinking", "reasoning")


def strip_thinking(text: str) -> str:
    """Remove chain-of-thought blocks from LLM output.

    Handles matched pairs plus the two unpaired cases servers actually emit:
    some (LM Studio) consume the opening tag and stream only the closer, and a
    generation truncated by max_tokens can open a block that never closes.
    An unpaired closer is the dangerous one -- the reasoning stays in the text
    and downstream scanners pick an answer out of the model's deliberation.
    """
    for tag in _THINK_TAGS:
        text = re.sub(rf'<{tag}>[\s\S]*?</{tag}>', '', text)

    # Unpaired closer: everything up to the last one is reasoning.
    for tag in _THINK_TAGS:
        close = f'</{tag}>'
        idx = text.rfind(close)
        if idx != -1:
            text = text[idx + len(close):]

    # Unpaired opener: reasoning ran to the end without closing.
    for tag in _THINK_TAGS:
        idx = text.find(f'<{tag}>')
        if idx != -1:
            text = text[:idx]

    return text.strip()


def extract_json_array(text: str) -> list:
    """Robustly extract a JSON array from LLM output.

    Handles: bare arrays, code fences, prose wrapping, trailing commas,
    nested in objects, reversed key ordering, thinking tags, etc.
    """
    text = text.strip()

    # Strip thinking tags (DeepSeek, Qwen, Gemini thinking mode). Keep the
    # original if stripping consumed everything -- an array buried in an
    # unterminated reasoning block is still better than nothing.
    stripped = strip_thinking(text)
    text = stripped or text

    # Strip code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            candidate = part.split("\n", 1)[-1] if "\n" in part else part
            candidate = candidate.strip()
            if candidate.startswith("[") or candidate.startswith("{"):
                text = candidate
                break

    # Try direct parse
    for attempt in [text, text.rstrip(",")]:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ["ranked_matches", "matches", "results", "incidents"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return [parsed]
        except json.JSONDecodeError:
            continue

    # Try to find a JSON array in the text
    match = re.search(r'\[[\s\S]*?\](?=\s*$|\s*\n)', text)
    if not match:
        match = re.search(r'\[[\s\S]+\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    # Last resort: regex extraction of match objects
    matches = []
    for m in re.finditer(r'"incident_id"\s*:\s*"([^"]+)"[^}]*?"confidence"\s*:\s*([\d.]+)', text):
        matches.append({"incident_id": m.group(1), "confidence": float(m.group(2))})
    if not matches:
        for m in re.finditer(r'"confidence"\s*:\s*([\d.]+)[^}]*?"incident_id"\s*:\s*"([^"]+)"', text):
            matches.append({"incident_id": m.group(2), "confidence": float(m.group(1))})
    return matches


def normalize_matches(raw_matches: list, k: int) -> list[dict]:
    """Normalize parsed matches into standard format."""
    normalized = []
    for m in raw_matches:
        if isinstance(m, dict) and "incident_id" in m:
            conf = m.get("confidence", m.get("score", 0.5))
            normalized.append({"incident_id": m["incident_id"], "confidence": float(conf)})
    return normalized[:k]


def extract_letter(text: str) -> str:
    """Extract a single letter (A-E) from remediation response."""
    text = strip_thinking(text).upper()
    for char in text:
        if char in "ABCDE":
            return char
    return text[0] if text else "A"
