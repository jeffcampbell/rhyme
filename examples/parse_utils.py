"""Shared JSON parsing utilities for adapter examples."""

import json
import re


def extract_json_array(text: str) -> list:
    """Robustly extract a JSON array from LLM output.

    Handles: bare arrays, code fences, prose wrapping, trailing commas,
    nested in objects, reversed key ordering, thinking tags, etc.
    """
    text = text.strip()

    # Strip thinking tags (DeepSeek, Qwen, Gemini thinking mode)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text)
    text = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', text)
    text = text.strip()

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
    # Strip thinking tags
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text)
    text = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', text)
    text = text.strip().upper()
    for char in text:
        if char in "ABCDE":
            return char
    return text[0] if text else "A"
