"""Load typed argparse defaults from a versioned YAML profile."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import yaml


def _flatten_profile(value: Mapping, prefix: tuple[str, ...] = ()) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for raw_key, child in value.items():
        key = str(raw_key).replace("-", "_")
        path = (*prefix, key)
        if isinstance(child, Mapping):
            nested = _flatten_profile(child, path)
            duplicates = flattened.keys() & nested.keys()
            if duplicates:
                raise ValueError(f"Duplicate profile keys: {sorted(duplicates)}")
            flattened.update(nested)
            continue
        leaf = path[-1]
        if leaf in flattened:
            raise ValueError(f"Duplicate profile key: {leaf}")
        flattened[leaf] = child
    return flattened


def apply_profile_defaults(parser: argparse.ArgumentParser, path: Path) -> None:
    """Apply YAML leaves to matching argparse destinations.

    Section names are organizational. For example,
    ``topics: {scan_topic: /scan}`` configures ``--scan-topic``.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Profile must contain a YAML mapping: {path}")

    values = _flatten_profile(payload)
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {argparse.SUPPRESS, "help", "config"}
    }
    unknown = sorted(set(values) - set(actions))
    if unknown:
        parser.error(f"unknown keys in {path}: {', '.join(unknown)}")

    typed: dict[str, object] = {}
    for key, value in values.items():
        action = actions[key]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                parser.error(f"{key} must be true or false in {path}")
            typed[key] = value
        elif action.type is not None and value is not None:
            typed[key] = action.type(value)
        else:
            typed[key] = value
    parser.set_defaults(**typed)
