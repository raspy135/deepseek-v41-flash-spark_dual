"""Cooperative decode boundary; ordinary engine callers never see these events."""

from dataclasses import dataclass
from typing import Any


@dataclass
class VerifyStep:
    block: Any
    position: int
    rows: Any


def event_signature(event):
    if isinstance(event, VerifyStep):
        return ('verify', event.position)
    if event is None:
        return ('done',)
    return ('tokens', tuple(event))
