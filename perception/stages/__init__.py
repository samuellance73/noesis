"""
perception/stages/__init__.py
──────────────────────────────
Exports all pipeline stage classes for convenient importing.
"""

from .intake import IntakeBuffer
from .dedup import Deduplicator
from .authority import AuthorityScorer
from .router import Router

__all__ = [
    "IntakeBuffer",
    "Deduplicator",
    "AuthorityScorer",
    "Router",
]

