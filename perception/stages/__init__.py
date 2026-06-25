"""
perception/stages/__init__.py
──────────────────────────────
Exports all six pipeline stage classes for convenient importing.
"""

from .intake import IntakeBuffer
from .dedup import Deduplicator
from .classifier import Classifier
from .authority import AuthorityScorer
from .synthesizer import Synthesizer
from .router import Router

__all__ = [
    "IntakeBuffer",
    "Deduplicator",
    "Classifier",
    "AuthorityScorer",
    "Synthesizer",
    "Router",
]
