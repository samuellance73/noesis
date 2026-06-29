"""
perception/stages/__init__.py
──────────────────────────────
Exports all pipeline stage classes for convenient importing.
"""

from .intake import IntakeBuffer
from .authority import AuthorityScorer

__all__ = [
    "IntakeBuffer",
    "AuthorityScorer",
]

 