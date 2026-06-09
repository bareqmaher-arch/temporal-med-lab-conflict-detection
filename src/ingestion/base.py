"""Abstract loader interface.

A loader turns *some* data source into the canonical tables. Downstream code never
knows which loader produced the data, which is what makes MIMIC-IV a drop-in later.
"""
from __future__ import annotations

import abc

import pandas as pd

from src.config import KNOWLEDGE_CSV
from src.db.canonical import CanonicalTables


class AbstractLoader(abc.ABC):
    """Base class: implement ``load()`` to return validated canonical tables."""

    @abc.abstractmethod
    def load(self) -> CanonicalTables:  # pragma: no cover - interface
        ...

    @staticmethod
    def load_knowledge() -> pd.DataFrame:
        """The drug-lab risk knowledge base is source-independent."""
        return pd.read_csv(KNOWLEDGE_CSV)
