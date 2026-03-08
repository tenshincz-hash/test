from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class InMemoryDataStore:
    _tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def write(self, name: str, df: pd.DataFrame) -> None:
        self._tables[name] = df.copy()

    def read(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            raise KeyError(f"Table '{name}' not found in store")
        return self._tables[name].copy()

    def list_tables(self) -> list[str]:
        return sorted(self._tables)
