"""Load and validate the project YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found at {path}. Copy config/config.example.yaml "
                "to config/config.yaml and fill in your own values."
            )
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup, e.g. config.get('gee', 'project')."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node
