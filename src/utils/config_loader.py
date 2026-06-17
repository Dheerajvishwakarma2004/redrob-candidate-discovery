"""
Configuration loader. Singleton. All modules import config from here.
"""

import yaml
from pathlib import Path
from typing import Any


class Config:
    def __init__(self, path: str = "config.yaml"):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Config not found: {self._path.absolute()}")
        with open(self._path) as f:
            self._data = yaml.safe_load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node is None:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


config = Config()