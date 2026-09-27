"""Загрузка конфигурации и разрешение путей относительно корня проекта."""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import yaml

# Корень проекта = на два уровня выше этого файла (src/selrisk/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config(dict):
    """Словарь конфигурации с доступом к путям, привязанным к корню проекта."""

    def path(self, key: str) -> Path:
        """Абсолютный путь для одной из записей блока `paths`."""
        rel = self["paths"][key]
        p = PROJECT_ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> Config:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def seed_everything(seed: int | None = None) -> int:
    """Фиксирует seed numpy для воспроизводимости; возвращает использованный seed."""
    import numpy as np

    if seed is None:
        seed = load_config()["project"]["random_seed"]
    np.random.seed(seed)
    return seed
