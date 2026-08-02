"""投擲モデル: プリセット + 正規分布ばらつきのサンプリング。"""
from __future__ import annotations

import numpy as np

from .params import ThrowPreset


def sample_throws(preset: ThrowPreset, n: int,
                  rng: np.random.Generator | None = None,
                  deterministic: bool = False
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n 投分の初期状態をサンプル。返り値 (pos0, vel0, omega0) 各 (n,3)。"""
    if rng is None:
        rng = np.random.default_rng()
    pm = np.asarray(preset.pos_mean)
    vm = np.asarray(preset.vel_mean)
    if deterministic:
        pos0 = np.tile(pm, (n, 1))
        vel0 = np.tile(vm, (n, 1))
        omega0 = np.zeros((n, 3))
        return pos0, vel0, omega0
    pos0 = pm + rng.standard_normal((n, 3)) * np.asarray(preset.pos_std)
    vel0 = vm + rng.standard_normal((n, 3)) * np.asarray(preset.vel_std)
    omega0 = rng.standard_normal((n, 3)) * preset.omega_std
    return pos0, vel0, omega0
