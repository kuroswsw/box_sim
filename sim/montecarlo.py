"""設計探索用モンテカルロ: 4設計パラメータの2軸ヒートマップ。

パラメータ: mouth (受け口展開径), deploy (展開時間), amax (最大加速度),
latency (センサ遅延)。任意の2つを軸に、残り2つはスライダ値で固定。
各格子点で n_trials 試行を run_batch (ベクトル化済) で回す。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .loop import run_batch
from .params import (ControlParams, ObjectParams, RobotParams, SensorParams,
                     ThrowPreset, WorldParams)

# step は GUI スライダの刻み幅。Dash では step=None にすると
# 「marks の位置にしかスナップしない」ため、必ず明示すること。
PARAM_SPECS = {
    "mouth": dict(label="受け口展開径 [m]", lo=0.10, hi=0.40, step=0.01),
    "deploy": dict(label="展開時間 [s]", lo=0.05, hi=0.40, step=0.01),
    "amax": dict(label="最大加速度 [m/s²]", lo=0.5, hi=8.0, step=0.1),
    "latency": dict(label="センサ遅延 [s]", lo=0.01, hi=0.25, step=0.005),
}


def apply_param(rp: RobotParams, sp: SensorParams, name: str, value: float
                ) -> tuple[RobotParams, SensorParams]:
    if name == "mouth":
        rp = replace(rp, mouth_deploy_radius=value)
    elif name == "deploy":
        rp = replace(rp, deploy_time=value)
    elif name == "amax":
        rp = replace(rp, a_max=value)
    elif name == "latency":
        sp = replace(sp, latency=value)
    else:
        raise ValueError(name)
    return rp, sp


def mc_heatmap(obj: ObjectParams, preset: ThrowPreset,
               rp_base: RobotParams, sp_base: SensorParams,
               cp: ControlParams, world: WorldParams,
               x_name: str, x_values: np.ndarray,
               y_name: str, y_values: np.ndarray,
               n_trials: int = 1000, seed: int = 0,
               progress=None) -> np.ndarray:
    """捕球率グリッド (len(y), len(x)) を返す。

    共通乱数 (同一シード) を全格子点で使う → 格子間の差が
    モンテカルロ雑音でなく設計差として比較できる (CRN法)。
    """
    grid = np.zeros((len(y_values), len(x_values)))
    total = len(y_values) * len(x_values)
    k = 0
    for iy, yv in enumerate(y_values):
        for ix, xv in enumerate(x_values):
            rp, sp = apply_param(rp_base, sp_base, x_name, float(xv))
            rp, sp = apply_param(rp, sp, y_name, float(yv))
            # 転倒制約: a_max は準静的転倒限界でクリップ
            a_tip = rp.tipping_a_max(world.g)
            if rp.a_max > a_tip:
                rp = replace(rp, a_max=a_tip)
            res = run_batch(obj, preset, rp, sp, cp, world,
                            n=n_trials, rng=np.random.default_rng(seed))
            grid[iy, ix] = res.catch_rate
            k += 1
            if progress:
                progress(k, total)
    return grid
