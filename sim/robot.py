"""3輪オムニ機体 + 展開式受け口。バッチ(N台同時)対応。

機体モデル:
  a_cmd → 車輪力配分(3輪, 120°配置) → 車輪ごとの電流(=力)飽和
       → 1次遅れ(FOC電流制御+機械系, tau_mech) → 摩擦限界チェック
       → スリップ時は動摩擦力に低下 → 積分, v_max クランプ

近似の破綻箇所:
- ヨー回転は無視(オムニで並進と独立に制御できる前提)。実機ではヨー外乱
  トルクで配分が数%食われる。
- 1次遅れ1本に電気系+機械系を縮約。FOC帯域(kHz)≫機械系(数十Hz)なので
  実用上は機械系時定数のみが見える、という近似。
- スリップは等方クーロン摩擦。オムニホイールの樽ローラは進行方向で
  摩擦係数が変わる(実測では±20%程度)が無視。μをスライダで振って
  感度を見る使い方を想定。
- 転倒は動力学として解かず「重心高さ≤トレッド半分」の設計制約と
  準静的限界 a_tip = g·R/h の表示のみ。急峻な加速度反転時の動的転倒は
  本シミュでは検出できない。

受け口:
  バネ蓄勢+ソレノイドシアー解放 → deploy_time で全展開。
  バネ機構なので展開初期が速い smoothstep で半径が増える近似。
  再装填(サーボウインチ, reload_time)は1投擲内では完了しない前提で、
  「展開は1試行1回のみ」とする。
"""
from __future__ import annotations

import numpy as np

from .params import RobotParams, WorldParams

# 3輪オムニ配置: 車輪位置角 90°, 210°, 330°。駆動方向は接線方向。
_WHEEL_ANGLES = np.deg2rad([90.0, 210.0, 330.0])
# 駆動単位ベクトル u_i = (-sin, cos)
_A = np.stack([-np.sin(_WHEEL_ANGLES), np.cos(_WHEEL_ANGLES)])  # (2,3)
_A_PINV = np.linalg.pinv(_A)  # (3,2)


class RobotBatch:
    """N台分の機体状態と受け口状態。"""

    def __init__(self, n: int, rp: RobotParams, world: WorldParams,
                 pos0: np.ndarray | None = None):
        self.rp = rp
        self.world = world
        self.n = n
        self.pos = np.zeros((n, 2)) if pos0 is None else np.array(pos0, dtype=float)
        self.vel = np.zeros((n, 2))
        self.f_act = np.zeros((n, 2))     # 実効駆動力 (1次遅れ状態)
        self.a_cmd = np.zeros((n, 2))     # ZOH された加速度指令
        self.slipping = np.zeros(n, dtype=bool)
        # 受け口: nan = 未展開
        self.deploy_t0 = np.full(n, np.nan)
        # 車輪あたり最大力: 指令方向最悪ケースで a_max を出せる値
        self.f_wheel_max = rp.mass * rp.a_max / 1.5

    def trigger_deploy(self, mask: np.ndarray, t: float) -> None:
        """展開トリガ(1試行1回のみ)。既に展開済みなら無視。"""
        fresh = mask & np.isnan(self.deploy_t0)
        self.deploy_t0[fresh] = t

    def mouth_radius(self, t: float) -> np.ndarray:
        """時刻 t での受け口有効半径 (N,)。バネ展開を smoothstep 近似。"""
        rp = self.rp
        r = np.full(self.n, rp.mouth_base_radius)
        started = ~np.isnan(self.deploy_t0)
        if np.any(started):
            prog = np.clip((t - self.deploy_t0[started]) / rp.deploy_time, 0.0, 1.0)
            s = prog * prog * (3.0 - 2.0 * prog)  # smoothstep
            r[started] = rp.mouth_base_radius + s * (
                rp.mouth_deploy_radius - rp.mouth_base_radius)
        return r

    def step(self, dt: float) -> None:
        """物理1ステップ。a_cmd は直前の制御ティックで設定済み(ZOH)。"""
        rp = self.rp
        m = rp.mass
        # 指令力 → 車輪配分 → 車輪飽和 → 合成
        f_des = m * self.a_cmd                     # (N,2)
        f_wheels = f_des @ _A_PINV.T               # (N,3)
        f_wheels = np.clip(f_wheels, -self.f_wheel_max, self.f_wheel_max)
        f_sat = f_wheels @ _A.T                    # (N,2)
        # 摩擦限界 (等方クーロン)
        f_norm = np.linalg.norm(f_sat, axis=1)
        f_limit = rp.mu_static * m * self.world.g
        slip = f_norm > f_limit
        self.slipping = slip
        if np.any(slip):
            f_kin = rp.mu_kinetic * m * self.world.g
            scale = np.where(slip, f_kin / np.maximum(f_norm, 1e-9), 1.0)
            f_sat = f_sat * scale[:, None]
        # 1次遅れ
        alpha = dt / max(rp.tau_mech, 1e-6)
        self.f_act += np.clip(alpha, 0.0, 1.0) * (f_sat - self.f_act)
        # 積分 + v_max クランプ
        self.vel += (self.f_act / m) * dt
        speed = np.linalg.norm(self.vel, axis=1)
        over = speed > rp.v_max
        if np.any(over):
            self.vel[over] *= (rp.v_max / speed[over])[:, None]
        self.pos += self.vel * dt
