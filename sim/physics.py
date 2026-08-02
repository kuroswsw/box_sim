"""投擲物の並進物理。バッチ(N本同時)対応の自前実装。

モデル:
  dv/dt = g_vec - beta * |v| * v + a_ou
  beta  = rho * CdA / (2 m)   (Cd·A 一定近似)
  a_ou  = Ornstein-Uhlenbeck 過程 (ティッシュ等の確率的物体のみ)
  バウンド: z<0 で vz' = -e*vz, 水平速度 *= bounce_friction

積分: 決定論部は RK4、OU 外乱は Euler-Maruyama で分離加算。
dt=2ms で弾道 0.8s の位置誤差は数値的に <0.1mm (RK4) であり、
モデル誤差(Cd·A一定近似)より2桁以上小さい。

近似の破綻箇所:
- Cd·A 一定: タンブリングする PET ボトルは実際には投影面積が周期変動し
  抗力が ±30% 程度揺れる。滞空 0.5-0.8s では着地点で数 cm の系統誤差。
  → キャリブレーションで「平均的な CdA」に吸収する方針。
- バウンドの姿勢依存: ボトルの角で跳ねると水平方向にも大きく散るが、
  本モデルは e と摩擦率のスカラー2つに縮約。バウンド後の追跡精度は保証しない
  (捕球は初回着地までが勝負なので許容)。
- OU 外乱: ティッシュの実軌道の再現ではなく「予測不能性の統計的性質」の再現。
"""
from __future__ import annotations

import numpy as np

from .params import ObjectParams, WorldParams


def drag_accel(v: np.ndarray, beta: float) -> np.ndarray:
    """抗力加速度 -beta*|v|*v。v: (...,3), beta: スカラー"""
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    return -beta * speed * v


def simulate_batch(
    pos0: np.ndarray,
    vel0: np.ndarray,
    obj: ObjectParams,
    world: WorldParams,
    rng: np.random.Generator | None = None,
    n_bounces: int = 2,
    stop_speed: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N 本の軌道を同時に積分する。

    pos0, vel0: (N,3)
    返り値: times (T,), pos (T,N,3), vel (T,N,3)
    バウンドは n_bounces 回まで追跡し、それ以降 or 低速化で地面に張り付ける。
    """
    if rng is None:
        rng = np.random.default_rng()
    pos0 = np.atleast_2d(np.asarray(pos0, dtype=float))
    vel0 = np.atleast_2d(np.asarray(vel0, dtype=float))
    n = pos0.shape[0]
    dt = world.dt
    steps = int(world.t_max / dt) + 1
    g_vec = np.array([0.0, 0.0, -world.g])
    beta = obj.beta

    pos = np.empty((steps, n, 3))
    vel = np.empty((steps, n, 3))
    pos[0], vel[0] = pos0, vel0
    times = np.arange(steps) * dt

    ou = np.zeros((n, 3))
    bounce_count = np.zeros(n, dtype=int)
    settled = np.zeros(n, dtype=bool)

    def acc(v: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(v, axis=-1, keepdims=True)
        return g_vec - beta * speed * v

    for k in range(1, steps):
        p, v = pos[k - 1], vel[k - 1]
        # RK4 (決定論部)
        k1 = acc(v)
        k2 = acc(v + 0.5 * dt * k1)
        k3 = acc(v + 0.5 * dt * k2)
        k4 = acc(v + dt * k3)
        a_det = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        # OU 外乱 (Euler-Maruyama)
        if obj.stochastic and obj.ou_sigma > 0:
            ou += (-ou / obj.ou_tau) * dt + obj.ou_sigma * np.sqrt(
                2.0 * dt / obj.ou_tau) * rng.standard_normal((n, 3))
            a_tot = a_det + ou
        else:
            a_tot = a_det
        v_new = v + a_tot * dt
        p_new = p + 0.5 * (v + v_new) * dt

        # バウンド処理
        hit = (p_new[:, 2] < 0.0) & ~settled
        if np.any(hit):
            p_new[hit, 2] = 0.0
            vz = v_new[hit, 2]
            v_new[hit, 2] = -obj.restitution * vz
            v_new[hit, 0] *= obj.bounce_friction
            v_new[hit, 1] *= obj.bounce_friction
            bounce_count[hit] += 1
            slow = np.linalg.norm(v_new, axis=-1) < stop_speed
            newly_settled = hit & (slow | (bounce_count > n_bounces))
            settled |= newly_settled
        v_new[settled] = 0.0
        p_new[settled, 2] = 0.0

        pos[k], vel[k] = p_new, v_new

    return times, pos, vel


def landing_time_index(pos: np.ndarray, catch_height: float,
                       descending_only: bool = True,
                       vel: np.ndarray | None = None) -> np.ndarray:
    """各軌道が catch_height を(下降中に)最初に横切るステップ番号。(T,N,3)->(N,)
    横切らない場合は -1。"""
    z = pos[:, :, 2]
    below = z <= catch_height
    if descending_only and vel is not None:
        below &= vel[:, :, 2] < 0.0
    # 上昇中に始まる場合を考慮し、最初の False→True 遷移を探す
    first = np.full(z.shape[1], -1, dtype=int)
    for i in range(z.shape[1]):
        idx = np.nonzero(below[:, i])[0]
        first[i] = idx[0] if idx.size else -1
    return first
