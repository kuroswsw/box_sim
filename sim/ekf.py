"""拡張カルマンフィルタ(バッチ対応)と着地点予測。

状態 x = [px py pz vx vy vz beta] (7次元)
  beta = rho*CdA/(2m)。物体分類器(YOLO/MobileNet想定)が与える事前分布
  (beta_prior_mean/std) で初期化し、飛行中に同定する。
運動モデル: dv/dt = g - beta*|v|*v (+ 白色加速度雑音)
観測: 位置のみ z = p + n, n~N(0, R)

ティッシュ等の確率的物体は OU 外乱を白色加速度換算 (PSD ≈ 2σ²τ) して
プロセス雑音 q_v に足す。→ 共分散が収束しないこと自体が
「予測不能」の判定材料になる (ビューB)。

着地点予測: 平均を短刻みで catch_height 交差まで前方積分し、
共分散もヤコビアンで同時伝播。交差時刻の不確かさ var(t)≈P_zz/vz² を
水平共分散へ v_xy·v_xyᵀ·var(t) として膨らませる。
(交差時刻とxyの相互相関 P_xz は無視 — 数%の過小評価。README参照)
"""
from __future__ import annotations

import numpy as np

from .params import ObjectParams, SensorParams, WorldParams

_I3 = np.eye(3)


def _f_jac(x: np.ndarray, g_vec: np.ndarray, dt: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray]:
    """1ステップ遷移と遷移ヤコビアン。x:(M,7), dt:(M,) → (x_next, F)"""
    m = x.shape[0]
    dt1 = dt[:, None]            # (M,1)
    dt2 = dt[:, None, None]      # (M,1,1)
    v = x[:, 3:6]
    beta = x[:, 6:7]
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    speed_s = np.maximum(speed, 1e-9)
    a = g_vec - beta * speed * v
    xn = x.copy()
    xn[:, 0:3] = x[:, 0:3] + v * dt1 + 0.5 * a * dt1 * dt1
    xn[:, 3:6] = v + a * dt1
    F = np.tile(np.eye(7), (m, 1, 1))
    F[:, 0:3, 3:6] = _I3 * dt2
    vvT = v[:, :, None] * v[:, None, :] / speed_s[:, :, None]
    dadv = -beta[:, :, None] * (speed_s[:, :, None] * _I3 + vvT)
    F[:, 3:6, 3:6] = _I3 + dt2 * dadv
    F[:, 3:6, 6] = -dt1 * (speed * v)
    return xn, F


class EkfBatch:
    """N 本同時 EKF。各フィルタの時刻 self.t は独立に管理する。"""

    def __init__(self, n: int, obj: ObjectParams, sp: SensorParams,
                 world: WorldParams):
        self.n = n
        self.world = world
        self.g_vec = np.array([0.0, 0.0, -world.g])
        # プロセス雑音: モデル不一致(Cd·A変動)ベース + OU換算
        q_base = 0.3          # [m^2/s^3] タンブリングによる抗力変動見込み
        q_ou = 2.0 * obj.ou_sigma ** 2 * obj.ou_tau if obj.stochastic else 0.0
        self.q_v = q_base + q_ou
        self.q_beta = (0.02 * max(obj.beta_prior_mean, 1e-4)) ** 2
        self.r_meas = sp.pos_noise_std ** 2
        self.beta_prior = (obj.beta_prior_mean, obj.beta_prior_std)
        self.x = np.zeros((n, 7))
        self.P = np.zeros((n, 7, 7))
        self.initialized = np.zeros(n, dtype=bool)
        self.t = np.zeros(n)
        self._prev_z: np.ndarray | None = None
        self._prev_tz: float = 0.0

    # ---- 内部ユーティリティ ----
    def _q_rate(self) -> np.ndarray:
        """単位時間あたりのプロセス雑音 (7,7)。Q = q_rate * dt"""
        Q = np.zeros((7, 7))
        Q[0:3, 0:3] = 1e-8 * _I3
        Q[3:6, 3:6] = self.q_v * _I3
        Q[6, 6] = self.q_beta
        return Q

    def _init_from_two(self, z0: np.ndarray, z1: np.ndarray, dt: float,
                       mask: np.ndarray, t_now: float) -> None:
        v = (z1 - z0) / dt
        self.x[mask, 0:3] = z1[mask]
        self.x[mask, 3:6] = v[mask]
        self.x[mask, 6] = self.beta_prior[0]
        P0 = np.zeros((7, 7))
        P0[0:3, 0:3] = self.r_meas * _I3
        P0[3:6, 3:6] = (2.0 * self.r_meas / dt ** 2 + 0.25) * _I3
        P0[6, 6] = self.beta_prior[1] ** 2
        self.P[mask] = P0
        self.t[mask] = t_now
        self.initialized |= mask

    def predict_to(self, t_target: float, mask: np.ndarray | None = None) -> None:
        """mask のフィルタをそれぞれの現在時刻から t_target まで前方予測。"""
        if mask is None:
            mask = self.initialized
        mask = mask & self.initialized & (self.t < t_target - 1e-9)
        if not np.any(mask):
            return
        idx = np.nonzero(mask)[0]
        dt_total = t_target - self.t[idx]          # (M,) 行ごとに異なってよい
        n_sub = max(1, int(np.ceil(dt_total.max() / 0.01)))
        dt_row = dt_total / n_sub
        x = self.x[idx]
        P = self.P[idx]
        q_rate = self._q_rate()
        Q = q_rate[None] * dt_row[:, None, None]
        for _ in range(n_sub):
            x, F = _f_jac(x, self.g_vec, dt_row)
            P = F @ P @ F.transpose(0, 2, 1) + Q
        x[:, 6] = np.maximum(x[:, 6], 1e-5)
        self.x[idx] = x
        self.P[idx] = P
        self.t[idx] = t_target

    def update(self, t_frame: float, z: np.ndarray) -> None:
        """観測時刻 t_frame の観測 z(N,3) で更新。NaN 行は予測のみ進める。"""
        valid = ~np.isnan(z[:, 0])
        if self._prev_z is not None:
            dt0 = t_frame - self._prev_tz
            can_init = valid & ~np.isnan(self._prev_z[:, 0]) & ~self.initialized
            if np.any(can_init) and dt0 > 1e-6:
                self._init_from_two(self._prev_z, z, dt0, can_init, t_frame)
        self._prev_z = z.copy()
        self._prev_tz = t_frame

        # 全初期化済みフィルタを観測時刻まで進める(観測なしは予測のみ)
        self.predict_to(t_frame)
        mask = valid & self.initialized & (np.abs(self.t - t_frame) < 1e-6)
        if not np.any(mask):
            return
        P = self.P[mask]
        x = self.x[mask]
        S = P[:, 0:3, 0:3] + self.r_meas * _I3          # (M,3,3)
        K = P[:, :, 0:3] @ np.linalg.inv(S)             # (M,7,3)
        innov = (z[mask] - x[:, 0:3])[:, :, None]
        x = x + (K @ innov)[:, :, 0]
        x[:, 6] = np.maximum(x[:, 6], 1e-5)
        KH = np.zeros_like(P)
        KH[:, :, 0:3] = K
        P = (np.tile(np.eye(7), (K.shape[0], 1, 1)) - KH) @ P
        self.x[mask] = x
        self.P[mask] = P

    def predict_landing(self, catch_height: float,
                        with_cov: bool = False,
                        t_horizon: float = 2.0, dt: float = 0.01
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """現在の推定から catch_height 交差(下降中)までを前方積分。

        返り値: xy_land (N,2), t_go (N,), cov2 (N,2,2) or None
        交差しない/未初期化は NaN。t_go は各フィルタ時刻 self.t 起点。
        """
        n = self.n
        xy = np.full((n, 2), np.nan)
        t_go = np.full(n, np.nan)
        cov2 = np.full((n, 2, 2), np.nan) if with_cov else None
        if not np.any(self.initialized):
            return xy, t_go, cov2
        idx = np.nonzero(self.initialized)[0]
        x = self.x[idx].copy()
        P = self.P[idx].copy() if with_cov else None
        m = len(idx)
        dt_row = np.full(m, dt)
        q_rate = self._q_rate()
        Q = q_rate[None] * dt
        elapsed = np.zeros(m)
        done = np.zeros(m, dtype=bool)
        for _ in range(int(t_horizon / dt)):
            run = ~done
            if not np.any(run):
                break
            z_prev = x[:, 2].copy()
            xn, F = _f_jac(x[run], self.g_vec, dt_row[run])
            if with_cov:
                P[run] = F @ P[run] @ F.transpose(0, 2, 1) + Q
            x[run] = xn
            elapsed[run] += dt
            crossed = run & (x[:, 2] <= catch_height) & (x[:, 5] < 0.0) \
                & (z_prev > catch_height)
            for j in np.nonzero(crossed)[0]:
                dz = z_prev[j] - x[j, 2]
                frac = (z_prev[j] - catch_height) / dz if dz > 1e-9 else 1.0
                xy_j = x[j, 0:2] - (1.0 - frac) * x[j, 3:5] * dt
                gi = idx[j]
                xy[gi] = xy_j
                t_go[gi] = elapsed[j] - (1.0 - frac) * dt
                if with_cov:
                    C = P[j, 0:2, 0:2].copy()
                    vz = min(x[j, 5], -0.1)
                    var_t = P[j, 2, 2] / vz ** 2
                    vxy = x[j, 3:5]
                    C += var_t * np.outer(vxy, vxy)
                    cov2[gi] = C
            done |= crossed
            done |= run & (x[:, 2] <= 0.0)   # 変則ケース打ち切り
        return xy, t_go, cov2
