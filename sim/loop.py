"""制御ループ統合: 検出 → EKF → 着地点予測 → 時間最適軌道 → 展開判断。

実機と同じ構成:
- カメラはフレームレートで撮像し、観測は latency 遅れて制御に届く。
- EKF はバッチで回し、観測到着イベントごとに更新+着地点予測。
- 軌道生成は「到着時刻指定の加速度則」 a = 2((r_target-r) - v·t_go)/t_go²
  を a_max でクリップ。t_go が足りない領域では自動的に全力加速
  (時間最適バンバンの実用近似。真の時間最適との差は数% — README参照)。
- 展開判断: 予測着地点が「機体が着地時刻までに到達できる基部円」の外に
  出たときのみトリガ (仕様通り)。誤展開防止に連続判定フレーム数を要求。

LSTM事前予測 (感度解析用):
  lead_time > 0 なら、リリース lead_time 秒前から「誤差 lstm_err_std の
  着地点予測」に向かって走り出せる。実LSTMの代わりに誤差入り予測で代替。
  = 「事前予測が何ms早いと捕球率が何%上がるか」だけを見るための機構。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ekf import EkfBatch
from .params import (ControlParams, ObjectParams, RobotParams, SensorParams,
                     ThrowPreset, WorldParams)
from .physics import landing_time_index, simulate_batch
from .robot import RobotBatch
from .sensor import make_measurements
from .throws import sample_throws


@dataclass
class FrameLog:
    """制御ティック1回分の記録 (record=True 時のみ)。"""
    t_frame: float
    t_avail: float
    pred_xy: np.ndarray          # (N,2)
    t_land_abs: np.ndarray       # (N,)
    cov2: np.ndarray | None      # (N,2,2)
    ekf_x: np.ndarray            # (N,7)
    ekf_ptrace: np.ndarray       # (N,) 位置共分散トレース


@dataclass
class BatchResult:
    n: int
    catch: np.ndarray            # (N,) bool
    miss_dist: np.ndarray        # (N,) 交差時の受け口中心との距離 [m]
    landed_xy: np.ndarray        # (N,2) 真の交差点
    t_land: np.ndarray           # (N,) 真の交差時刻
    deployed: np.ndarray         # (N,) bool
    deploy_t0: np.ndarray        # (N,)
    slip_frac: np.ndarray        # (N,) スリップしていた時間割合
    # record=True のみ
    times: np.ndarray | None = None
    obj_pos: np.ndarray | None = None      # (T,N,3)
    robot_pos: np.ndarray | None = None    # (T,N,2)
    mouth_r: np.ndarray | None = None      # (T,N)
    frames: list[FrameLog] = field(default_factory=list)

    @property
    def catch_rate(self) -> float:
        return float(np.mean(self.catch))


def run_batch(obj: ObjectParams, preset: ThrowPreset, rp: RobotParams,
              sp: SensorParams, cp: ControlParams, world: WorldParams,
              n: int, rng: np.random.Generator | None = None,
              record: bool = False,
              deterministic_throw: bool = False) -> BatchResult:
    if rng is None:
        rng = np.random.default_rng()
    dt = world.dt

    # --- 1. 投擲サンプル & 真値軌道 ---
    pos0, vel0, _omega0 = sample_throws(preset, n, rng,
                                        deterministic=deterministic_throw)
    times, pos, vel = simulate_batch(pos0, vel0, obj, world, rng=rng)
    steps = len(times)

    # 真の捕球面交差
    idx_cross = landing_time_index(pos, rp.catch_height, vel=vel)
    has_cross = idx_cross >= 0
    t_land = np.where(has_cross, times[np.maximum(idx_cross, 0)], np.nan)
    landed_xy = np.full((n, 2), np.nan)
    landed_xy[has_cross] = pos[idx_cross[has_cross], np.nonzero(has_cross)[0], 0:2]

    # --- 2. 観測列 ---
    t_frames, t_avail, z_meas = make_measurements(times, pos, sp, rng)
    n_frames = len(t_frames)

    # --- 3. 機体 & EKF ---
    robot = RobotBatch(n, rp, world)
    ekf = EkfBatch(n, obj, sp, world)
    n_updates = np.zeros(n, dtype=int)
    deploy_votes = np.zeros(n, dtype=int)
    pred_xy_cur = np.full((n, 2), np.nan)
    t_land_abs_cur = np.full(n, np.nan)
    slip_steps = np.zeros(n, dtype=int)

    # --- 3a. LSTM事前予測フェーズ (t<0) ---
    if cp.lead_time > 0:
        pre_target = landed_xy + rng.standard_normal((n, 2)) * cp.lstm_err_std
        pre_target[~has_cross] = 0.0
        pre_steps = int(cp.lead_time / dt)
        for k in range(pre_steps):
            t_pre = -cp.lead_time + k * dt
            t_go = t_land - t_pre
            ok = has_cross & (t_go > 0.05)
            a = np.zeros((n, 2))
            dr = pre_target - robot.pos
            a[ok] = 2.0 * (dr[ok] - robot.vel[ok] * t_go[ok, None]) \
                / (t_go[ok, None] ** 2)
            norm = np.linalg.norm(a, axis=1, keepdims=True)
            a = np.where(norm > rp.a_max, a * rp.a_max / np.maximum(norm, 1e-9), a)
            robot.a_cmd = a
            robot.step(dt)

    # --- 4. メインループ ---
    frames: list[FrameLog] = []
    robot_pos_hist = np.zeros((steps, n, 2)) if record else None
    mouth_r_hist = np.zeros((steps, n)) if record else None
    rob_at_cross = np.full((n, 2), np.nan)
    mouth_at_cross = np.full(n, np.nan)
    j = 0  # 観測到着ポインタ

    for k in range(steps):
        t = times[k]
        # 観測到着イベント
        while j < n_frames and t_avail[j] <= t:
            ekf.update(t_frames[j], z_meas[j])
            n_updates += (~np.isnan(z_meas[j, :, 0])).astype(int)
            if np.any(ekf.initialized):
                pxy, tgo, cov2 = ekf.predict_landing(
                    rp.catch_height, with_cov=record)
                valid = ~np.isnan(tgo)
                pred_xy_cur[valid] = pxy[valid]
                t_land_abs_cur[valid] = ekf.t[valid] + tgo[valid]
                if record:
                    frames.append(FrameLog(
                        t_frame=t_frames[j], t_avail=t_avail[j],
                        pred_xy=pxy.copy(),
                        t_land_abs=np.where(valid, ekf.t + tgo, np.nan),
                        cov2=None if cov2 is None else cov2.copy(),
                        ekf_x=ekf.x.copy(),
                        ekf_ptrace=np.trace(ekf.P[:, 0:3, 0:3],
                                            axis1=1, axis2=2)))
                # --- 制御則 + 展開判断 (観測到着ティックで更新) ---
                t_go_now = t_land_abs_cur - t
                active = (n_updates >= cp.min_updates) & ~np.isnan(t_go_now) \
                    & ~np.isnan(pred_xy_cur[:, 0])
                mov = active & (t_go_now > 0.05)
                a = np.zeros((n, 2))
                dr = pred_xy_cur - robot.pos
                a[mov] = 2.0 * (dr[mov] - robot.vel[mov] * t_go_now[mov, None]) \
                    / (t_go_now[mov, None] ** 2)
                # 着地後/直前はブレーキ
                brake = active & (t_go_now <= 0.05)
                a[brake] = -robot.vel[brake] / 0.1
                norm = np.linalg.norm(a, axis=1, keepdims=True)
                a = np.where(norm > rp.a_max,
                             a * rp.a_max / np.maximum(norm, 1e-9), a)
                robot.a_cmd = a
                # 展開判断: 到達可能性を1次元近似で見積もる
                d = np.linalg.norm(dr, axis=1)
                dirv = dr / np.maximum(d[:, None], 1e-9)
                v_par = np.sum(robot.vel * dirv, axis=1)
                tg = np.maximum(t_go_now, 0.0)
                reach = np.minimum(v_par * tg + 0.5 * rp.a_max * tg * tg,
                                   rp.v_max * tg)
                miss_est = np.maximum(0.0, d - np.maximum(reach, 0.0))
                vote = active & (miss_est > rp.mouth_base_radius
                                 - cp.deploy_margin)
                deploy_votes = np.where(vote, deploy_votes + 1, 0)
                robot.trigger_deploy(deploy_votes >= cp.deploy_confirm, t)
            j += 1
        robot.step(dt)
        slip_steps += robot.slipping.astype(int)
        # 交差時点の機体状態を捕捉 (捕球判定用)
        crossing_now = idx_cross == k
        if np.any(crossing_now):
            rob_at_cross[crossing_now] = robot.pos[crossing_now]
            mouth_at_cross[crossing_now] = robot.mouth_radius(t)[crossing_now]
        if record:
            robot_pos_hist[k] = robot.pos
            mouth_r_hist[k] = robot.mouth_radius(t)

    # --- 5. 捕球判定: 交差時に受け口有効半径内なら捕球 ---
    miss = np.linalg.norm(landed_xy - rob_at_cross, axis=1)
    catch = has_cross & (miss <= mouth_at_cross)
    return BatchResult(
        n=n, catch=catch, miss_dist=miss, landed_xy=landed_xy, t_land=t_land,
        deployed=~np.isnan(robot.deploy_t0), deploy_t0=robot.deploy_t0.copy(),
        slip_frac=slip_steps / steps,
        times=times if record else None,
        obj_pos=pos if record else None,
        robot_pos=robot_pos_hist, mouth_r=mouth_r_hist, frames=frames)
