"""実測軌跡からのパラメータ同定 (キャリブレーション)。

入力: CSV (列: t, x, y, z)。1ファイル=1投擲。ヘッダ行は任意。
同定パラメータ: theta = [p0(3), v0(3), CdA, e]
  初期状態も未知数に含める(最初の数点から求めると雑音で偏るため)。
  反発係数 e はデータにバウンドが含まれる場合のみ実質的に同定される
  (含まれない場合は感度ゼロ → 信頼区間が発散するのでそれが分かる)。

方法: scipy.optimize.least_squares (Levenberg-Marquardt系)。
残差 = 実測位置 - 順モデル位置。順モデルはシミュレータ本体と同じ
simulate_batch を使う(モデルの二重管理を避ける)。

信頼区間: 収束点のヤコビアン J から cov = (JᵀJ)⁻¹·σ², σ²=RSS/(m-p)。
CdA と e の 95%CI を報告し、これをシミュレータの信頼区間として明示する。

破綻箇所:
- Cd·A 一定モデル自体の誤差は残差に残る。残差 RMS がカメラ計測誤差
  (数mm〜2cm) より有意に大きければ「このモデルではこれ以上詰められない」
  というシグナルであり、それも含めて表示する。
- 確率的物体(ティッシュ)には適用不可 (残差が同定誤差でなく外乱)。
"""
from __future__ import annotations

from dataclasses import dataclass
import io

import numpy as np
from scipy.optimize import least_squares

from .params import ObjectParams, WorldParams
from .physics import simulate_batch


@dataclass
class CalibResult:
    cda: float
    cda_std: float
    restitution: float
    restitution_std: float
    p0: np.ndarray
    v0: np.ndarray
    rms_before: float
    rms_after: float
    residuals_before: np.ndarray   # (M,) 各サンプルの距離残差
    residuals_after: np.ndarray
    t: np.ndarray
    note: str = ""

    def ci95(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.cda - 1.96 * self.cda_std, self.cda + 1.96 * self.cda_std),
                (self.restitution - 1.96 * self.restitution_std,
                 self.restitution + 1.96 * self.restitution_std))


def load_trajectory_csv(text_or_path) -> tuple[np.ndarray, np.ndarray]:
    """CSV → (t (M,), xyz (M,3))。ヘッダ有無を自動判別。"""
    if isinstance(text_or_path, (str,)) and "\n" not in text_or_path:
        raw = open(text_or_path, "r", encoding="utf-8").read()
    else:
        raw = text_or_path
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    start = 0
    try:
        [float(v) for v in lines[0].replace(",", " ").split()]
    except ValueError:
        start = 1
    data = np.loadtxt(io.StringIO("\n".join(lines[start:])), delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    t = data[:, 0]
    xyz = data[:, 1:4]
    order = np.argsort(t)
    return t[order], xyz[order]


def _forward(theta: np.ndarray, t_meas: np.ndarray, obj_mass: float,
             bounce_friction: float, world: WorldParams) -> np.ndarray:
    """順モデル: theta から t_meas 各時刻の位置 (M,3) を返す。"""
    p0, v0, cda, e = theta[0:3], theta[3:6], theta[6], theta[7]
    obj = ObjectParams(name="calib", label="calib", mass=obj_mass,
                       cda=max(cda, 1e-5),
                       restitution=float(np.clip(e, 0.0, 0.95)),
                       bounce_friction=bounce_friction)
    w = WorldParams(dt=world.dt, t_max=float(t_meas[-1]) + 0.05,
                    g=world.g, rho_air=world.rho_air)
    times, pos, _ = simulate_batch(p0[None], v0[None], obj, w, n_bounces=3)
    out = np.empty((len(t_meas), 3))
    for a in range(3):
        out[:, a] = np.interp(t_meas, times, pos[:, 0, a])
    return out


def calibrate(t: np.ndarray, xyz: np.ndarray, obj: ObjectParams,
              world: WorldParams | None = None) -> CalibResult:
    """1投擲分の実測軌跡から CdA と反発係数を同定する。"""
    if world is None:
        world = WorldParams(dt=0.004)
    t = t - t[0]
    m_pts = len(t)
    if m_pts < 8:
        raise ValueError("同定には最低8点必要です")

    # 初期値: 最初の3点から p0, v0。CdA, e は物体カタログの既定値
    v0_init = (xyz[2] - xyz[0]) / (t[2] - t[0])
    theta0 = np.concatenate([xyz[0], v0_init, [obj.cda, obj.restitution]])

    def resid(theta):
        return (_forward(theta, t, obj.mass, obj.bounce_friction, world)
                - xyz).ravel()

    # 「同定前」= CdA, e をカタログ既定値に固定し初期状態のみフィット。
    # (初期速度の読み取り誤差を除いた、純粋なモデルパラメータ誤差を見るため)
    def resid_fixed(theta6):
        th = np.concatenate([theta6, [obj.cda, obj.restitution]])
        return resid(th)

    sol_b = least_squares(resid_fixed, theta0[:6],
                          x_scale=np.concatenate([np.full(3, 0.05),
                                                  np.full(3, 0.5)]))
    r_before = sol_b.fun.reshape(-1, 3)
    res_before = np.linalg.norm(r_before, axis=1)

    scale = np.concatenate([np.full(3, 0.05), np.full(3, 0.5),
                            [max(obj.cda, 1e-3), 0.2]])
    theta0_full = np.concatenate([sol_b.x, [obj.cda, obj.restitution]])
    sol = least_squares(resid, theta0_full, x_scale=scale, method="trf",
                        bounds=(np.concatenate([np.full(6, -np.inf),
                                                [1e-5, 0.0]]),
                                np.concatenate([np.full(6, np.inf),
                                                [0.5, 0.95]])))
    r_after = sol.fun.reshape(-1, 3)
    res_after = np.linalg.norm(r_after, axis=1)

    # 共分散推定
    dof = max(3 * m_pts - 8, 1)
    sigma2 = float(sol.fun @ sol.fun) / dof
    JTJ = sol.jac.T @ sol.jac
    note = ""
    cov = np.linalg.pinv(JTJ) * sigma2   # 特異(感度なし)でも擬似逆で評価
    cda_std = float(np.sqrt(max(cov[6, 6], 0)))
    e_std = float(np.sqrt(max(cov[7, 7], 0)))
    if np.linalg.cond(JTJ) > 1e10:
        note = "ヤコビアンがほぼ特異 (一部パラメータに感度なし)。"
    if e_std > 0.3:
        note += " 反発係数の信頼区間が広い→データにバウンドが含まれていない可能性。"

    return CalibResult(
        cda=float(sol.x[6]), cda_std=cda_std,
        restitution=float(sol.x[7]), restitution_std=e_std,
        p0=sol.x[0:3], v0=sol.x[3:6],
        rms_before=float(np.sqrt(np.mean(res_before ** 2))),
        rms_after=float(np.sqrt(np.mean(res_after ** 2))),
        residuals_before=res_before, residuals_after=res_after,
        t=t, note=note.strip())


def make_sample_csv(obj: ObjectParams, noise_std: float = 0.008,
                    seed: int = 3, include_bounce: bool = True) -> str:
    """デモ用の擬似「実測」CSV を生成 (真値CdAを既定から+20%ずらす)。"""
    rng = np.random.default_rng(seed)
    true_obj = ObjectParams(
        name="true", label="true", mass=obj.mass, cda=obj.cda * 1.2,
        restitution=min(obj.restitution * 1.3, 0.9),
        bounce_friction=obj.bounce_friction)
    world = WorldParams(dt=0.002, t_max=1.6 if include_bounce else 0.75)
    p0 = np.array([[-2.5, 0.1, 0.9]])
    v0 = np.array([[3.4, -0.2, 3.1]])
    times, pos, _ = simulate_batch(p0, v0, true_obj, world)
    # 90fps 相当でサンプル
    t_s = np.arange(0.0, times[-1], 1 / 90.0)
    lines = ["t,x,y,z"]
    for ts in t_s:
        k = int(round(ts / world.dt))
        if k >= len(times):
            break
        p = pos[k, 0] + rng.standard_normal(3) * noise_std
        if pos[k, 0, 2] < 0.01 and ts > 0.5:
            continue
        lines.append(f"{ts:.4f},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}")
    return "\n".join(lines)
