"""パラメータ定義。全ステージ共通のデータクラス。

座標系: x 前方, y 左, z 上。地面 z=0。単位は SI (m, kg, s)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class WorldParams:
    g: float = 9.81          # 重力加速度 [m/s^2]
    rho_air: float = 1.204   # 空気密度 [kg/m^3]
    dt: float = 0.002        # 物理積分刻み [s]
    t_max: float = 3.0       # 1試行の最大時間 [s]


@dataclass
class ObjectParams:
    """投擲物。3種とも「並進は Cd·A 一定近似の質点」+ 姿勢は表示用に別積分。

    割り切り:
    - PETボトル: タンブリングによる Cd·A の変動を無視(定数近似)。
      回転は慣性テンソルで自由回転として積分するが並進へは影響させない。
      半分入りのスロッシング(液体の揺動)は無視 → 破綻箇所は README 参照。
    - 一般ゴミ: 同上の質点近似。
    - ティッシュ: 剛体近似が破綻するため等価球 + 大Cd + OU過程外乱。
      「決定論的に予測できない」性質の再現が目的で、個々の軌道精度は求めない。
    """
    name: str = "pet_full"
    label: str = "PET(満)"
    mass: float = 0.525            # [kg]
    cda: float = 0.006             # Cd*A [m^2] (並進抗力, 一定近似)
    restitution: float = 0.12      # 反発係数 (z方向)
    bounce_friction: float = 0.35  # バウンド時の水平速度残存率
    # 慣性テンソル (主軸, 対角) [kg m^2] — 表示/バウンド姿勢用
    inertia: tuple[float, float, float] = (2.3e-3, 2.3e-3, 2.4e-4)
    # ティッシュ用 OU 外乱 (加速度 [m/s^2])
    ou_sigma: float = 0.0          # 定常標準偏差
    ou_tau: float = 0.15           # 時定数 [s]
    stochastic: bool = False       # True なら決定論的予測不能フラグ
    # EKF が使う事前分布 (物体分類器が与える想定)
    beta_prior_mean: float = 0.0   # beta = rho*CdA/(2m) — __post_init__ で計算
    beta_prior_std: float = 0.0

    def __post_init__(self) -> None:
        rho = 1.204
        beta = rho * self.cda / (2.0 * self.mass)
        if self.beta_prior_mean == 0.0:
            self.beta_prior_mean = beta
        if self.beta_prior_std == 0.0:
            # 分類器の事前分布幅: 決定論物体は±30%, 確率的物体は±80%
            self.beta_prior_std = beta * (0.8 if self.stochastic else 0.3)

    @property
    def beta(self) -> float:
        """抗力係数 beta = rho*CdA/(2m). 加速度 = -beta*|v|*v"""
        return 1.204 * self.cda / (2.0 * self.mass)


def make_objects() -> dict[str, ObjectParams]:
    """3種の物体カタログ。PETは中身量で慣性テンソルと質量を変える。"""
    objs = {}
    # 500ml PET: 半径0.033m, 長さ0.21m。円筒近似で慣性を計算
    def cyl_inertia(m: float, r: float = 0.033, h: float = 0.21):
        ixx = m * (3 * r * r + h * h) / 12.0
        izz = m * r * r / 2.0
        return (ixx, ixx, izz)

    objs["pet_empty"] = ObjectParams(
        name="pet_empty", label="PET(空)", mass=0.025, cda=0.0075,
        restitution=0.45, bounce_friction=0.5, inertia=cyl_inertia(0.025))
    objs["pet_half"] = ObjectParams(
        name="pet_half", label="PET(半分)", mass=0.275, cda=0.0065,
        restitution=0.18, bounce_friction=0.35, inertia=cyl_inertia(0.275))
    objs["pet_full"] = ObjectParams(
        name="pet_full", label="PET(満)", mass=0.525, cda=0.006,
        restitution=0.12, bounce_friction=0.3, inertia=cyl_inertia(0.525))
    # 一般ゴミ: アルミ缶想定
    objs["trash_can"] = ObjectParams(
        name="trash_can", label="缶", mass=0.016, cda=0.004,
        restitution=0.35, bounce_friction=0.5,
        inertia=(6.0e-6, 6.0e-6, 4.5e-6))
    # 紙くず(丸めた紙): やや確率的
    objs["trash_paper"] = ObjectParams(
        name="trash_paper", label="紙くず", mass=0.005, cda=0.005,
        restitution=0.15, bounce_friction=0.3,
        inertia=(2.0e-6, 2.0e-6, 2.0e-6),
        ou_sigma=0.8, stochastic=True)
    # ティッシュ: 等価球 r=0.04m, Cd=2.0 → CdA≈0.010。強いOU外乱
    objs["tissue"] = ObjectParams(
        name="tissue", label="ティッシュ", mass=0.003, cda=0.010,
        restitution=0.02, bounce_friction=0.1,
        inertia=(1.6e-6, 1.6e-6, 1.6e-6),
        ou_sigma=3.0, ou_tau=0.15, stochastic=True)
    return objs


@dataclass
class ThrowPreset:
    """投擲プリセット。平均と標準偏差(人間のばらつき, 正規分布)。"""
    name: str = "underhand"
    label: str = "下手投げ"
    pos_mean: tuple[float, float, float] = (-2.5, 0.0, 0.9)
    pos_std: tuple[float, float, float] = (0.05, 0.05, 0.05)
    vel_mean: tuple[float, float, float] = (3.2, 0.0, 3.0)
    vel_std: tuple[float, float, float] = (0.35, 0.25, 0.35)
    omega_std: float = 8.0   # 初期角速度の各軸標準偏差 [rad/s]


def make_throws() -> dict[str, ThrowPreset]:
    return {
        "underhand": ThrowPreset(
            name="underhand", label="下手投げ",
            pos_mean=(-2.5, 0.0, 0.9), vel_mean=(3.2, 0.0, 3.0),
            vel_std=(0.35, 0.25, 0.35), omega_std=8.0),
        "overhand": ThrowPreset(
            name="overhand", label="オーバースロー",
            pos_mean=(-3.5, 0.0, 1.6), vel_mean=(5.0, 0.0, 1.2),
            vel_std=(0.5, 0.35, 0.4), omega_std=15.0),
        "flick": ThrowPreset(
            name="flick", label="弾く",
            pos_mean=(-1.8, 0.0, 1.1), vel_mean=(2.2, 0.0, 1.6),
            pos_std=(0.03, 0.03, 0.03),
            vel_std=(0.5, 0.4, 0.5), omega_std=25.0),
    }


@dataclass
class RobotParams:
    """3輪オムニ機体。加速度支配の前提に合わせた既定値。

    0.3s で 11cm → a ≈ 2.4 m/s^2 が基準。
    """
    mass: float = 2.5              # [kg]
    tread_radius: float = 0.11     # 車輪接地円半径 [m]
    com_height: float = 0.055      # 重心高さ ≤ tread_radius/2 … 転倒条件は h ≤ R/2 ではなく仕様の「重心高さ≤トレッド半分」に従う
    a_max: float = 2.4             # 電流制限由来の最大加速度 [m/s^2]
    v_max: float = 1.6             # 最大速度 [m/s]
    tau_mech: float = 0.03         # 駆動系1次遅れ時定数 [s] (FOC電流制御+機械系)
    mu_static: float = 0.7         # 静止摩擦係数 (スリップ限界)
    mu_kinetic: float = 0.5        # 動摩擦係数 (スリップ中)
    # 受け口
    mouth_base_radius: float = 0.10     # 非展開時の受け口半径 [m]
    mouth_deploy_radius: float = 0.20   # 展開時の受け口半径 [m]
    deploy_time: float = 0.12           # シアー解放→全展開 [s]
    catch_height: float = 0.35          # 受け口リム高さ [m]
    reload_time: float = 3.0            # サーボウインチ再装填 [s]

    def tipping_a_max(self, g: float = 9.81) -> float:
        """転倒限界加速度 a_tip = g * R_tread / h_com (準静的)。"""
        return g * self.tread_radius / max(self.com_height, 1e-6)


@dataclass
class SensorParams:
    """カメラ系。合計レイテンシ = 露光遅延 + 処理遅延。最大のボトルネック。"""
    fps: float = 60.0
    latency: float = 0.08          # 露光+処理の合計遅延 [s]
    pos_noise_std: float = 0.02    # 3D位置計測ノイズ [m]
    detect_height: float = 0.25    # これ未満の高さでは検出打ち切り(視野外)

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps


@dataclass
class ControlParams:
    lead_time: float = 0.0       # LSTM事前予測によるリード時間 [s] (感度解析用)
    lstm_err_std: float = 0.15   # 事前予測の着地点誤差 std [m] (実測で同定する想定)
    deploy_margin: float = 0.01  # 展開判断のヒステリシス余裕 [m]
    deploy_confirm: int = 2      # 展開に必要な連続判定フレーム数
    min_updates: int = 3         # 制御を有効にする最小EKF更新回数
