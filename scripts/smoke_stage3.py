"""Stage3 スモーク: キャリブレーションとMCヒートマップ。"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from sim.params import (ControlParams, RobotParams, SensorParams, WorldParams,
                        make_objects, make_throws)
from sim.calibration import calibrate, load_trajectory_csv, make_sample_csv
from sim.montecarlo import mc_heatmap

objs = make_objects()
obj = objs["pet_empty"]   # 軽い物体ほど抗力感度が高く同定しやすい

# --- キャリブレーション: 真値 CdA=既定*1.2 を復元できるか ---
csv = make_sample_csv(obj)
t, xyz = load_trajectory_csv(csv)
print(f"サンプル点数: {len(t)}")
res = calibrate(t, xyz, obj)
true_cda = obj.cda * 1.2
true_e = min(obj.restitution * 1.3, 0.9)
(ci_cda, ci_e) = res.ci95()
print(f"CdA 同定: {res.cda:.5f} ±{1.96*res.cda_std:.5f} (真値 {true_cda:.5f})")
print(f"e   同定: {res.restitution:.3f} ±{1.96*res.restitution_std:.3f} "
      f"(真値 {true_e:.3f})")
print(f"残差RMS: 同定前 {res.rms_before*100:.2f}cm → 同定後 {res.rms_after*100:.2f}cm")
print(f"note: {res.note or '-'}")
assert ci_cda[0] < true_cda < ci_cda[1], "CdA が95%CIに入っていない"
assert res.rms_after < res.rms_before
assert res.rms_after < 0.02, "残差が計測雑音レベルまで落ちていない"

# --- MC ヒートマップ (小規模) ---
world = WorldParams()
t0 = time.perf_counter()
grid = mc_heatmap(objs["pet_full"], make_throws()["underhand"], RobotParams(),
                  SensorParams(), ControlParams(), world,
                  "mouth", np.linspace(0.10, 0.40, 4),
                  "latency", np.linspace(0.02, 0.20, 3),
                  n_trials=300, seed=1)
el = time.perf_counter() - t0
print(f"heatmap 4x3x300 trials: {el:.1f}s")
print(np.round(grid, 2))
# 単調性: 受け口大ほど高く, 遅延大ほど低い (概ね)
assert grid[0, -1] >= grid[0, 0] - 0.05
assert grid[0, 0] >= grid[-1, 0] - 0.05
print("Stage3 smoke OK")
