"""Stage2 スモーク: 制御ループ一式 (EKF+予測+機体+展開判断) の妥当性確認。"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from sim.params import (ControlParams, RobotParams, SensorParams, WorldParams,
                        make_objects, make_throws)
from sim.loop import run_batch

world = WorldParams()
rp = RobotParams()
sp = SensorParams()
cp = ControlParams()
objs = make_objects()
throws = make_throws()

# --- 単発 (record=True): PET満, 下手投げ, ばらつきなし ---
res = run_batch(objs["pet_full"], throws["underhand"], rp, sp, cp, world,
                n=1, rng=np.random.default_rng(42), record=True,
                deterministic_throw=True)
print(f"[単発] 着地 t={res.t_land[0]:.3f}s xy=({res.landed_xy[0,0]:.3f},"
      f"{res.landed_xy[0,1]:.3f}) miss={res.miss_dist[0]*100:.1f}cm "
      f"catch={res.catch[0]} deployed={res.deployed[0]}")
assert len(res.frames) > 5, "制御ティックが記録されていない"
# EKF 予測誤差がフレームを追うごとに縮むか
errs = [np.linalg.norm(f.pred_xy[0] - res.landed_xy[0]) for f in res.frames
        if not np.isnan(f.pred_xy[0, 0])]
print(f"[単発] 予測誤差: 初回={errs[0]*100:.1f}cm → 最終={errs[-1]*100:.1f}cm")
assert errs[-1] < errs[0], "EKF予測が収束していない"
assert errs[-1] < 0.10, "最終予測誤差が大きすぎる"
# 機体は動いたか
moved = np.linalg.norm(res.robot_pos[-1, 0] - res.robot_pos[0, 0])
print(f"[単発] 機体移動量={moved*100:.1f}cm, スリップ率={res.slip_frac[0]*100:.0f}%")

# --- バッチ 200試行: ばらつきあり ---
for name in ["pet_full", "tissue"]:
    t0 = time.perf_counter()
    res = run_batch(objs[name], throws["underhand"], rp, sp, cp, world,
                    n=200, rng=np.random.default_rng(7))
    el = time.perf_counter() - t0
    print(f"[{name}] 捕球率={res.catch_rate*100:.0f}% "
          f"展開率={res.deployed.mean()*100:.0f}% "
          f"miss中央値={np.nanmedian(res.miss_dist)*100:.1f}cm ({el:.1f}s)")

# 受け口を大きくすると捕球率が上がるはず
rp_big = RobotParams(mouth_base_radius=0.10, mouth_deploy_radius=0.35)
res_small = run_batch(objs["pet_full"], throws["underhand"],
                      RobotParams(mouth_deploy_radius=0.12), sp, cp, world,
                      n=200, rng=np.random.default_rng(7))
res_big = run_batch(objs["pet_full"], throws["underhand"], rp_big, sp, cp,
                    world, n=200, rng=np.random.default_rng(7))
print(f"受け口 12cm: {res_small.catch_rate*100:.0f}% / "
      f"35cm: {res_big.catch_rate*100:.0f}%")
assert res_big.catch_rate >= res_small.catch_rate
# レイテンシを悪化させると捕球率が下がるはず
res_slow = run_batch(objs["pet_full"], throws["underhand"], rp,
                     SensorParams(latency=0.25), cp, world,
                     n=200, rng=np.random.default_rng(7))
res_fast = run_batch(objs["pet_full"], throws["underhand"], rp,
                     SensorParams(latency=0.02), cp, world,
                     n=200, rng=np.random.default_rng(7))
print(f"latency 20ms: {res_fast.catch_rate*100:.0f}% / "
      f"250ms: {res_slow.catch_rate*100:.0f}%")
assert res_fast.catch_rate >= res_slow.catch_rate
print("Stage2 smoke OK")
