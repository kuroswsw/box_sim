"""Stage1 スモークテスト: PET(満) 1本を2D(x-z)で飛ばし、妥当性を確認する。"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from sim.params import WorldParams, make_objects, make_throws
from sim.physics import simulate_batch, landing_time_index

world = WorldParams()
obj = make_objects()["pet_full"]
throw = make_throws()["underhand"]

pos0 = np.array([throw.pos_mean])
vel0 = np.array([throw.vel_mean])
times, pos, vel = simulate_batch(pos0, vel0, obj, world, rng=np.random.default_rng(0))

# 真空弾道との比較 (抗力が小さいので近いはず)
t_land_vac = (vel0[0, 2] + np.sqrt(vel0[0, 2] ** 2 + 2 * 9.81 * pos0[0, 2])) / 9.81
idx = landing_time_index(pos, 0.35, vel=vel)
assert idx[0] > 0, "catch height を横切らない"
t_land = times[idx[0]]
x_land = pos[idx[0], 0, 0]
print(f"真空理論 全落下時間: {t_land_vac:.3f}s")
print(f"シミュ z=0.35m 通過: t={t_land:.3f}s, x={x_land:.3f}m (投擲点 x={pos0[0,0]})")
print(f"最高到達: z={pos[:,0,2].max():.3f}m")
assert 0.3 < t_land < t_land_vac + 0.1
assert -2.5 < x_land < 3.0
# 抗力の影響: 満タンPETは軽微(数cm)のはず
vac_x = pos0[0, 0] + vel0[0, 0] * t_land
print(f"抗力による x 減少: {vac_x - x_land:.4f}m")
assert 0 <= vac_x - x_land < 0.15

# ティッシュ: 同条件で20本 → ばらつき(OU)と大抗力で明確に手前落ち
tis = make_objects()["tissue"]
p0 = np.tile(pos0, (20, 1)); v0 = np.tile(vel0, (20, 1))
_, pos_t, vel_t = simulate_batch(p0, v0, tis, world, rng=np.random.default_rng(1))
idx_t = landing_time_index(pos_t, 0.35, vel=vel_t)
xy = np.array([pos_t[i, j, :2] for j, i in enumerate(idx_t) if i > 0])
print(f"ティッシュ 着地点平均 x={xy[:,0].mean():.3f}m, 散らばり std={xy.std(axis=0)}")
assert xy[:, 0].mean() < x_land - 0.3, "ティッシュは大きく手前落ちするはず"
assert xy.std(axis=0).max() > 0.03, "OU外乱による散らばりが出るはず"
print("Stage1 smoke OK")
