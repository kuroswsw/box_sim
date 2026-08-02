"""カメラセンサモデル。

フレームレート fps で真値位置をサンプルし、ガウス雑音を加える。
時刻 t_frame の観測は t_frame + latency (露光遅延+処理遅延の合計) に
制御ループへ届く。latency は最大のボトルネックなので必ず可変。

近似の破綻箇所:
- 3D位置が等方ガウス雑音で直接得られる前提(ステレオ or 既知サイズ単眼)。
  実際は奥行き誤差が視差に反比例して大きく、距離依存・異方性。
  雑音 std をスライダで振り「どこまで許容か」を見る使い方を想定。
- 検出率100% (detect_height 以上にある間)。モーションブラーによる
  検出落ちは扱わない。fps を下げることで擬似的に評価は可能。
"""
from __future__ import annotations

import numpy as np

from .params import SensorParams


def make_measurements(times: np.ndarray, pos: np.ndarray,
                      sp: SensorParams,
                      rng: np.random.Generator
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """真値軌道 (T,N,3) から観測列を作る。

    返り値:
      t_frames  (K,)   : 撮像時刻
      t_avail   (K,)   : 制御ループに届く時刻 (= t_frames + latency)
      z_meas    (K,N,3): 雑音入り観測 (視野外は NaN)
    """
    dt = times[1] - times[0]
    t_end = times[-1]
    t_frames = np.arange(0.0, t_end, sp.frame_dt)
    idx = np.clip((t_frames / dt).round().astype(int), 0, len(times) - 1)
    true_at_frames = pos[idx]                     # (K,N,3)
    noise = rng.standard_normal(true_at_frames.shape) * sp.pos_noise_std
    z = true_at_frames + noise
    # 視野外 (低すぎる) は NaN
    invisible = true_at_frames[:, :, 2] < sp.detect_height
    z[invisible] = np.nan
    return t_frames, t_frames + sp.latency, z
