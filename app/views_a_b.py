"""ビューA (3Dシーン) とビューB (予測誤差) の Figure 生成。"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from sim.loop import BatchResult
from sim.params import RobotParams

COLORS = {
    "pet_empty": "#4C9BE8", "pet_half": "#2E7DD1", "pet_full": "#1B4F9C",
    "trash_can": "#E8A33D", "trash_paper": "#8BC34A", "tissue": "#D9534F",
}


def _cov_ellipse_xy(mean: np.ndarray, cov: np.ndarray, nsig: float = 2.0,
                    n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """2D共分散の nσ 楕円 (既定は 2σ ≒ 95%)。"""
    try:
        w, v = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.array([]), np.array([])
    w = np.maximum(w, 1e-9)
    th = np.linspace(0, 2 * np.pi, n)
    pts = (v @ (np.sqrt(w)[:, None] * np.array([np.cos(th), np.sin(th)]))) * nsig
    return mean[0] + pts[0], mean[1] + pts[1]


def _circle3d(cx: float, cy: float, r: float, z: float, n: int = 40):
    th = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(th), cy + r * np.sin(th), np.full(n, z)


def scene_figure(res: BatchResult, rp: RobotParams, k: int,
                 obj_key: str = "pet_full") -> go.Figure:
    """ビューA: 時刻インデックス k での3Dシーン。"""
    fig = go.Figure()
    if res.times is None:
        return fig
    t = res.times[k]
    col = COLORS.get(obj_key, "#1B4F9C")

    # 軌道 (既通過を実線, 未来を薄く)。表示は4ステップ間引き
    p = res.obj_pos[:, 0, :]
    past, fut = p[:k + 1:4], p[k::4]
    fig.add_trace(go.Scatter3d(x=past[:, 0], y=past[:, 1], z=past[:, 2],
                               mode="lines", line=dict(color=col, width=5),
                               name="軌道(既)"))
    fig.add_trace(go.Scatter3d(x=fut[:, 0], y=fut[:, 1], z=fut[:, 2],
                               mode="lines",
                               line=dict(color=col, width=2, dash="dot"),
                               opacity=0.35, name="軌道(未来)"))
    fig.add_trace(go.Scatter3d(x=[p[k, 0]], y=[p[k, 1]], z=[p[k, 2]],
                               mode="markers",
                               marker=dict(size=7, color=col), name="物体"))

    # 機体 + 受け口
    rpos = res.robot_pos[k, 0]
    mr = res.mouth_r[k, 0]
    cx, cy, cz = _circle3d(rpos[0], rpos[1], rp.tread_radius, 0.02)
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines",
                               line=dict(color="#555", width=4), name="機体"))
    mx, my, mz = _circle3d(rpos[0], rpos[1], mr, rp.catch_height)
    deployed = mr > rp.mouth_base_radius + 1e-4
    fig.add_trace(go.Scatter3d(
        x=mx, y=my, z=mz, mode="lines",
        line=dict(color="#00A86B" if deployed else "#999", width=6),
        name=f"受け口 r={mr*100:.0f}cm"))
    # リブ4本 (展開の可視化)
    for a in range(4):
        th = a * np.pi / 2
        fig.add_trace(go.Scatter3d(
            x=[rpos[0], rpos[0] + mr * np.cos(th)],
            y=[rpos[1], rpos[1] + mr * np.sin(th)],
            z=[rp.catch_height * 0.55, rp.catch_height],
            mode="lines", line=dict(color="#00A86B" if deployed else "#aaa",
                                    width=3),
            showlegend=False))

    # 予測着地点 + 共分散楕円 (この時刻で最新の制御ティック)
    frame = None
    for f in res.frames:
        if f.t_avail > t:
            break
        if not np.isnan(f.pred_xy[0, 0]):
            frame = f
    if frame is not None and not np.isnan(frame.pred_xy[0, 0]):
        pxy = frame.pred_xy[0]
        fig.add_trace(go.Scatter3d(x=[pxy[0]], y=[pxy[1]],
                                   z=[rp.catch_height], mode="markers",
                                   marker=dict(size=6, color="#E8443D",
                                               symbol="x"),
                                   name="予測着地点"))
        if frame.cov2 is not None and not np.isnan(frame.cov2[0, 0, 0]):
            ex, ey = _cov_ellipse_xy(pxy, frame.cov2[0])
            if ex.size:
                fig.add_trace(go.Scatter3d(
                    x=ex, y=ey, z=np.full(len(ex), rp.catch_height),
                    mode="lines", line=dict(color="#E8443D", width=3),
                    opacity=0.7, name="2σ 共分散楕円"))
    # 真の着地点
    if not np.isnan(res.landed_xy[0, 0]):
        fig.add_trace(go.Scatter3d(
            x=[res.landed_xy[0, 0]], y=[res.landed_xy[0, 1]],
            z=[rp.catch_height], mode="markers",
            marker=dict(size=6, color="#111", symbol="diamond"),
            name="真の交差点"))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x [m]", range=[-4.0, 1.5]),
            yaxis=dict(title="y [m]", range=[-1.5, 1.5]),
            zaxis=dict(title="z [m]", range=[0, 2.0]),
            aspectmode="manual", aspectratio=dict(x=2.2, y=1.2, z=0.8),
            camera=dict(eye=dict(x=1.4, y=-1.8, z=0.9))),
        margin=dict(l=0, r=0, t=30, b=0), height=520,
        title=f"t = {t:.3f} s" + ("  [捕球]" if res.catch[0] else ""),
        legend=dict(x=0, y=1))
    return fig


def error_figure(results: dict[str, BatchResult]) -> go.Figure:
    """ビューB: 着地点予測誤差と EKF 共分散の時間推移 (物体種別で色分け)。"""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("着地点予測誤差 |pred - true| [m]",
                                        "EKF 位置共分散 trace(P_pp) [m²]"))
    for key, res in results.items():
        if not res.frames or np.isnan(res.landed_xy[0, 0]):
            continue
        col = COLORS.get(key, "#666")
        ts, errs, ptr = [], [], []
        for f in res.frames:
            if np.isnan(f.pred_xy[0, 0]):
                continue
            ts.append(f.t_avail)
            errs.append(float(np.linalg.norm(f.pred_xy[0] - res.landed_xy[0])))
            ptr.append(float(f.ekf_ptrace[0]))
        if not ts:
            continue
        fig.add_trace(go.Scatter(x=ts, y=errs, mode="lines+markers", name=key,
                                 line=dict(color=col)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=ptr, mode="lines", name=key,
                                 line=dict(color=col), showlegend=False),
                      row=2, col=1)
        if not np.isnan(res.t_land[0]):
            fig.add_vline(x=res.t_land[0], line=dict(color=col, dash="dot",
                                                     width=1))
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_xaxes(title_text="時刻 [s] (観測が制御に届いた時刻)",
                     row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=50, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=1.12))
    return fig
