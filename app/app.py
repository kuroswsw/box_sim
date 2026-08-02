"""移動式ゴミ箱シミュレータ GUI (Plotly Dash)。

起動: python app/app.py  →  http://127.0.0.1:8050

タブ構成:
  A 3Dシーン / B 予測誤差 / C 設計探索(主目的) / D 学習 / キャリブレーション
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
from dash import (Dash, Input, Output, State, dcc, html, no_update)

from app.views_a_b import COLORS, error_figure, scene_figure
from sim.calibration import calibrate, load_trajectory_csv, make_sample_csv
from sim.loop import run_batch
from sim.montecarlo import PARAM_SPECS, mc_heatmap
from sim.params import (ControlParams, RobotParams, SensorParams, WorldParams,
                        make_objects, make_throws)

OBJECTS = make_objects()
THROWS = make_throws()
RUNS_DIR = pathlib.Path(__file__).resolve().parents[1] / "learning" / "runs"
_CACHE: dict[str, object] = {}   # ローカル単一ユーザ前提のメモリキャッシュ

app = Dash(__name__, title="box_sim — 自動捕球ロボット設計シミュレータ")

obj_options = [{"label": o.label, "value": k} for k, o in OBJECTS.items()]
throw_options = [{"label": t.label, "value": k} for k, t in THROWS.items()]


def _num_input(id_, value, step=0.01, mn=None, mx=None, w="90px"):
    return dcc.Input(id=id_, type="number", value=value, step=step,
                     min=mn, max=mx, style={"width": w})


# ---------------- レイアウト ----------------
tab_a = html.Div([
    html.Div([
        html.Label("物体"), dcc.Dropdown(obj_options, "pet_full", id="a-obj",
                                        clearable=False,
                                        style={"width": "160px"}),
        html.Label("投擲"), dcc.Dropdown(throw_options, "underhand",
                                        id="a-throw", clearable=False,
                                        style={"width": "170px"}),
        html.Label("シード"), _num_input("a-seed", 0, 1, 0, 9999, "70px"),
        dcc.Checklist([{"label": "ばらつき注入", "value": "rand"}],
                      ["rand"], id="a-rand"),
        html.Button("▶ 実行", id="a-run", n_clicks=0,
                    style={"fontWeight": "bold"}),
    ], style={"display": "flex", "gap": "10px", "alignItems": "center",
              "flexWrap": "wrap"}),
    dcc.Loading(dcc.Graph(id="a-scene")),
    html.Div([
        html.Button("再生", id="a-play", n_clicks=0),
        html.Button("一時停止", id="a-pause", n_clicks=0),
        dcc.Dropdown([{"label": "等速", "value": 50},
                      {"label": "1/4 スロー", "value": 12},
                      {"label": "1/10 スロー", "value": 5}],
                     12, id="a-speed", clearable=False,
                     style={"width": "140px"}),
        html.Div(dcc.Slider(0, 100, 1, value=0, id="a-time",
                            marks=None,
                            tooltip={"placement": "bottom"}),
                 style={"flexGrow": 1}),
    ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
    dcc.Interval(id="a-interval", interval=200, disabled=True),
    dcc.Store(id="a-token"),
    html.Div(id="a-summary", style={"marginTop": "6px", "color": "#444"}),
])

tab_b = html.Div([
    html.Div([
        html.Label("投擲"), dcc.Dropdown(throw_options, "underhand",
                                        id="b-throw", clearable=False,
                                        style={"width": "170px"}),
        html.Label("シード"), _num_input("b-seed", 0, 1, 0, 9999, "70px"),
        html.Button("実行 (全物体種)", id="b-run", n_clicks=0,
                    style={"fontWeight": "bold"}),
        html.Span("同一の投擲条件で全物体を1投ずつ飛ばし、"
                  "着地点予測誤差とEKF共分散の収束を比較します。",
                  style={"color": "#666"}),
    ], style={"display": "flex", "gap": "10px", "alignItems": "center",
              "flexWrap": "wrap"}),
    dcc.Loading(dcc.Graph(id="b-graph")),
    html.Div("見方: ティッシュ/紙くずは共分散が収束しない(=決定論的に予測"
             "できない)ことが確認できるはず。これはモデルの意図した性質。",
             style={"color": "#666"}),
])


_SLIDER_LABEL_STYLE = {"fontSize": "0.9em", "whiteSpace": "nowrap"}
_SLIDER_VALUE_STYLE = {"fontWeight": "bold", "color": "#3B1E8C",
                       "marginLeft": "4px"}


def _slider_block(label: str, id_: str, lo: float, hi: float, value: float,
                  step: float | None = None):
    """ラベル内に現在値を出すスライダ。

    ツールチップ(always_visible)は下の要素に隠れて読めなくなるため使わない。
    単位は値側に付けるので、ラベル末尾の "[m]" 等は落とす。
    """
    name = re.sub(r"\s*\[[^\]]*\]\s*$", "", label)
    return html.Div([
        html.Label([name, html.Span(id=f"{id_}-val",
                                    style=_SLIDER_VALUE_STYLE)],
                   style=_SLIDER_LABEL_STYLE),
        dcc.Slider(lo, hi, step=step, value=value, id=id_, marks=None,
                   tooltip={"placement": "top", "always_visible": False},
                   updatemode="drag"),
    ], style={"width": "220px"})


def _slider(name: str, id_: str, value: float):
    spec = PARAM_SPECS[name]
    return _slider_block(spec["label"], id_, spec["lo"], spec["hi"], value)


tab_c = html.Div([
    html.P("受け口径・展開時間・最大加速度・センサ遅延の4軸から2軸を選び、"
           "モンテカルロ試行の捕球率をヒートマップ表示します (本ツールの主目的)。"
           "残り2軸はスライダ値に固定。全格子点で共通乱数を使うため格子間の差は"
           "設計の差を表します。", style={"color": "#444"}),
    html.Div([
        html.Label("X軸"),
        dcc.Dropdown([{"label": PARAM_SPECS[k]["label"], "value": k}
                      for k in PARAM_SPECS], "mouth", id="c-x",
                     clearable=False, style={"width": "190px"}),
        html.Label("Y軸"),
        dcc.Dropdown([{"label": PARAM_SPECS[k]["label"], "value": k}
                      for k in PARAM_SPECS], "latency", id="c-y",
                     clearable=False, style={"width": "190px"}),
        html.Label("物体"),
        dcc.Dropdown(obj_options, "pet_full", id="c-obj", clearable=False,
                     style={"width": "150px"}),
        html.Label("投擲"),
        dcc.Dropdown(throw_options, "underhand", id="c-throw",
                     clearable=False, style={"width": "160px"}),
        html.Label("試行数/格子点"),
        dcc.Dropdown([{"label": str(v), "value": v} for v in
                      [200, 500, 1000]], 1000, id="c-trials",
                     clearable=False, style={"width": "100px"}),
        html.Button("探索実行", id="c-run", n_clicks=0,
                    style={"fontWeight": "bold"}),
    ], style={"display": "flex", "gap": "10px", "alignItems": "center",
              "flexWrap": "wrap"}),
    html.Div([
        _slider("mouth", "c-mouth", RobotParams().mouth_deploy_radius),
        _slider("deploy", "c-deploy", RobotParams().deploy_time),
        _slider("amax", "c-amax", RobotParams().a_max),
        _slider("latency", "c-latency", SensorParams().latency),
        _slider_block("LSTM事前予測リード時間 [s]", "c-lead", 0.0, 0.3, 0.0),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
              "marginTop": "8px", "marginBottom": "10px"}),
    html.Div(id="c-warn", style={"color": "#B00", "marginTop": "4px"}),
    dcc.Loading(dcc.Graph(id="c-heatmap"), type="cube"),
], )

tab_d = html.Div([
    html.P("事前予測LSTM (骨格→初速回帰) の学習結果を表示します。学習データは"
           "実測のみ (learning/train_lstm.py 参照)。シミュレータ内では学習済み"
           "精度を lead_time / lstm_err_std に縮約してビューCの感度解析に"
           "使います。", style={"color": "#444"}),
    html.Div([
        html.Label("学習ラン"),
        dcc.Dropdown(id="d-run", clearable=False, style={"width": "280px"}),
        html.Button("更新", id="d-refresh", n_clicks=0),
    ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
    dcc.Loading(dcc.Graph(id="d-graph")),
])

tab_cal = html.Div([
    html.P("実測した投擲軌跡 CSV (列: t,x,y,z) を読み込み、Cd·A と反発係数を"
           "最小二乗同定します。同定前後の残差と95%信頼区間を表示します。",
           style={"color": "#444"}),
    html.Div([
        html.Label("物体 (質量既知として使用)"),
        dcc.Dropdown(obj_options, "pet_empty", id="cal-obj", clearable=False,
                     style={"width": "160px"}),
        dcc.Upload(html.Button("CSV を選択"), id="cal-upload"),
        html.Button("サンプルCSVで試す", id="cal-demo", n_clicks=0),
    ], style={"display": "flex", "gap": "12px", "alignItems": "center"}),
    dcc.Loading([html.Div(id="cal-report",
                          style={"marginTop": "8px", "whiteSpace": "pre-wrap",
                                 "fontFamily": "monospace"}),
                 dcc.Graph(id="cal-graph")]),
])

app.layout = html.Div([
    html.H3("box_sim — 自動捕球ロボット 設計シミュレータ"),
    dcc.Tabs([
        dcc.Tab(tab_a, label="A: 3Dシーン"),
        dcc.Tab(tab_b, label="B: 予測誤差"),
        dcc.Tab(tab_c, label="C: 設計探索 ★"),
        dcc.Tab(tab_d, label="D: 学習"),
        dcc.Tab(tab_cal, label="キャリブレーション"),
    ]),
], style={"margin": "12px", "fontFamily": "sans-serif"})


# ---------------- ビューA ----------------
@app.callback(
    Output("a-token", "data"), Output("a-time", "max"),
    Output("a-time", "value"), Output("a-summary", "children"),
    Input("a-run", "n_clicks"),
    State("a-obj", "value"), State("a-throw", "value"),
    State("a-seed", "value"), State("a-rand", "value"),
    prevent_initial_call=True)
def a_run(_n, obj_key, throw_key, seed, rand):
    world = WorldParams()
    rp = RobotParams()
    res = run_batch(OBJECTS[obj_key], THROWS[throw_key], rp, SensorParams(),
                    ControlParams(), world, n=1,
                    rng=np.random.default_rng(int(seed or 0)), record=True,
                    deterministic_throw=("rand" not in (rand or [])))
    token = str(uuid.uuid4())
    # 表示範囲: 着地+0.6s まで
    t_end = res.t_land[0] + 0.6 if not np.isnan(res.t_land[0]) else 2.0
    k_max = min(int(t_end / world.dt), len(res.times) - 1)
    _CACHE[token] = (res, rp, obj_key, k_max)
    _CACHE.pop("a-prev", None)
    catch_txt = "捕球成功" if res.catch[0] else \
        f"失敗 (miss {res.miss_dist[0]*100:.1f}cm)"
    dep_txt = f"展開 t={res.deploy_t0[0]:.2f}s" if res.deployed[0] else "展開なし"
    return token, k_max, 0, (
        f"{catch_txt} / {dep_txt} / 着地 t={res.t_land[0]:.3f}s / "
        f"スリップ率 {res.slip_frac[0]*100:.0f}%")


@app.callback(Output("a-scene", "figure"),
              Output("a-time", "value", allow_duplicate=True),
              Input("a-time", "value"), Input("a-interval", "n_intervals"),
              Input("a-token", "data"), State("a-speed", "value"),
              prevent_initial_call=True)
def a_figure(k_slider, _n, token, speed):
    """再生位置はサーバ側キャッシュで管理 (スライダはシーク兼表示)。"""
    from dash import ctx
    entry = _CACHE.get(token)
    if entry is None:
        return no_update, no_update
    res, rp, obj_key, k_max = entry
    trig = ctx.triggered_id
    k_cur = _CACHE.get(("k", token), 0)
    if trig == "a-time":
        if int(k_slider or 0) == k_cur:      # 自分の出力の跳ね返り
            return no_update, no_update
        k_cur = int(k_slider or 0)
    elif trig == "a-interval":
        k_cur = k_cur + int(speed or 12)
        if k_cur > k_max:
            k_cur = 0
    else:  # 新規実行
        k_cur = 0
    k_cur = min(max(k_cur, 0), k_max)
    _CACHE[("k", token)] = k_cur
    return scene_figure(res, rp, k_cur, obj_key), k_cur


@app.callback(Output("a-interval", "disabled"),
              Input("a-play", "n_clicks"), Input("a-pause", "n_clicks"))
def a_playpause(play, pause):
    from dash import ctx
    return ctx.triggered_id != "a-play"


# ---------------- ビューB ----------------
@app.callback(Output("b-graph", "figure"),
              Input("b-run", "n_clicks"),
              State("b-throw", "value"), State("b-seed", "value"),
              prevent_initial_call=True)
def b_run(_n, throw_key, seed):
    world = WorldParams()
    results = {}
    for key, obj in OBJECTS.items():
        res = run_batch(obj, THROWS[throw_key], RobotParams(), SensorParams(),
                        ControlParams(), world, n=1,
                        rng=np.random.default_rng(int(seed or 0)),
                        record=True, deterministic_throw=True)
        results[key] = res
    return error_figure(results)


# ---------------- ビューC ----------------
@app.callback(
    Output("c-mouth-val", "children"), Output("c-deploy-val", "children"),
    Output("c-amax-val", "children"), Output("c-latency-val", "children"),
    Output("c-lead-val", "children"),
    Input("c-mouth", "value"), Input("c-deploy", "value"),
    Input("c-amax", "value"), Input("c-latency", "value"),
    Input("c-lead", "value"))
def c_slider_labels(mouth, deploy, amax, lat, lead):
    """スライダの現在値をラベルに出す (ツールチップは隠れて読めないため)。"""
    return (f" {mouth*100:.0f} cm", f" {deploy*1000:.0f} ms",
            f" {amax:.1f} m/s²", f" {lat*1000:.0f} ms",
            f" {lead*1000:.0f} ms")


@app.callback(Output("c-heatmap", "figure"), Output("c-warn", "children"),
              Input("c-run", "n_clicks"),
              State("c-x", "value"), State("c-y", "value"),
              State("c-obj", "value"), State("c-throw", "value"),
              State("c-trials", "value"),
              State("c-mouth", "value"), State("c-deploy", "value"),
              State("c-amax", "value"), State("c-latency", "value"),
              State("c-lead", "value"),
              prevent_initial_call=True)
def c_run(_n, xk, yk, obj_key, throw_key, trials, s_mouth, s_deploy, s_amax,
          s_lat, s_lead):
    if xk == yk:
        return go.Figure(), "X軸とY軸に同じパラメータは選べません。"
    world = WorldParams()
    rp = RobotParams(mouth_deploy_radius=s_mouth, deploy_time=s_deploy,
                     a_max=s_amax)
    sp = SensorParams(latency=s_lat)
    cp = ControlParams(lead_time=s_lead or 0.0)
    warn = ""
    a_tip = rp.tipping_a_max()
    if s_amax > a_tip:
        warn = (f"注意: a_max {s_amax:.1f} m/s² は準静的転倒限界 "
                f"{a_tip:.1f} m/s² を超えるためクリップされます "
                "(重心高さ≤トレッド半分の制約)。")
    nx = ny = 6
    xs = np.linspace(PARAM_SPECS[xk]["lo"], PARAM_SPECS[xk]["hi"], nx)
    ys = np.linspace(PARAM_SPECS[yk]["lo"], PARAM_SPECS[yk]["hi"], ny)
    grid = mc_heatmap(OBJECTS[obj_key], THROWS[throw_key], rp, sp, cp, world,
                      xk, xs, yk, ys, n_trials=int(trials), seed=0)
    fig = go.Figure(go.Heatmap(
        x=xs, y=ys, z=grid * 100, zmin=0, zmax=100,
        colorscale="RdYlGn", colorbar=dict(title="捕球率 %"),
        text=np.round(grid * 100).astype(int), texttemplate="%{text}",
    ))
    err95 = 1.96 * np.sqrt(0.25 / int(trials)) * 100
    fig.update_layout(
        xaxis_title=PARAM_SPECS[xk]["label"],
        yaxis_title=PARAM_SPECS[yk]["label"],
        title=(f"捕球率 [%] — {OBJECTS[obj_key].label} / "
               f"{THROWS[throw_key].label} / {trials}試行/格子点 "
               f"(MC誤差 ≤±{err95:.1f}pt)"),
        height=560, margin=dict(l=60, r=20, t=50, b=50))
    return fig, warn


# ---------------- ビューD ----------------
@app.callback(Output("d-run", "options"), Output("d-run", "value"),
              Input("d-refresh", "n_clicks"))
def d_refresh(_n):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = sorted([p.name for p in RUNS_DIR.iterdir()
                   if (p / "train_log.json").exists()])
    if not runs:
        from learning.train_lstm import make_demo_run
        make_demo_run(str(RUNS_DIR / "demo"))
        runs = ["demo"]
    opts = [{"label": r, "value": r} for r in runs]
    return opts, runs[0]


@app.callback(Output("d-graph", "figure"), Input("d-run", "value"),
              prevent_initial_call=True)
def d_show(run):
    from plotly.subplots import make_subplots
    if not run:
        return go.Figure()
    d = RUNS_DIR / run
    log = json.loads((d / "train_log.json").read_text(encoding="utf-8"))
    fig = make_subplots(cols=2, rows=1,
                        subplot_titles=("損失曲線 (MSE)",
                                        "予測初速 vs 実測初速 [m/s]"))
    ep = list(range(len(log["train_loss"])))
    fig.add_trace(go.Scatter(x=ep, y=log["train_loss"], name="train"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=log["val_loss"], name="val"),
                  row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    pred_f = d / "val_predictions.csv"
    if pred_f.exists():
        arr = np.loadtxt(pred_f, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None]
        names = ["vx", "vy", "vz"]
        for i, nm in enumerate(names):
            fig.add_trace(go.Scatter(x=arr[:, i], y=arr[:, 3 + i],
                                     mode="markers", name=nm), row=1, col=2)
        lim = [arr[:, 0:3].min() - 0.5, arr[:, 0:3].max() + 0.5]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="y=x",
                                 line=dict(color="#999", dash="dash")),
                      row=1, col=2)
        err = np.linalg.norm(arr[:, 0:3] - arr[:, 3:6], axis=1)
        fig.add_annotation(text=f"|Δv| mean = {err.mean():.3f} m/s",
                           xref="x2 domain", yref="y2 domain", x=0.05, y=0.95,
                           showarrow=False)
    title = "学習ラン: " + run
    if log.get("demo"):
        title += " 【DEMO データ — GUI配線確認用。実測ではありません】"
    fig.update_layout(title=title, height=520,
                      margin=dict(l=50, r=20, t=70, b=40))
    return fig


# ---------------- キャリブレーション ----------------
@app.callback(Output("cal-report", "children"), Output("cal-graph", "figure"),
              Input("cal-upload", "contents"), Input("cal-demo", "n_clicks"),
              State("cal-obj", "value"), prevent_initial_call=True)
def cal_run(contents, _n, obj_key):
    from dash import ctx
    import base64
    obj = OBJECTS[obj_key]
    if ctx.triggered_id == "cal-demo":
        csv_text = make_sample_csv(obj)
        src = "サンプルCSV (真値 CdA=既定×1.2, e=既定×1.3 で生成)"
    else:
        if not contents:
            return no_update, no_update
        csv_text = base64.b64decode(contents.split(",", 1)[1]).decode("utf-8")
        src = "アップロードCSV"
    try:
        t, xyz = load_trajectory_csv(csv_text)
        res = calibrate(t, xyz, obj)
    except Exception as e:
        return f"エラー: {e}", go.Figure()
    (lo_c, hi_c), (lo_e, hi_e) = res.ci95()
    report = (
        f"入力: {src}  ({len(t)}点, {t[-1]:.2f}s)\n"
        f"CdA        : {res.cda:.5f} m²   95%CI [{lo_c:.5f}, {hi_c:.5f}]"
        f"   (カタログ値 {obj.cda:.5f})\n"
        f"反発係数 e : {res.restitution:.3f}      95%CI [{lo_e:.3f}, {hi_e:.3f}]"
        f"   (カタログ値 {obj.restitution:.3f})\n"
        f"残差RMS    : 同定前 {res.rms_before*100:.2f} cm → 同定後 "
        f"{res.rms_after*100:.2f} cm\n"
        f"シミュレータ信頼区間: 着地点予測に ±{res.rms_after*100*2:.1f} cm 程度の"
        f"モデル誤差を見込むこと (残差RMSの2倍を目安)。\n"
        + (f"注意: {res.note}" if res.note else ""))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.t, y=res.residuals_before * 100,
                             name="同定前残差", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=res.t, y=res.residuals_after * 100,
                             name="同定後残差", mode="lines+markers"))
    fig.update_layout(xaxis_title="t [s]", yaxis_title="位置残差 [cm]",
                      height=380, margin=dict(l=50, r=20, t=30, b=40))
    return report, fig


if __name__ == "__main__":
    app.run(debug=False, port=8050)
