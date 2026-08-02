"""投擲の事前予測 LSTM: MediaPipe Pose 骨格系列 → リリース初速ベクトル回帰。

重要な原則 (仕様):
- 学習データは実測のみ。シミュレータで生成した骨格データで学習しない
  (シミュには骨格モデルがないので物理的にも不可能)。
- シミュレータ側では、この LSTM の効果を「リード時間 lead_time」と
  「予測誤差 lstm_err_std」の2パラメータに縮約して感度解析にのみ使う。
  学習済みモデルの精度 (val 損失から求まる誤差std) を GUI に入力すれば、
  「事前予測が何ms早いと捕球率が何%上がるか」が読める。

データ形式 (実測CSV):
  data/pose/<throw_id>.csv : 各行 = 1フレーム,
    列: t, x0,y0,z0, x1,y1,z1, ... (MediaPipe Pose 33点 × 3座標)
  data/pose/labels.csv : throw_id, vx, vy, vz (モーキャプ等による実測初速)

学習: PyTorch (LSTM 2層 hidden=64 → FC)。推論: ONNX Runtime 用に export。

使い方:
  python learning/train_lstm.py --data data/pose --out learning/runs/run1
  python learning/train_lstm.py --demo --out learning/runs/demo   # GUI確認用

--demo は GUI (ビューD) の動作確認専用のダミーデータ生成。
実測データが無い段階でビューDの配線を確認するためのもので、
学習性能の評価には一切使えない (出力に DEMO と明記される)。
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

SEQ_LEN = 30       # リリース前 30 フレーム (60fps で 0.5s)
N_KEYPOINTS = 33
IN_DIM = N_KEYPOINTS * 3


def load_dataset(data_dir: str):
    d = pathlib.Path(data_dir)
    labels = {}
    with open(d / "labels.csv", encoding="utf-8") as f:
        for line in f.read().strip().splitlines()[1:]:
            tid, vx, vy, vz = line.split(",")
            labels[tid] = [float(vx), float(vy), float(vz)]
    xs, ys = [], []
    for tid, v in labels.items():
        arr = np.loadtxt(d / f"{tid}.csv", delimiter=",", skiprows=1)
        seq = arr[-SEQ_LEN:, 1:1 + IN_DIM]
        if seq.shape != (SEQ_LEN, IN_DIM):
            continue
        xs.append(seq)
        ys.append(v)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def train(x: np.ndarray, y: np.ndarray, out_dir: str,
          epochs: int = 200, lr: float = 1e-3, demo: bool = False) -> None:
    import torch
    import torch.nn as nn

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(x)
    idx = np.random.default_rng(0).permutation(n)
    n_val = max(1, n // 5)
    tr, va = idx[n_val:], idx[:n_val]
    xt = torch.tensor(x)
    yt = torch.tensor(y)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(IN_DIM, 64, num_layers=2, batch_first=True)
            self.fc = nn.Linear(64, 3)

        def forward(self, s):
            o, _ = self.lstm(s)
            return self.fc(o[:, -1])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    log = {"demo": demo, "train_loss": [], "val_loss": []}
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(xt[tr]), yt[tr])
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xt[va]), yt[va]).item()
        log["train_loss"].append(float(loss.item()))
        log["val_loss"].append(vl)

    with torch.no_grad():
        pred_va = net(xt[va]).numpy()
    _write_outputs(out, log, y[va], pred_va)

    # ONNX export (推論は ONNX Runtime)
    try:
        torch.onnx.export(net, xt[:1], str(out / "model.onnx"),
                          input_names=["pose_seq"], output_names=["v0"],
                          dynamic_axes={"pose_seq": {0: "batch"}})
        print(f"ONNX exported: {out/'model.onnx'}")
    except Exception as e:  # onnx 未インストール等
        print(f"ONNX export skipped: {e}")


def _write_outputs(out: pathlib.Path, log: dict,
                   y_true: np.ndarray, y_pred: np.ndarray) -> None:
    with open(out / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f)
    lines = ["vx_true,vy_true,vz_true,vx_pred,vy_pred,vz_pred"]
    for yt_, yp in zip(y_true, y_pred):
        lines.append(",".join(f"{v:.4f}" for v in [*yt_, *yp]))
    (out / "val_predictions.csv").write_text("\n".join(lines),
                                             encoding="utf-8")
    err = np.linalg.norm(y_true - y_pred, axis=1)
    print(f"val 初速ベクトル誤差: mean={err.mean():.3f} m/s, "
          f"p90={np.percentile(err, 90):.3f} m/s")
    print(f"→ シミュ側 lstm_err_std の目安 = {err.mean() * 0.25:.3f} m "
          "(誤差[m/s] x 飛行時間感度 ~0.25s/(m/s) の粗い換算)")


def make_demo_run(out_dir: str) -> None:
    """torch 無し環境でも GUI 配線を確認できるダミー出力 (DEMO明記)。"""
    rng = np.random.default_rng(1)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ep = np.arange(200)
    train_loss = (2.5 * np.exp(-ep / 40) + 0.05
                  + 0.02 * rng.standard_normal(200)).clip(0.02, None)
    val_loss = (2.5 * np.exp(-ep / 45) + 0.12
                + 0.05 * rng.standard_normal(200)).clip(0.05, None)
    log = {"demo": True, "train_loss": train_loss.tolist(),
           "val_loss": val_loss.tolist()}
    y_true = rng.normal([3.5, 0.0, 3.0], [0.8, 0.4, 0.6], size=(40, 3))
    y_pred = y_true + rng.standard_normal((40, 3)) * 0.35
    _write_outputs(out, log, y_true.astype(np.float32),
                   y_pred.astype(np.float32))
    print(f"DEMO run written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pose")
    ap.add_argument("--out", default="learning/runs/latest")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--demo", action="store_true",
                    help="GUI確認用ダミー出力 (学習しない)")
    args = ap.parse_args()
    if args.demo:
        make_demo_run(args.out)
        return
    x, y = load_dataset(args.data)
    print(f"dataset: {x.shape[0]} throws")
    train(x, y, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
