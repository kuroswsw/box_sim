"""物体分類 (YOLOv8n / MobileNetV3) → 質量・Cd の事前分布を選ぶ。

役割はそれだけ。分類結果は EKF の beta 初期値と初期共分散 P[6,6] に
入るだけで、軌道予測そのものはあくまで物理モデル+EKF が行う。

推論は ONNX Runtime。ここではシミュレータ側インタフェースと、
誤分類が捕球率に与える影響を評価するためのラッパを提供する。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim.params import ObjectParams, make_objects

CLASSES = ["pet_empty", "pet_half", "pet_full", "trash_can",
           "trash_paper", "tissue"]


class OnnxClassifier:
    """ONNX Runtime 推論ラッパ (モデルファイルがある場合のみ)。"""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def predict(self, image_chw: np.ndarray) -> tuple[str, np.ndarray]:
        logits = self.sess.run(None, {self.input_name: image_chw[None]})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return CLASSES[int(p.argmax())], p


def prior_from_class(cls_name: str, confusion_p: np.ndarray | None = None
                     ) -> ObjectParams:
    """分類結果 → EKF に渡す事前分布付き ObjectParams。

    confusion_p を渡すと、クラス確率で重み付けした beta 事前分布
    (混合分布の平均と標準偏差) を設定する。誤分類の影響評価用。
    """
    catalog = make_objects()
    obj = catalog[cls_name]
    if confusion_p is None:
        return obj
    betas = np.array([catalog[c].beta for c in CLASSES])
    mean = float(np.sum(confusion_p * betas))
    var = float(np.sum(confusion_p * (betas - mean) ** 2))
    # クラス内ばらつきも足す
    var += float(np.sum(confusion_p *
                        np.array([catalog[c].beta_prior_std ** 2
                                  for c in CLASSES])))
    return replace(obj, beta_prior_mean=mean, beta_prior_std=float(np.sqrt(var)))
