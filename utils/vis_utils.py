import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from config import LABEL_MAP

# ---------- Helpers (safe/lazy imports and utilities) ----------

def _require_grad_cam():
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        return GradCAMPlusPlus
    except Exception as e:
        raise RuntimeError(
            "pytorch-grad-cam is required. Install with:\n"
            "  pip install pytorch-grad-cam"
        ) from e

def _blend_cam(img_rgb: np.ndarray, heatmap: np.ndarray, image_weight: float = 0.5) -> np.ndarray:
    """
    Pure NumPy overlay (no OpenCV). img_rgb: HxWx3 uint8 or float in [0,255]/[0,1]
    heatmap: HxW float (unnormalized CAM). Returns HxWx3 float in [0,1].
    """
    # ensure float copy of image in [0,1]
    img = img_rgb.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # normalize heatmap to [0,1]
    h = heatmap.astype(np.float32)
    h = h - h.min()
    h = h / (h.max() + 1e-6)

    # colorize with matplotlib colormap (returns RGBA in [0,1])
    cmap = plt.get_cmap("jet")
    h_color = cmap(h)[..., :3]  # drop alpha

    # blend
    image_weight = float(np.clip(image_weight, 0.0, 1.0))
    blended = (1.0 - image_weight) * h_color + image_weight * img

    # clip to [0,1]
    return np.clip(blended, 0.0, 1.0)

# ---------- Core API ----------

class BoxScoreTarget:
    def __init__(self, idx): 
        self.idx = idx
    def __call__(self, out):  # out is model output dict for detector models
        return out["scores"][self.idx]

def draw_boxes(img_rgb, detections, thresh):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb)
    for box, score, lbl in zip(*[detections[k] for k in ("boxes", "scores", "labels")]):
        if score < thresh:
            continue
        x1, y1, x2, y2 = box.cpu()
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=2, edgecolor='r', facecolor='none'))
        ax.text(x1, y1 - 5, f"{LABEL_MAP.get(int(lbl), lbl)}:{score:.2f}",
                color="white", fontsize=18,
                bbox=dict(facecolor="red", edgecolor="none", pad=1))
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

def gradcam_overlay(tensor, img_rgb, model, detections, image_weight=0.5):
    """
    tensor: CHW torch.FloatTensor (preprocessed input) for the model
    img_rgb: HxWx3 uint8/float RGB image to overlay on
    model: torchvision-style detector with .backbone.body.layer4
    detections: dict with 'boxes','scores','labels' (1-based classes)
    image_weight: how much of the original image to keep vs heatmap [0..1]
    """
    # pick the last block of layer4 for a ResNet backbone
    target_layer = model.backbone.body.layer4[-1]

    GradCAMPlusPlus = _require_grad_cam()
    # Use context manager so hooks are cleaned up
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        # pick the top-scoring label==1 (if present) to target
        idxs = (detections.get("labels", torch.tensor([])) == 1).nonzero(as_tuple=True)[0]
        if idxs.numel():
            chosen = int(idxs[torch.argmax(detections['scores'][idxs])])
            targets = [BoxScoreTarget(chosen)]
        else:
            targets = None  # fallback to class-agnostic cam

        # CAM returns [N, H, W]; here N=1
        heat = cam(tensor.unsqueeze(0), targets=targets)[0]

    # Make an overlay without cv2
    blended = _blend_cam(img_rgb, heat, image_weight=image_weight)  # HxWx3 in [0,1]
    return blended

