import numpy as np, torch, matplotlib.pyplot as plt, matplotlib.patches as patches, streamlit as st
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import LABEL_MAP

class BoxScoreTarget:
    def __init__(self, idx): self.idx = idx
    def __call__(self, out):  return out["scores"][self.idx]

def draw_boxes(img_rgb, detections, thresh):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img_rgb)
    for box,score,lbl in zip(*[detections[k] for k in ("boxes","scores","labels")]):
        if score<thresh: continue
        x1,y1,x2,y2 = box.cpu()
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                     linewidth=2,edgecolor='r',facecolor='none'))
        ax.text(x1, y1-5, f"{LABEL_MAP.get(int(lbl),lbl)}:{score:.2f}",
                color="white", fontsize=18,
                bbox=dict(facecolor="red",edgecolor="none",pad=1))
    ax.axis("off"); st.pyplot(fig, clear_figure=True)

def gradcam_overlay(tensor, img_rgb, model, detections, image_weight=0.5):
    target_layer = model.backbone.body.layer4[-1]
    cam = GradCAMPlusPlus(model, [target_layer])
    idxs = (detections["labels"]==1).nonzero(as_tuple=True)[0]
    target = [BoxScoreTarget(int(idxs[torch.argmax(detections['scores'][idxs])]))] if idxs.numel() else None
    heat = cam(tensor.unsqueeze(0), targets=target)[0]
    return show_cam_on_image(img_rgb/255.0, heat, use_rgb=True, image_weight=image_weight)