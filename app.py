# app.py
import os
import sys
import time
import traceback

import torch
import streamlit as st
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Local imports (kept light/safe)
try:
    import config
    from utils import vis_utils
except Exception as e:
    st.error(f"Failed to import local modules: {e}")
    st.stop()

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="Week 7 – Detector + Grad-CAM", layout="wide")

st.title("📷 Object Detection + Grad-CAM (Week 7)")
st.caption("Robust imports; Grad-CAM overlay without OpenCV.")

# ---------------- Utilities ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Try common patterns to obtain the model from your local 'model' module.
    Adjust this if your repo exposes a different entrypoint.
    """
    try:
        import model as model_mod
    except Exception as e:
        raise RuntimeError(
            "Could not import your local 'model' module. "
            "Confirm it's on PYTHONPATH and imports cleanly.\n\n"
            f"Import error: {e}"
        ) from e

    # Try common factory names in order
    factories = [
        getattr(model_mod, "get_model", None),
        getattr(model_mod, "build_model", None),
        getattr(model_mod, "load_model", None),
    ]
    for fn in factories:
        if callable(fn):
            m = fn()
            return m

    # Fallback: a directly exposed attribute
    for attr in ("model", "MODEL", "net"):
        if hasattr(model_mod, attr):
            return getattr(model_mod, attr)

    raise RuntimeError(
        "Could not find a model factory in 'model'. "
        "Expose one of: get_model(), build_model(), load_model(), or a 'model' variable."
    )

def to_device(m):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(device), device

def preprocess_pil(img_pil):
    """
    Convert PIL -> normalized CHW tensor for a torchvision-style detector.
    """
    t = T.Compose([
        T.ToTensor(),  # [0,1], CHW
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return t(img_pil)

def run_inference(model, device, tensor):
    model.eval()
    with torch.no_grad():
        # torchvision detectors expect a list of tensors
        preds = model([tensor.to(device)])
    if isinstance(preds, (list, tuple)) and len(preds):
        return preds[0]
    # If your model returns a dict directly, just return it
    if isinstance(preds, dict):
        return preds
    raise RuntimeError("Unexpected model output format. Expected list[dict] with keys boxes/scores/labels.")

def to_numpy_rgb(img_pil):
    arr = np.array(img_pil.convert("RGB"))
    return arr

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Options")
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    show_gradcam = st.checkbox("Show Grad-CAM overlay", value=True)
    gradcam_mix = st.slider("Grad-CAM image weight", 0.0, 1.0, 0.5, 0.05)
    st.divider()
    st.caption("If Grad-CAM fails to import, you'll see a friendly error.")

# ---------------- Main UI ----------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

# Load image
img_pil = Image.open(uploaded).convert("RGB")
img_rgb = to_numpy_rgb(img_pil)
tensor = preprocess_pil(img_pil)  # CHW normalized

# Load model (cached)
try:
    model = load_model()
    model, device = to_device(model)
except Exception as e:
    st.error(f"Model load failed:\n\n{e}")
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exc()))
    st.stop()

# Inference
try:
    detections = run_inference(model, device, tensor)
    # Ensure keys exist
    for k in ("boxes", "scores", "labels"):
        if k not in detections:
            raise KeyError(f"Missing '{k}' in model detections.")
except Exception as e:
    st.error(f"Inference failed:\n\n{e}")
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exc()))
    st.stop()

# Columns: image + overlay
col1, col2 = st.columns(2)

with col1:
    st.subheader("Detections")
    try:
        vis_utils.draw_boxes(img_rgb, detections, conf_thresh)
    except Exception as e:
        st.error(f"Failed to draw boxes:\n\n{e}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exc()))

with col2:
    st.subheader("Grad-CAM Overlay")
    if show_gradcam:
        try:
            blended = vis_utils.gradcam_overlay(
                tensor=tensor.cpu(),          # CHW normalized
                img_rgb=img_rgb,              # HxWx3 uint8
                model=model,                  # detector with .backbone.body.layer4
                detections=detections,
                image_weight=gradcam_mix,
            )
            st.image(blended, caption="Grad-CAM (no OpenCV)", use_container_width=True)
        except RuntimeError as re:
            # Friendly message from our lazy import (e.g., pytorch-grad-cam not installed)
            st.warning(str(re))
        except Exception as e:
            st.error(f"Grad-CAM failed:\n\n{e}")
            with st.expander("Traceback"):
                st.code("".join(traceback.format_exc()))
    else:
        st.info("Enable Grad-CAM in the sidebar to view the overlay.")

st.success("Done.")
st.caption("Tip: If you see import errors related to OpenCV, use the headless wheel: "
           "`pip install \"opencv-python-headless<5\"`")
