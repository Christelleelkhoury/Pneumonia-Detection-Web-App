import streamlit as st, torch, numpy as np, pandas as pd
from PIL import Image
import model, utils.io_utils as io_utils, utils.vis_utils as vis_utils, config
from utils.dicom_utils import list_dicom_files, filter_by_patient_id, read_dicom_raw, read_dicom_metadata
from utils.utils import load_config
# App UI
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("Pneumonia Detection")
st.markdown("(Faster R-CNN + Grad-CAM)")
st.caption("Upload or select a chest X-ray to detect pneumonia, view bounding-box predictions, and explore Grad-CAM heatmaps for model interpretability.")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running inference on: **{device}**")
score_th = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

# Our variables
raw = None
name = None
model_missing=None

with st.sidebar:
    model_path = st.text_input("Checkpoint path", "fasterrcnn_resnet50_fpn.pth")
    model_name = model_path.removesuffix(".pth")
    ckpt = config.CHECKPOINT_PATH / model_path

    try:
        dectector_model = model.build_model(model_path, device)
        json_config = load_config(config.CHECKPOINT_PATH/ f"{model_name}.json")
        ROC_CURVE_PATH    = json_config["roc_curve_path"]
        metrics_list      = json_config["metrics"]
        if st.button("Show model metrics"):
            # compute trainable params
            num_params = sum(p.numel() for p in dectector_model.parameters() if p.requires_grad)

            # Display ROC curve
            st.sidebar.image(ROC_CURVE_PATH, caption="ROC Curve", use_container_width=True)

            try:
                # build DataFrame, filling in the trainable‐params cell
                metrics_list[-1]["Value"] = f"{num_params:,}"
                df = pd.DataFrame(metrics_list)

                st.sidebar.data_editor(
                    df,
                    hide_index=True,
                    num_rows="fixed",
                )
            except:
                pass
    except:
        model_missing = True

if model_missing:
    st.warning("Please add a checkpoint PATH from your model in the sidebar")
    st.stop()

# Local DICOM Database Logic
use_local = st.checkbox("Use local DICOM folder", value=False)
if use_local:
    data_dir = st.text_input("Path to DICOM folder", "./inference_data/dcm")
    all_paths = list_dicom_files(data_dir)
    if not all_paths:
        st.error(f"No .dcm files in {data_dir}")
        st.stop()

    query = st.text_input("🔍 Search Patient ID", "")
    dicom_paths = (
        filter_by_patient_id(all_paths, query) 
        if query else all_paths
    )
    if not dicom_paths:
        st.warning("No files matched your search.")
        st.stop()

    selected_path = st.selectbox("Choose a DICOM file", dicom_paths)
    raw, name = read_dicom_raw(selected_path)

    md = read_dicom_metadata(selected_path)
    st.markdown("#### Selected File Metadata")
    for k, v in md.items():
        st.markdown(f"- **{k}:** {v or '—'}")
else:
    uploaded = st.file_uploader("Upload DICOM / PNG / JPG",
                            type=["dcm","png","jpg","jpeg"])
    if not uploaded:
        st.info("Please upload a file or enable local folder mode.")
        st.stop()

    raw = uploaded.read()
    name = uploaded.name

# Inference Logic
if st.button("Run inference") and raw:
    rgb = io_utils.raw_bytes_to_rgb(raw, name)
    rgb_resized = np.array(Image.fromarray(rgb).resize(config.IMAGE_SIZE))
    tensor = io_utils.preprocess(rgb).to(device)

    with torch.no_grad():
        det = dectector_model([tensor])[0]

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Detections")
        vis_utils.draw_boxes(rgb_resized, det, score_th)
    with col2:
        st.subheader("Grad-CAM++")
        cam_img = vis_utils.gradcam_overlay(tensor, rgb_resized, dectector_model, det)
        st.image(cam_img, use_container_width=True)
    
    st.subheader("Detections + Heatmaps")
    cam_img = vis_utils.gradcam_overlay(tensor, rgb_resized, dectector_model, det, image_weight=0.8)
    vis_utils.draw_boxes(cam_img, det, score_th)