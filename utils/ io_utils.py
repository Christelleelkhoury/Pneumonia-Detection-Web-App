import io, numpy as np, pydicom
from PIL import Image
import torchvision.transforms.functional as F
import streamlit as st
from config import IMAGE_SIZE

def raw_bytes_to_rgb(raw, filename) -> np.ndarray:
    buf = io.BytesIO(raw)
    buf.name = filename
    return file_to_rgb(buf)

def file_to_rgb(uploaded_file) -> np.ndarray:
    raw  = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".dcm"):
        return dicom_to_rgb(raw)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)

def dicom_to_rgb(buf: bytes) -> np.ndarray:
    ds   = pydicom.dcmread(io.BytesIO(buf))
    arr  = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, (1,99))
    arr  = np.clip((arr-lo)/(hi-lo+1e-6), 0, 1)
    rgb  = (arr*255).astype(np.uint8)
    return np.stack([rgb]*3, axis=-1)

def preprocess(rgb: np.ndarray):
    img = Image.fromarray(rgb)
    img = F.resize(img, IMAGE_SIZE)
    return F.pil_to_tensor(img).float() / 255.0
