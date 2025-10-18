# Pneumonia-Detection-Web-App
**A Streamlit-based demo that detects pneumonia on chest X-rays using a fine-tuned Faster R-CNN and visualizes model attention via Grad-CAM++.

## Download Dataset

You can download the datasets here:

- **PNG images (450 MB)**:  
  https://uni-bonn.sciebo.de/s/ey8iHK6HxsXGcLW/download/pneumonia-png.zip  

- **Alternate JPG images (69.8 MB)**:  
   https://uni-bonn.sciebo.de/s/kJfcEiroW7tZ6Qo/download/pneumonia-jpg.zip

- **DICOM images (144 MB)**:  
  https://uni-bonn.sciebo.de/s/QX2S6T3EykCsFNG/download/pneumonia-dicom.zip  

After downloading, unzip into a folder, and point the **Use local DICOM folder** path in the app sidebar to that directory.

- **Model**:  
  https://uni-bonn.sciebo.de/s/sgLspCcjMtrppFa/download/fasterrcnn_resnet50_fpn.pth

# Pneumonia Detection Web App

## Quick Start

1. **Clone & enter directory**  
   ```bash
   git clone https://github.com/your-repo/pneumonia-app.git
   cd pneumonia-app/pneumonia_app
   ```

2. **Create Conda env & install**  
   ```bash
   conda create -n pneumonia-app
   conda activate pneumonia-app
   conda install pip
   pip install -r requirements.txt
   ```

3. **Run the app**  
   ```bash
   streamlit run app.py
   ```

4. **Interact**  
   - Drag-and-drop a DICOM, PNG or JPEG  
   - Toggle **Use local DICOM folder** to browse a folder of `.dcm` files  
   - Adjust the **Score threshold** slider  
   - Click **Run inference** to see bounding boxes, Grad-CAM heat-map and overlay  
   - Open **Show model metrics** in the sidebar to view Precision/Recall/AUC and ROC curve

## Project Structure

```
pneumonia_app/
├── app.py            # Streamlit UI glue
├── config.py         # Constants: paths, image size, labels
├── model.py          # Builds & loads the Faster R-CNN model
├── io_utils.py       # Ingests DICOM / JPEG / PNG → RGB tensor
├── dicom_utils.py    # Local-folder (mini-PACS) helpers
├── vis_utils.py      # Box drawing & Grad-CAM logic
└── requirements.txt  # Python dependencies
```

## Customization

- **Swap model**: replace `CHECKPOINT_PATH` in `config.py` and drop in your own `.pth`.  
- **Add classes**: change `NUM_CLASSES` and update `LABEL_MAP`.  
- **Explainability**: swap `GradCAMPlusPlus` for other CAM methods in `vis_utils.py`.

---

Made for clinicians & developers to prototype explainable medical AI in minutes. Enjoy!**
