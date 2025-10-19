import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import NUM_CLASSES, CHECKPOINT_PATH

def build_model(model_name, device="cpu"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)
    state = torch.load(CHECKPOINT_PATH/model_name, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()