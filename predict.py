# import os
# import numpy as np
import torch
# import torchvision
# import matplotlib.pyplot as plt
# from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = "cuda" if torch.cuda.is_available() else "cpu"

NAME_MODEL = "sam2_hiera_tiny"
# NAME_MODEL = "sam2_hiera_small"
# NAME_MODEL = "sam2_hiera_base_plus"
# NAME_MODEL = "sam2_hiera_large"
DICT_CONFIG = {
    "sam2_hiera_tiny": "sam2_hiera_t",
    "sam2_hiera_small": "sam2_hiera_s",
    "sam2_hiera_base_plus": "sam2_hiera_b_plus",
    "sam2_hiera_large": "sam2_hiera_l",
}
PATH_WEIGHT = f"/data/weights/{NAME_MODEL}.pt"
# sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(DICT_CONFIG[NAME_MODEL], PATH_WEIGHT, device=device)

predictor = SAM2ImagePredictor(sam2_model)

print(predictor.__dict__)
