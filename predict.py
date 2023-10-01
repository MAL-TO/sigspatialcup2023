import pytorch_lightning as pl
from model import SegmentationModel, SigspatialDataset
from pathlib import Path
import os
import torch
import rasterio
import numpy as np
from torchvision.transforms import ToTensor
import cv2
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = utils.BASE_DIR
LOGS_DIR = BASE_DIR / "logs"

def generate_predictions():
    if not os.path.exists(BASE_DIR / "predictions"):
        os.mkdir(BASE_DIR / "predictions")

    last_version = max([int(s.replace("version_", "")) for s in os.listdir(f"{LOGS_DIR}/lightning_logs")])
    checkpoint_folder = Path(f"{LOGS_DIR}/lightning_logs/version_{last_version}/checkpoints")
    filenames = [name for name in os.listdir(checkpoint_folder) if name.split(".")[-1] == "ckpt"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegmentationModel.load_from_checkpoint(checkpoint_folder / filenames[0])
    model.eval()

    test_images = sorted([name for name in os.listdir(BASE_DIR / "test_image") if name.split(".")[-1] == "tif"])
    for test_image in tqdm(test_images):
        img_path = BASE_DIR / "test_image" / test_image
        img = rasterio.open(img_path)
        img_array = img.read()
        img_array = np.transpose(img_array, ((1, 2, 0)))
        img_array = cv2.resize(img_array, dsize=(512, 512))
        img_tensor = ToTensor()(img_array)
        img_tensor = img_tensor.type(torch.float32)
        img_tensor = img_tensor * 2 - 1
        
        x = img_tensor.to(device=device).unsqueeze(dim=0)
        pred = model(x)
        pred_tensor = torch.argmax(pred.squeeze(dim=0), dim=0)
        pred_array = pred_tensor.to(device='cpu').squeeze(dim=0).detach().numpy()
        pred_array = 255 * pred_array
        pred_array = pred_array.reshape(1, 512, 512)
        pred_img = np.vstack(3 * [pred_array])

        out_meta = img.meta.copy()

        # Save the clipped image to a new .tif file
        with rasterio.open(BASE_DIR / "predictions" / test_image, "w", **out_meta) as dest:
            dest.write(pred_img)

if __name__ == "__main__":
    generate_predictions()