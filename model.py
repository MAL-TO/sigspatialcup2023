from typing import Any, Callable, Optional
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import Metric
from torchvision.datasets import VisionDataset
import torch
import os
import pandas as pd
from pathlib import Path
import geopandas as gp
import numpy as np
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomRotation, GaussianBlur, ElasticTransform
import rasterio
import utils
from lightning.pytorch import Trainer
import argparse
from torch.utils.data import DataLoader
import cv2
from unified_focal_loss import AsymmetricUnifiedFocalLoss

BASE_DIR = utils.BASE_DIR
BATCH_SIZE = 80
NUM_WORKERS = 16
torch.set_float32_matmul_precision('medium')

# the label dataset provided should be the one built using get_labels_datasets function in utils.py
# img_dir can alternatively be '/data1/malto/train' or '/data1/malto/test'

class SigspatialDataset(VisionDataset):
    def __init__(self, train=True, img_dir: Path=BASE_DIR, multi_class=True, aug=False):
        self.img_dir = img_dir
        self.aug = aug
        self.multi_class = multi_class
        self.kind = "train" if train else "val"
        self.names = sorted([name for name in os.listdir(img_dir / f"ds_{self.kind}_images") if name.split(".")[-1] == "tif"])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = self.img_dir / f"ds_{self.kind}_images" / self.names[idx]
        lbl_path = self.img_dir / f"ds_{self.kind}_label" / self.names[idx]
        img = rasterio.open(img_path)
        lbl = rasterio.open(lbl_path)
        img_array = img.read()
        lbl_array = lbl.read()
        
        # transforms
        img_array = np.transpose(img_array, ((1, 2, 0)))
        lbl_array = np.transpose(lbl_array, ((1, 2, 0)))

        img_array = cv2.resize(img_array, dsize=(512, 512))
        lbl_array = cv2.resize(lbl_array, dsize=(512, 512))

        if self.kind == "train" and self.aug == True:
            glob_array = np.vstack((img_array, lbl_array))
            transform = Compose([
                RandomHorizontalFlip(),
                RandomRotation(degrees=(0, 20, 30, 60, 80, 90, 100, 105, 150, 180, 200, 270, 345, 350)),
            ])
            glob_array = transform(glob_array)
            img_array, lbl_array = np.vsplit(glob_array, 2)
        
        img_tensor = ToTensor()(img_array)
        lbl_tensor = ToTensor()(lbl_array)

        lbl_tensor = lbl_tensor.mean(dim=0).unsqueeze(dim=0)

        img_tensor = img_tensor.type(torch.float32)
        lbl_tensor = lbl_tensor.type(torch.float32)

        img_tensor = img_tensor * 2 - 1

        if self.multi_class:
            fg = lbl_tensor
            bg = torch.where(fg > 0.5, 0, 1)
            lbl_tensor = torch.cat((bg, fg), dim=0)

        return img_tensor, lbl_tensor


class IoU(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        preds = preds >= 0.5
        self.intersection += preds.logical_and(target).sum()
        self.union += preds.logical_or(target).sum()

    def compute(self):
        return self.intersection.float() / self.union


class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-4,  multi_class=True):
        super().__init__()
        self.segmentation_model = smp.DeepLabV3Plus(encoder_name="resnet18", activation='softmax', classes=2)
        self.loss = AsymmetricUnifiedFocalLoss()
        self.learning_rate = lr
        self.train_iou = IoU()
        self.val_iou = IoU()
        self.multi_class = multi_class

    def forward(self, x):
        return self.segmentation_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        if self.multi_class:
            self.train_iou.update(torch.argmax(logits, dim=1), torch.argmax(y, dim=1))
        else:
            self.train_iou.update(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        if self.multi_class:
            self.val_iou.update(torch.argmax(logits, dim=1), torch.argmax(y, dim=1))
        else:
            self.val_iou.update(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=self.learning_rate)
    

class SatelliteDataModule(pl.LightningDataModule):
    def __init__(self, multi_class=True, aug=False):
        super().__init__()
        self.multi_class = multi_class
        self.aug = aug

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_ds = SigspatialDataset(train=True, multi_class=self.multi_class)
        self.val_ds = SigspatialDataset(train=False, multi_class=self.multi_class)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, drop_last=True, num_workers=NUM_WORKERS, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=2*BATCH_SIZE, drop_last=True, num_workers=NUM_WORKERS)

def allena_modello(lr, epochs):
    model = SegmentationModel(lr=lr, multi_class=True)
    data = SatelliteDataModule(multi_class=True,aug=True)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        logger=CSVLogger(save_dir=BASE_DIR / "logs"),
        log_every_n_steps=1
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate")
    parser.add_argument("-e", "--epochs")
    args = parser.parse_args()

    allena_modello(float(args.learning_rate), int(args.epochs))
    
