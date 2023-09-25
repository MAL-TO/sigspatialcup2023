from typing import Any, Callable, Optional
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from pytorch_lightning.loggers import CSVLogger

from torchmetrics import Metric
from torchvision.datasets import VisionDataset, Dataset
import torch

import os
import pandas as pd
from pathlib import Path
import geopandas as gp
import numpy as np
from torchvision.transforms import ToTensor
import rasterio

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

BATCH_SIZE = 32
NUM_WORKERS = 16
torch.set_float32_matmul_precision('medium')

# the label dataset provided should be the one built using get_labels_datasets function in utils.py
# img_dir can alternatively be '/data1/malto/train' or '/data1/malto/test'

class SigspatialDataset(Dataset):
    def __init__(self, img_dir: Path, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.names = sorted([name for name in os.listdir(img_dir / "image") if name.split(".")[-1] == "tif"])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir / "image" / self.names[idx]
        lbl_path = self.img_dir / "label" / self.names[idx]
        img = rasterio.open(img_path)
        lbl = rasterio.open(lbl_path)
        img_array = img.read()
        lbl_array = lbl.read().mean(axis=0)

        # transforms
        img_array = np.transpose(img_array, ((1, 2, 0)))
        lbl_array = np.transpose(lbl_array, ((1, 2, 0)))

        img_tensor = ToTensor()(img_array)
        lbl_tensor = ToTensor()(lbl_array)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        if self.target_transform is not None:
            lbl_tensor = self.target_transform(lbl_tensor)

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
    def __init__(self, in_channels=3, lr=1e-3):
        super().__init__()
        self.segmentation_model = smp.DeepLabV3Plus(in_channels=in_channels, activation='sigmoid')
        self.loss = nn.BCELoss()
        self.learning_rate = lr
        self.in_channels = in_channels
        self.train_iou = IoU()
        self.val_iou = IoU()

    def forward(self, x):
        return self.segmentation_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.train_iou.update(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.val_iou.update(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

class SatelliteDataModule(pl.LightningDataModule):
    def __init__(self,  transforms: str='default'):
        super().__init__()
        
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
        


if __name__ == "__main__":
    in_channels = 3
    model = SegmentationModel(in_channels=in_channels)
    data = SatelliteDataModule(in_channels=in_channels)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=30,
        logger=CSVLogger(save_dir="/data1/chabud/logs/"),
        log_every_n_steps=1
    )
    trainer.fit(model, data)
