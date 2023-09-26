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
from torchvision.transforms import ToTensor
import rasterio
import utils
from lightning.pytorch import Trainer
import argparse
from torch.utils.data import DataLoader

BASE_DIR = utils.BASE_DIR
BATCH_SIZE = 32
NUM_WORKERS = 16
torch.set_float32_matmul_precision('medium')

# the label dataset provided should be the one built using get_labels_datasets function in utils.py
# img_dir can alternatively be '/data1/malto/train' or '/data1/malto/test'

class SigspatialDataset(VisionDataset):
    def __init__(self, train=True, img_dir: Path=BASE_DIR, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.kind = "val" if train else "train"
        self.names = sorted([name for name in os.listdir(img_dir / f"ds_{self.kind}_images") if name.split(".")[-1] == "tif"])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = self.img_dir / f"ds_{self.kind}_images" / self.names[idx]
        lbl_path = self.img_dir / f"ds_{self.kind}_label" / self.names[idx]
        img = rasterio.open(img_path)
        lbl = rasterio.open(lbl_path)
        img_array = img.read()
        lbl_array = np.expand_dims(lbl.read().mean(axis=0), axis=0)

        # transforms
        img_array = np.transpose(img_array, ((1, 2, 0)))
        lbl_array = np.transpose(lbl_array, ((1, 2, 0)))

        img_tensor = ToTensor()(img_array)
        lbl_tensor = ToTensor()(lbl_array)

        img_tensor = img_tensor.type(torch.float32)
        lbl_tensor = lbl_tensor.type(torch.float32)

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
    def __init__(self, lr=1e-4):
        super().__init__()
        self.segmentation_model = smp.DeepLabV3Plus( activation='sigmoid')
        self.loss = nn.BCELoss()
        self.learning_rate = lr
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
    def __init__(self, transforms: str='default'):
        super().__init__()
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_ds = SigspatialDataset(train=True)
        self.val_ds = SigspatialDataset(train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate")
    args = parser.parse_args()
    model = SegmentationModel(lr=args.learning_rate)
    data = SatelliteDataModule()
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=50,
        logger=CSVLogger(save_dir=BASE_DIR / "logs"),
        log_every_n_steps=1
    )
    trainer.fit(model, data)
    print("---DONE---")
