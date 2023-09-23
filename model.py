from typing import Any, Callable, Optional
# import pytorch_lightning as pl
# import segmentation_models_pytorch as smp
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader, Dataset
# import numpy as np
# from pytorch_lightning.loggers import CSVLogger
# from dataset_utils import get_folds, np_to_torch
# from torchmetrics import Metric
from torchvision.datasets import VisionDataset, Dataset
import torch

import os
import pandas as pd
import geopandas as gp
import numpy as np
from torchvision.transforms import ToTensor
import rasterio

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler
from torchgeo.trainers import SemanticSegmentationTask

from utils import get_labels_datasets

BATCH_SIZE = 32
NUM_WORKERS = 16
torch.set_float32_matmul_precision('medium')

# the label dataset provided should be the one built using get_labels_datasets function in utils.py
# img_dir can alternatively be '/data1/malto/train' or '/data1/malto/test'

class SigspatialDataset(Dataset):
    def __init__(self, labels_dataset, img_dir, transform=ToTensor(), target_transform=None):
        self.img_labels = labels_dataset
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx, train = True):
        if train:
            img_path = f'train_img{idx+1}.tif'
        else:
            img_path = f'test_img{idx+1}.tif'
        img = rasterio.open(img_path)
        img_array = img.read()
        label = self.img_labels[self.img_labels.Img_number == idx+1]
        if self.transform:
            image = self.transform(img_array)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

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
    def __init__(self, in_channels: int, transforms: str='default'):
        super().__init__()
        self.in_channels = in_channels
        self.train_folds = [1]
        self.val_folds = [0]
        
    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        train_post, train_pre, train_masks, _ = get_folds(self.train_folds, self.in_channels)
        val_post, val_pre, val_masks, _ = get_folds(self.val_folds, self.in_channels)

        if self.in_channels == 12: 
            train = train_post
            val = val_post
        elif self.in_channels == 24: 
            train = np.concatenate((train_pre, train_post), axis=3)
            val = np.concatenate((val_pre, val_post), axis=3)
        elif self.in_channels == 36: 
            train_diff = train_pre - train_post 
            train = np.concatenate((train_pre, train_post, train_diff), axis=3)
            val_diff = val_pre - val_post
            val = np.concatenate((val_pre, val_post, val_diff), axis=3)
        else: 
            raise NotImplementedError
        
        self.train_dataset = sigspatialDataset(TensorDataset(np_to_torch(train), np_to_torch(train_masks)))
        self.val_dataset = sigspatialDataset(TensorDataset(np_to_torch(val), np_to_torch(val_masks)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


if __name__ == "__main__":
    in_channels = 24
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
