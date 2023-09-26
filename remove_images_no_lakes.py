import os 
import utils
import rasterio

BASE_DIR = utils.BASE_DIR

train_images = sorted([name for name in os.listdir(BASE_DIR / "ds_train_images") if name.split(".")[-1] == "tif"])
val_images = sorted([name for name in os.listdir(BASE_DIR / "ds_val_images") if name.split(".")[-1] == "tif"])

for im in train_images:
    if rasterio.open(BASE_DIR / "ds_train_label" / im).read().max() == 0:
        os.remove(BASE_DIR / "ds_train_label" / im)
        os.remove(BASE_DIR / "ds_train_images" / im)

for im in val_images:
    if rasterio.open(BASE_DIR / "ds_val_label" / im).read().max() == 0:
        os.remove(BASE_DIR / "ds_val_label" / im)
        os.remove(BASE_DIR / "ds_val_images" / im)
