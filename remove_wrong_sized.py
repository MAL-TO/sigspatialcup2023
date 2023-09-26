from tqdm import tqdm
import os 
import utils
import rasterio
from pathlib import Path

BASE_DIR = utils.BASE_DIR


train_image_path = BASE_DIR / "ds_train_images"
l = os.listdir(train_image_path)
print(len(l))


for i in tqdm(range(len(l))):
    img = l[i]
    img_region = rasterio.open(train_image_path / img)
    img_array = img_region.read()
    #if img_array.max() != 0:
    #    break
    if img_array.shape != (3, 2048, 2048):
        os.remove(train_image_path / img)
        os.remove(BASE_DIR / "ds_train_label" / img)


val_image_path = BASE_DIR / "ds_val_images"
l = os.listdir(val_image_path)
print(len(l))

for i in tqdm(range(len(l))):
    img = l[i]
    img_region = rasterio.open(val_image_path / img)
    img_array = img_region.read()
    #if img_array.max() != 0:
    #    break
    if img_array.shape != (3, 2048, 2048):
        os.remove(val_image_path / img)
        os.remove(BASE_DIR / "ds_val_label" / img)