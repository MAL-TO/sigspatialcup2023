from tqdm import tqdm
import os 
import utils
import rasterio
from pathlib import Path
import generate_dataset

BASE_DIR = utils.BASE_DIR
DIM = generate_dataset.DIM

def remove_wrong(tipo: str):
    if tipo == "test":
        image_path = BASE_DIR / "test_image"
    else:
        image_path = BASE_DIR / f"ds_{tipo}_images"
    l = os.listdir(image_path)
    print(len(l))

    for i in tqdm(range(len(l))):
        img = l[i]
        img_region = rasterio.open(image_path / img)
        img_array = img_region.read()
        if img_array.shape != (3, DIM, DIM):
            os.remove(image_path / img)
            if tipo != "test":
                os.remove(BASE_DIR / f"ds_{tipo}_label" / img)

if __name__ == "__main__":
    remove_wrong("train")
    remove_wrong("val")
    remove_wrong("test")