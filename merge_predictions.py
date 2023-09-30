import utils
import os
import rasterio
from pathlib import Path
import os
import utils

BASE_DIR = utils.BASE_DIR

if __name__ == "__main__":
    big_images = sorted([name for name in os.listdir(BASE_DIR) if name.split(".")[-1] == "tif"])

    for big_image in big_images:
        small_images = sorted([name for name in os.listdir(BASE_DIR / "test_predictions") if name.startswith(big_image)])
        src_to_merge = [rasterio.open(small_image) for small_image in small_images]

        mosaic, out_transform = rasterio.merge.merge(src_to_merge)
        out_meta = rasterio.open(small_images[0]).meta.copy()
        out_meta.update({"driver": "GTiff",
                          "height": mosaic.shape[1],
                          "width": mosaic.shape[2],
                          "transform": out_transform
                          }
                        )
        
        with rasterio.open(BASE_DIR / "predictions" / big_image, "w", **out_meta) as dest:
            dest.write(mosaic)

