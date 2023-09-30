import utils
import os
import rasterio
from rasterio.merge import merge
from pathlib import Path
import os
import utils

BASE_DIR = utils.BASE_DIR

if __name__ == "__main__":
    if not os.path.exists(BASE_DIR / "big_predictions"):
            os.mkdir(BASE_DIR / "big_predictions")

    regions = [1, 2, 3, 4, 5, 6]
    for region in regions:
        if not os.path.exists(BASE_DIR / "big_predictions" / f"{region}"):
            os.mkdir(BASE_DIR / "big_predictions" / f"{region}")

    big_images = sorted([name for name in os.listdir(BASE_DIR) if name.split(".")[-1] == "tif"])
    for big_image in big_images:
        for region in regions:
            small_images = sorted([name for name in os.listdir(BASE_DIR / "predictions") if name.startswith(big_image) and f"_region_{region}" in name])
            
            if len(small_images > 1000):
                # max 2900 so <3000
                for i in [0, 1000, 2000]:
                    src_to_merge = [rasterio.open(BASE_DIR / "predictions" / small_image) for small_image in small_images[i:i+1000]]
                    if len(src_to_merge) > 0:
                        mosaic, out_transform = merge(src_to_merge)
                        out_meta = src_to_merge[0].meta.copy()
                        out_meta.update({"driver": "GTiff",
                                        "height": mosaic.shape[1],
                                        "width": mosaic.shape[2],
                                        "transform": out_transform
                                        }
                                        )
                        if not os.path.exists(BASE_DIR / "big_predictions" / f"{region}" / "part"):
                            os.mkdir(BASE_DIR / "big_predictions" / f"{region}" / "part")
                        with rasterio.open(BASE_DIR / "big_predictions" / f"{region}" / "part" / f"{i // 1000}" + big_image , "w", **out_meta) as dest:
                            dest.write(mosaic)
                partial_images = os.listdir(BASE_DIR / "big_predictions" / f"{region}" / "part")
                src_to_merge = [rasterio.open(BASE_DIR / "big_predictions" / f"{region}" / "part" / partial_image) for partial_image in partial_images]
                if len(src_to_merge) > 0:
                    mosaic, out_transform = merge(src_to_merge)
                    out_meta = src_to_merge[0].meta.copy()
                    out_meta.update({"driver": "GTiff",
                                    "height": mosaic.shape[1],
                                    "width": mosaic.shape[2],
                                    "transform": out_transform
                                    }
                                    )
                    with rasterio.open(BASE_DIR / "big_predictions" / f"{region}" / big_image , "w", **out_meta) as dest:
                        dest.write(mosaic)
            else:
                src_to_merge = [rasterio.open(BASE_DIR / "predictions" / small_image) for small_image in small_images]
                if len(src_to_merge) > 0:
                    mosaic, out_transform = merge(src_to_merge)
                    out_meta = src_to_merge[0].meta.copy()
                    out_meta.update({"driver": "GTiff",
                                    "height": mosaic.shape[1],
                                    "width": mosaic.shape[2],
                                    "transform": out_transform
                                    }
                                    )
                    with rasterio.open(BASE_DIR / "big_predictions" / f"{region}" / big_image , "w", **out_meta) as dest:
                        dest.write(mosaic)
            
