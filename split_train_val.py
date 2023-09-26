import os
import utils

BASE_DIR = utils.BASE_DIR

if __name__ == "__main__":
    if not os.path.exists(BASE_DIR / "ds_train_images"):
        os.mkdir(BASE_DIR / "ds_train_images")
    if not os.path.exists(BASE_DIR / "ds_train_label"):
        os.mkdir(BASE_DIR / "ds_train_label")
    if not os.path.exists(BASE_DIR / "ds_val_images" ):
        os.mkdir(BASE_DIR / "ds_val_images" )
    if not os.path.exists(BASE_DIR / "ds_val_label"):
        os.mkdir(BASE_DIR / "ds_val_label")
    
    originating_images = sorted([name for name in os.listdir(BASE_DIR / "train") if name.split(".")[-1] == "tif"])
    train_images = originating_images[:9]
    val_images = originating_images[9:]

    small_images = sorted([name for name in os.listdir(BASE_DIR / "train_image") if name.split(".")[-1] == "tif"])

    for small_image in small_images:
        for train_prefix in train_images:
            if small_image.startswith(train_prefix):
                os.rename(BASE_DIR / "train_image" / small_image, BASE_DIR / "ds_train_images" / small_image)
                os.rename(BASE_DIR / "train_label" / small_image, BASE_DIR / "ds_train_label" / small_image)
        for val_prefix in val_images:
            if small_image.startswith(val_prefix):
                os.rename(BASE_DIR / "train_image" / small_image, BASE_DIR / "ds_val_images" / small_image)
                os.rename(BASE_DIR / "train_label" / small_image, BASE_DIR / "ds_val_label" / small_image)
                