import pandas as pd
import geopandas as gp
import numpy as np
from pathlib import Path
from matplotlib import pyplot
import rasterio
from rasterio.plot import show
import shapely
from typing import List
import affine
from rasterio.features import geometry_mask
import rasterio.mask
import os

IMG_DIR = Path("/data1/malto/sigspatial")

def get_train_test_images():
    images_namelist = sorted([name for name in os.listdir(IMG_DIR) if name.split(".")[-1] == "tif"])
    # pd.concat([train_labels, lbl_reg1_img1])
    test_folder = IMG_DIR / "test"
    train_folder = IMG_DIR / "train"
    
    regions_file_path = IMG_DIR / "lakes_regions.gpkg"
    regions = gp.read_file(regions_file_path)

    for j in range(4):
        for i in range(6):
            if i % 2 == 0 and j % 2 == 0:
                new_img_name = test_folder / f'{images_namelist[j]}_region_{i+1}.tif'
                divide_tif_into_regions(IMG_DIR / images_namelist[j], regions.iloc[i]['geometry'], new_img_name)
            if i % 2 == 0 and j % 2 == 1:
                new_img_name = train_folder / f'{images_namelist[j]}_region_{i+1}.tif'
                divide_tif_into_regions(IMG_DIR / images_namelist[j], regions.iloc[i]['geometry'], new_img_name)
            if i % 2 == 1 and j % 2 == 0:
                new_img_name = train_folder / f'{images_namelist[j]}_region_{i+1}.tif'
                divide_tif_into_regions(IMG_DIR / images_namelist[j], regions.iloc[i]['geometry'], new_img_name)
            if i % 2 == 1 and j % 2 == 1:
                new_img_name = test_folder / f'{images_namelist[j]}_region_{i+1}.tif'
                divide_tif_into_regions(IMG_DIR / images_namelist[j], regions.iloc[i]['geometry'], new_img_name)
    return None

def plot_image(image_path):
    pyplot.clf()
    img = rasterio.open(image_path)
    image_array = img.read()
    image_array = np.transpose(image_array, (1, 2, 0))
    pyplot.imshow(image_array)
    pyplot.show()

def from_tif_to_tensor(image_path):
    pass
    #img = rasterio.open(image_path)
    #image_array = img.read()
    #torch_image = ToTensor()(image_array)
    #print(torch_image.shape)
    #return torch_image

def save_as_tif(out_image: np.ndarray, out_transform: affine.Affine, src: rasterio.io.DatasetReader, new_img_path: Path):
    out_meta = src.meta
    # Update the metadata for the output file
    out_meta.update({"driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform})

    # Save the clipped image to a new .tif file
    output_filename = new_img_path
    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(out_image)

def divide_tif_into_regions(tif_file_path: Path, region_geometry: gp.GeoSeries, new_img_path: Path):
    
    src = rasterio.open(tif_file_path)
    # Extract the geometry for the current region
    # Mask the image using the region's geometry
    out_image, out_transform = rasterio.mask.mask(src, [region_geometry], crop=True)
    save_as_tif(out_image, out_transform, src, new_img_path)
            

def get_square(rectangle: gp.GeoSeries, dim=100000) -> shapely.Polygon:
    
    A = rectangle.sample_points(1)

    B = A.translate(xoff=dim, yoff=0.0, zoff=0.0)
    C = A.translate(xoff=dim, yoff=dim, zoff=0.0)
    D = A.translate(xoff=0.0, yoff=dim, zoff=0.0)
    return shapely.Polygon((A[0], B[0], C[0], D[0]))

def get_external_rectangle(regions: gp.GeoDataFrame, num_region: int) -> gp.GeoSeries:
    return gp.GeoSeries(regions.iloc[num_region][1].minimum_rotated_rectangle)

def is_valid(image: np.ndarray):
    # Sum the pixel values along the color channels (axis=2)
    channel_sums = np.sum(image, axis=0)
    print(channel_sums.min(), channel_sums.max())
    # Check if all pixel sums are either 0 (black) or 255*3 (white)
    return np.all((channel_sums > 0) & (channel_sums < 255 * 3))

def get_valid_image(img_region_path: str, rectangle: gp.GeoSeries) -> np.ndarray:
    img_region = rasterio.open(img_region_path)
    while True:
        poly = get_square(rectangle)
        out_image, _ = rasterio.mask.mask(img_region, [poly], crop=True)
        if not is_valid(out_image):
            continue
        else:
            break

def generate_label(cropped_img_tiff_path: Path, lake_geom: gp.GeoSeries):
    """
    lake_geom is lakes_regions[lakes_regions['region_num'] == 2]['geometry']
    """
    img_trial = rasterio.open(cropped_img_tiff_path)

    out_image, out_transform = rasterio.mask.mask(img_trial, lake_geom)
    out_image = out_image.sum(axis=0)
    out_image[out_image != 0] = 1

    new_img_path = "label" / cropped_img_tiff_path

    save_as_tif(out_image, out_transform, img_trial, new_img_path)

if __name__ == "__main__":
    if not os.path.exists(IMG_DIR / "test"):
        os.mkdir(IMG_DIR / "test")
    if not os.path.exists(IMG_DIR / "train"):
        os.mkdir(IMG_DIR / "train")
    get_train_test_images()