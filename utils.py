import pandas as pd
import geopandas as gp
import numpy as np
from pathlib import Path
from matplotlib import pyplot
import rasterio
from rasterio.plot import show
import shapely
from typing import List, Tuple
import affine
from rasterio.features import geometry_mask
import rasterio.mask
import os
import cv2
from shapely import Polygon

BASE_DIR = Path('/data1/malto/sigspatial')

def get_train_test_images():
    images_namelist = sorted([name for name in os.listdir(BASE_DIR) if name.split(".")[-1] == "tif"])
    # pd.concat([train_labels, lbl_reg1_img1])
    test_folder = BASE_DIR / "test"
    train_folder = BASE_DIR / "train"
    
    regions_file_path = BASE_DIR / "lakes_regions.gpkg"
    regions = gp.read_file(regions_file_path)

    for j in range(4):
        for i in range(6):
            if i % 2 == 0 and j % 2 == 0:
                new_img_name = test_folder / f'{images_namelist[j]}_region_{i+1}.tif'
                divide_tif_into_regions(BASE_DIR / images_namelist[j], regions.iloc[i]['geometry'], new_img_name)
            if i % 2 == 0 and j % 2 == 1:
                new_img_name = train_folder / f'{images_namelist[j]}_region_{i+1}.tif'
                divide_tif_into_regions(BASE_DIR / images_namelist[j], regions.iloc[i]['geometry'], new_img_name)
            if i % 2 == 1 and j % 2 == 0:
                new_img_name = train_folder / f'{images_namelist[j]}_region_{i+1}.tif'
                divide_tif_into_regions(BASE_DIR / images_namelist[j], regions.iloc[i]['geometry'], new_img_name)
            if i % 2 == 1 and j % 2 == 1:
                new_img_name = test_folder / f'{images_namelist[j]}_region_{i+1}.tif'
                divide_tif_into_regions(BASE_DIR / images_namelist[j], regions.iloc[i]['geometry'], new_img_name)
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
            

def get_square(x: int, y: int, dim: int=100000) -> shapely.Polygon:
    
    A = shapely.Point(x, y) 
    B = shapely.Point(x + dim, y)
    C = shapely.Point(x + dim, y + dim)
    D = shapely.Point(x, y + dim)

    return shapely.Polygon((A, B, C, D))

def get_external_rectangle(regions: gp.GeoDataFrame, num_region: int) -> gp.GeoSeries:
    """
    The rectangle is parallel to the axes
    """
    return gp.GeoSeries(regions[regions['region_num'] == num_region]['geometry'].envelope)

def is_valid(image: np.ndarray):
    # Sum the pixel values along the color channels (axis=2)
    channel_sums = np.sum(image, axis=0)
    print(channel_sums.min(), channel_sums.max())
    # Check if all pixel sums are either 0 (black) or 255*3 (white)
    return np.all((channel_sums > 0) & (channel_sums < 255 * 3))

def generate_image(img_region_path: Path, rectangle: gp.GeoSeries, out_file_path: Path):
    img_region = rasterio.open(img_region_path)
    out_image, out_transform = rasterio.mask.mask(img_region, [rectangle], crop=True)
    save_as_tif(out_image, out_transform, img_region, out_file_path)


def get_image_and_region(img_tiff_path: str) -> Tuple[str, int]:
    split = img_tiff_path.split("_region_")
    image = split[0]
    region = int(split[-1].split(".")[0])
    return image, region

def generate_label(cropped_img_tiff_path: Path, lake_geom: gp.GeoSeries, out_image_path: Path):
    """
    lake_geom is lakes_regions[lakes_regions['region_num'] == 2 & lakes_region['image'] == image]['geometry']
    """
    img_trial = rasterio.open(cropped_img_tiff_path)

    out_image, out_transform = rasterio.mask.mask(img_trial, lake_geom)
    #out_image = out_image.sum(axis=0)
    out_image[out_image != 0] = 255

    save_as_tif(out_image, out_transform, img_trial, out_image_path)

def mask_to_polygons(mask):

    transformer = rasterio.transform.AffineTransformer(mask.transform)
    mask_array = mask.read()
    grayscale_image = cv2.cvtColor(mask_array.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.flip(grayscale_image, 0)
    grayscale_image = cv2.flip(grayscale_image, 1)

    contours, _ = cv2.findContours(grayscale_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Construct the polygons 
    polygons = []
    for c in contours:
        c = c.reshape(-1,2)
        new_poly = []
        for point in c:
            y = point[0]
            x = point[1]
            new_poly.append(transformer.xy(x, y))
        if len(new_poly) > 2:
            polygons.append(shapely.Polygon(new_poly))

    return polygons



if __name__ == "__main__":
    if not os.path.exists(BASE_DIR / "test"):
        os.mkdir(BASE_DIR / "test")
    if not os.path.exists(BASE_DIR / "train"):
        os.mkdir(BASE_DIR / "train")
    get_train_test_images()
