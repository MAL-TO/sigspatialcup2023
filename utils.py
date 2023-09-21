import pandas as pd
import geopandas as gp
import numpy as np
from matplotlib import pyplot
import rasterio
from rasterio.plot import show
import shapely
from typing import List
from rasterio.features import geometry_mask
import rasterio.mask

def get_labels_datasets(self):
    images_namelist = ["Greenland26X_22W_Sentinel2_2019-06-03_05.tif", "Greenland26X_22W_Sentinel2_2019-06-19_20.tif", "Greenland26X_22W_Sentinel2_2019-07-31_25.tif",  "Greenland26X_22W_Sentinel2_2019-08-25_29.tif"]
    train_labels = gp.GeoDataFrame()
    test_labels = gp.GeoDataFrame()
    counter_train = 1
    counter_test = 1
    # pd.concat([train_labels, lbl_reg1_img1])
    for j in range(4):
        for i in range(6):
            lbls = self.img_labels[self.img_labels.region_num == i]
            lbls = lbls[self.img_labels.image == images_namelist[j]]
            if i%2 == 0 & j%2 == 0:
                img_num = counter_test*np.ones(len(lbls), dtype=np.int8)
                lbls['Img_number'] = img_num
                counter_test += 1
                test_labels = pd.concat([test_labels, lbls])
            if i%2 == 0 & j%2 == 1:
                img_num = counter_train*np.ones(len(lbls), dtype=np.int8)
                lbls['Img_number'] = img_num
                counter_train += 1
                train_labels = pd.concat([train_labels, lbls])
            if i%2 == 1 & j%2 == 0:
                img_num = counter_train*np.ones(len(lbls), dtype=np.int8)
                lbls['Img_number'] = img_num
                counter_train += 1
                train_labels = pd.concat([train_labels, lbls])
            if i%2 == 1 & j%2 == 1:
                img_num = counter_test*np.ones(len(lbls), dtype=np.int8)
                lbls['Img_number'] = img_num
                counter_test += 1
                test_labels = pd.concat([test_labels, lbls])
    return train_labels, test_labels

def plot_image(image_path):
    pyplot.clf()
    img = rasterio.open(image_path)
    image_array = img.read()
    image_array = np.transpose(image_array, (1, 2, 0))
    pyplot.imshow(image_array)
    pyplot.show()

def from_tif_to_tensor(image_path):

    img = rasterio.open(image_path)
    image_array = img.read()
    torch_image = ToTensor()(image_array)
    print(torch_image.shape)
    return torch_image


def divide_tif_into_regions(tif_file_paths, regions_file_path):
    regions = gp.read_file(regions_file_path)
    for j in range(4):
        src = rasterio.open(tif_file_paths[j])
        for i in range(6):
            # Extract the geometry for the current region
            region = regions.iloc[i]
            region_geometry = region['geometry']

            # Mask the image using the region's geometry
            out_image, out_transform = rasterio.mask.mask(src, [region_geometry], crop=True)
            out_meta = src.meta

            # Update the metadata for the output file
            out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})

            # Save the clipped image to a new .tif file
            output_filename = f'img{j}_region_{i+1}.tif'
            with rasterio.open(output_filename, "w", **out_meta) as dest:
                dest.write(out_image)

def get_squares(rectangle: gp.GeoSeries, num_points=10, dim=224) -> List[shapely.Polygon]:
    l = []
    for i in range(num_points):
        A = rectangle.sample_points(1)
        B = A.translate(xoff=dim, yoff=0.0, zoff=0.0)
        C = A.translate(xoff=dim, yoff=dim, zoff=0.0)
        D = A.translate(xoff=0.0, yoff=dim, zoff=0.0)

        l.append(shapely.Polygon((A[0], B[0], C[0], D[0])))
    return l

