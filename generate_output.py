import utils
import geopandas as gp
import os
import pathlib
import rasterio
import pandas as pd

BASE_DIR = utils.BASE_DIR

if __name__ == "__main__":
    # Create empty dataFrame
    column_names = ["image", "region_num", "geometry"]
    # column_names = [str , numpy.int64, shapely.geometry.polygon.Polygon]
    gdf = gp.GeoDataFrame(columns = column_names)

    regions = sorted([name for name in os.listdir(BASE_DIR / "big_predictions")])


    for region in regions:
        for mask_path in os.listdir(BASE_DIR / "big_predictions" / region):
            whole_mask_path = BASE_DIR / "big_predictions" / region / mask_path
            mask = rasterio.open(whole_mask_path)
            mask_array = mask.read()
            polygons_list = utils.mask_to_polygons(mask_array)
            for polygon in polygons_list:
                new_row = {'image': mask_path,'region_num': region, 'geometry': polygon}
                gdf.loc[len(gdf)] = new_row   
    gdf.set_geometry('geometry', drop=False, inplace=True, crs="EPSG:4326") 

            
    gdf.to_file('lake_polygons_test.gpkg', driver='GPKG') 