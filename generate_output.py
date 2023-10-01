import utils
import geopandas as gp
import os
import pathlib
import rasterio
import pandas as pd

BASE_DIR = utils.BASE_DIR

def generate_output():
    # Create empty dataFrame
    column_names = ["image", "region_num", "geometry"]
    # column_names = [str , numpy.int64, shapely.geometry.polygon.Polygon]
    gdf = gp.GeoDataFrame(columns = column_names)

    regions = sorted([name for name in os.listdir(BASE_DIR / "big_predictions")])

    for region in regions:
        for mask_path in os.listdir(BASE_DIR / "big_predictions" / region):
            if 'part' not in mask_path:
                whole_mask_path = BASE_DIR / "big_predictions" / region / mask_path
                mask = rasterio.open(whole_mask_path)
                polygons_list = utils.mask_to_polygons(mask)
                for polygon in polygons_list:
                    new_row = {'image': mask_path,'region_num': region, 'geometry': polygon}
                    gdf.loc[len(gdf)] = new_row 
                mask.close()
                gdf.set_crs(3857)

    # Filter small lakes
    areas = gdf.area
    gdf['Area'] = areas
    gdf_filt = gdf[gdf['Area']>= 100000]
    gdf_filt.drop('Area', inplace=True, axis=1)
    

    gdf_filt.to_file('lake_polygons_test.gpkg', driver='GPKG') 

if __name__ == "__main__":
    generate_output()