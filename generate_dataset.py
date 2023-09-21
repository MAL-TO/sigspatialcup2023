from utils import *
import pathlib


DIR = pathlib.Path("/data1/malto/sigspatial")


if __name__ == "__main__":
    
    lakes_train_test = DIR / "lakes_regions.gpkg"
    lakes_regions_path = DIR / "lake_polygons_training.gpkg"
    regions = gp.read_file(lakes_train_test)
    lakes_regions = gp.read_file(lakes_regions_path)

    for region in regions:
        