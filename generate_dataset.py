from utils import *
import pathlib
from threading import Thread

DIR = pathlib.Path("/data1/malto/sigspatial")
CQPP = 38.2185141426

def thread_function(starting_image_path, full_name, t, x, y, step):
    image_path = DIR / (t + "_image") / (full_name + f"_{x}_{y}.tif")
    label_path = DIR / (t + "_label") / (full_name + f"_{x}_{y}.tif")
    src = rasterio.open(starting_image_path)
            #print(x, y, src.read().shape, src.bounds, src.bounds)
            #print(lake_geom)
    generate_image(starting_image_path, get_square(x, y, step), image_path) 
    if t != "test":
        generate_label(image_path, lake_geom, label_path)

if __name__ == "__main__":
   
    if not os.path.exists(DIR / "train_image"):
        os.mkdir(DIR / "train_image")
    if not os.path.exists(DIR / "train_label"):
        os.mkdir(DIR / "train_label")
    if not os.path.exists(DIR / "test_image"):
        os.mkdir(DIR / "test_image")

    lakes_train_test = DIR / "lakes_regions.gpkg"
    lakes_regions_path = DIR / "lake_polygons_training.gpkg"
    regions = gp.read_file(lakes_train_test)
    lakes_regions = gp.read_file(lakes_regions_path)

    ts = ['train', 'test']

    for t in ts:
        full_names = os.listdir(DIR / t)
        for full_name in full_names:
            image, region_num = get_image_and_region(full_name)
            lake_geom = lakes_regions[(lakes_regions['image'] == image) & (lakes_regions['region_num'] == region_num)]['geometry']

            # sliding window to crop big image
            # coodrinate quantum per pixel is 38.2185141426
            big_rect = get_external_rectangle(regions, region_num)
            #print(big_rect)
            xx, yy = big_rect.iloc[0].exterior.coords.xy
            #print(xx)
            #print(yy)
            xmin, xmax = min(xx), max(xx)
            ymin, ymax = min(yy), max(yy)
            
            step = 224 * CQPP
            
            thread_list = []
            starting_image_path = DIR / t / full_name 
            for x in np.arange(xmin, xmax, step / 2):
                for y in np.arange(ymin, ymax, step / 2):
                    thread_list.append(Thread(target=thread_function, args=(starting_image_path, full_name, t, x, y, step)))

            for thread in thread_list:
                thread.start()

            for thread in thread_list:
                thread.join()

