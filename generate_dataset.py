from utils import *
import pathlib

DIR = pathlib.Path("/data1/malto/sigspatial")

if __name__ == "__main__":
    
    lakes_train_test = DIR / "lakes_regions.gpkg"
    lakes_regions_path = DIR / "lake_polygons_training.gpkg"
    regions = gp.read_file(lakes_train_test)
    lakes_regions = gp.read_file(lakes_regions_path)

    ts = ['train', 'test']

    for t in ts:
        full_names = os.listdir(IMG_DIR / t)
        for full_name in full_names:
            image, region_num = get_image_and_region(full_names)
            lake_geom = lakes_regions[(lakes_regions['image'] == image) & (lakes_regions['region_num'] == region_num)]

            # sliding window to crop big image
            # coodrinate quantum per pixel is 38.2185141426
            big_rect = get_external_rectangle(regions, region_num)
            xx, yy = big_rect[0].exterior.coords.xy
            s = set()
            for x, y in zip(xx, yy):
                s.add((x, y))
            l = list(s)
            l.sort(key=lambda p: p[0])
            l.sort(key=lambda p: p[1])
            l


                #out_image, out_transform = generate_label(IMG_DIR / t / full_name, lake_geom, IMG_DIR / (t + "_label") / full_name)

