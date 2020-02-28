import shutil
import pandas as pd
from tqdm import tqdm


def copyUniqueImages(coordinates_path, source_path, destination_path):
    image_file = pd.read_csv(
        coordinates_path, sep=',',
        header=None)
    image_names = image_file[3]
    for i in tqdm(range(1, len(image_names))):
        shutil.copy(source_path + image_names[i], destination_path + image_names[i])
