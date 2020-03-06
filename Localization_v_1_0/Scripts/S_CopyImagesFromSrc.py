import shutil
import pandas as pd
from tqdm import tqdm


def copyUniqueImages(coordinates_path, source_path, destination_path):

    image_file = pd.read_csv(
        coordinates_path, sep=',',
        header=None)
    # image_names = image_file[3]
    print("Source Path : " + source_path)
    print("Destination Path : " + destination_path)

    for i in tqdm(range(len(image_file))):
        shutil.copy(source_path+str(i)+'.png', destination_path)
