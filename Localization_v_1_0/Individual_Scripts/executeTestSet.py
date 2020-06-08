# -*- coding: utf-8 -*-

import numpy as np
import pickle
import sys
from os import walk
from PIL import Image
import os


class dataGenerator:
    def __init__(self, totalBlocks, blockSize, imageSource, imgSuffix,last_count):
        self.totalBlocks = totalBlocks
        self.imageSrc = imageSource
        self.blockSize = blockSize
        self.imgSuffix = imgSuffix
        self.last_count = last_count

    def __iter__(self):
        for i in range(self.totalBlocks):
            stackedImages = []
            for j in range(self.last_count):
                if os.path.isfile ( self.imageSrc + 'omni_' + str ( (i * self.blockSize) + j ) + self.imgSuffix ) :
                    # img = np.array(Image.open(self.imageSrc + str((i * self.blockSize) + self.fileArray[j]) + self.imgSuffix))
                    img = np.array (Image.open ( self.imageSrc + 'omni_' + str ( (i * self.blockSize) + j ) + self.imgSuffix ) )
                    stackedImages.append ( np.float64 ( img.ravel ( ) ) )
                '''    
                else:
                    print(self.imageSrc + 'omni_' + str ( (i * self.blockSize) + j ) + self.imgSuffix )
                '''
            print("Block size: " + str(np.array(stackedImages).shape))
            yield np.array(stackedImages)


# ===========================================
# Execute
# ===========================================
def execute(imageSource,saveFileAt, imgCount, label_id, imgSuffix, modelName):
    tags = ['Building_Corner', 'Hut', 'Garden_Entry']
    label_name = tags[label_id]
    # saving the exact file numbers present in the folder
    f = []
    for (dir_path, dir_names, filenames) in walk(imageSource):
        f.extend(filenames)
        break

    # clearing the part of .jpg from the string and converting it into int
    fileNumbers = []
    for i in range(0, len(f)):
        fileNumbers.append(int(f[i][5:-4]))

    # last file number
    max_file_num = max (fileNumbers) + 1
    executeData = dataGenerator(1, imgCount, imageSource, imgSuffix, max_file_num)
    flow = pickle.load(open(saveFileAt + modelName + '_' + label_name + '.sav', 'rb'))
    slowFeatures = flow.execute(executeData)
    np.save(saveFileAt + '/test_' + label_name + '_' + 'slowFeatures.npy', slowFeatures)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Not a successful Testing")
        sys.exit()
    execute(sys.argv[1],sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
