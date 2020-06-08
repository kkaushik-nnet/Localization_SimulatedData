# -*- coding: utf-8 -*-

# from createSFANetwork import *
import numpy as np
import pickle
import sys
from os import walk
from PIL import Image
import os

from createSFANetwork import createSFANetwork


class dataGenerator:
    def __init__(self, totalBlocks, blockSize, imageSource, imgSuffix,last_count, lastRow=False):
        self.totalBlocks = totalBlocks
        self.imageSrc = imageSource
        self.blockSize = blockSize
        self.last_row = []
        self.lastRow = lastRow
        self.imgSuffix = imgSuffix
        self.last_count = last_count

    def __iter__(self):
        for i in range(self.totalBlocks):
            stackedImages = []
            for j in range(self.last_count):
                if os.path.isfile (self.imageSrc + 'omni_' + str ( (i * self.blockSize) + j ) + self.imgSuffix):
                    # img = np.array(Image.open(self.imageSrc + str((i * self.blockSize) + self.fileArray[j]) + self.imgSuffix))
                    img = np.array (Image.open ( self.imageSrc + 'omni_' + str ( (i * self.blockSize) + j ) + self.imgSuffix ) )
                    # print ( self.imageSrc + 'omni_' + str ( (i * self.blockSize) + j ) + self.imgSuffix )
                    stackedImages.append ( np.float64 ( img.ravel ( ) ) )
                '''
                else:
                    print ( self.imageSrc + 'omni_' + str ( (i * self.blockSize) + j ) + self.imgSuffix )
                '''
            if i > 0 and self.lastRow:
                stackedImages = np.vstack((self.last_row, stackedImages))
            self.last_row = stackedImages[-1].copy()
            print("Block size: " + str(np.array(stackedImages).shape))
            yield np.array(stackedImages)


# ===========================================
# train and execute
# ===========================================
def trainAndExecute(imageSource, saveFileAt, imgCount,label_id, imgSuffix, modelName):
    tags = ['Building_Corner', 'Hut', 'Garden_Entry']
    label_name = tags[label_id]
    flow = createSFANetwork()

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
    max_file_num = max(fileNumbers)+1
    trainingData = dataGenerator(1, imgCount, imageSource, imgSuffix,max_file_num)
    flow.train([trainingData, trainingData, trainingData, trainingData, trainingData, trainingData,trainingData])
    executeData = dataGenerator(1, imgCount, imageSource, imgSuffix,max(fileNumbers)+1)
    slowFeatures = flow.execute(executeData)

    pickle.dump(flow, open(saveFileAt + modelName + '_' + label_name + '.sav', 'wb'))
    print('npy file is at : ' + saveFileAt + modelName + '_' + label_name + '_slowFeatures.npy')
    np.save(saveFileAt + modelName + '_' + label_name + '_slowFeatures.npy', slowFeatures)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Not Successful..!")
        sys.exit()
    trainAndExecute(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
