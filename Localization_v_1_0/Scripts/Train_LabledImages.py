# -*- coding: utf-8 -*-

# from createSFANetwork import *
import numpy as np
import pickle
import sys
from PIL import Image

from createSFANetwork import createSFANetwork


class dataGenerator:
    def __init__(self, totalBlocks, blockSize, imageSource, imgSuffix, lastRow=False):
        self.totalBlocks = totalBlocks
        self.imageSrc = imageSource
        self.blockSize = blockSize
        self.last_row = []
        self.lastRow = lastRow
        self.imgSuffix = imgSuffix

    def __iter__(self):
        for i in range(self.totalBlocks):
            stackedImages = []
            for j in range(self.blockSize):
                img = np.array(Image.open(self.imageSrc + str((i * self.blockSize) + j) + self.imgSuffix))
                stackedImages.append(np.float64(img.ravel()))
            if i > 0 and self.lastRow:
                stackedImages = np.vstack((self.last_row, stackedImages))
            self.last_row = stackedImages[-1].copy()
            print("Block size: " + str(np.array(stackedImages).shape))
            yield np.array(stackedImages)


# ===========================================
# train and execute
# ===========================================
def trainAndExecute(imageSource, saveFileAt, imgCount, imgSuffix, modelName):
    flow = createSFANetwork()
    trainingData = dataGenerator(1, imgCount, imageSource, imgSuffix)
    flow.train([trainingData, trainingData, trainingData, trainingData, trainingData, trainingData,trainingData])
    executeData = dataGenerator(1, imgCount, imageSource, imgSuffix)
    slowFeatures = flow.execute(executeData)

    pickle.dump(flow, open(saveFileAt + modelName + '_' + '.sav', 'wb'))
    print('npy file is at : ' + saveFileAt + modelName + '_slowFeatures.npy')
    np.save(saveFileAt + modelName + '_slowFeatures.npy', slowFeatures)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Not Successful..!")
        sys.exit()
    trainAndExecute(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
