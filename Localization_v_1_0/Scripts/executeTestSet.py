# -*- coding: utf-8 -*-

import numpy as np
import pickle
import sys
from PIL import Image


class dataGenerator:
    def __init__(self, totalBlocks, blockSize, imageSource, imgSuffix):
        self.totalBlocks = totalBlocks
        self.imageSrc = imageSource
        self.blockSize = blockSize
        self.imgSuffix = imgSuffix

    def __iter__(self):
        for i in range(self.totalBlocks):
            stackedImages = []
            for j in range(self.blockSize):
                img = np.array(Image.open(self.imageSrc + str((i * self.blockSize) + j) + self.imgSuffix))
                stackedImages.append(np.float64(img.ravel()))

            print("Block size: " + str(np.array(stackedImages).shape))
            yield np.array(stackedImages)


# ===========================================
# Execute
# ===========================================
def execute(imageSource,saveFileAt, imgCount, imgSuffix, modelName,m_id):
    executeData = dataGenerator(1, imgCount, imageSource, imgSuffix)
    flow = pickle.load(open(saveFileAt + '/' + modelName + '_' + str(m_id) + '.sav', 'rb'))
    slowFeatures = flow.execute(executeData)
    np.save(saveFileAt + '/test_' + str(m_id) + '_slowFeatures.npy', slowFeatures)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("usage: python executeTestSet")
        print("     image folder        # path to the folder containing images")
        print("     total Number of Images in the folder containing images     ")
        print("     image suffix        # suffix of the image file")
        print("     Model Name          # Trained model name")
        print("\nExample: python executeTestSet.py images/ 1600 .png train\n")
        sys.exit()
    execute(sys.argv[1],sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5],int(sys.argv[6]))
