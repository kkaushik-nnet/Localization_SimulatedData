# -*- coding: utf-8 -*-

# from createSFANetwork import *
import numpy as np
import pickle
import sys
from PIL import Image

from Localization_v_1_0.Scripts.createSFANetwork import createSFANetwork


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
def trainAndExecute(imageSource, saveFileAt, imgCount, imgSuffix, modelName, m_id):
    flow = createSFANetwork()
    trainingData = dataGenerator(1, imgCount, imageSource, imgSuffix)
    flow.train([trainingData, trainingData, trainingData, trainingData, trainingData, trainingData, trainingData])

    executeData = dataGenerator(1, imgCount, imageSource, imgSuffix)
    slowFeatures = flow.execute(executeData)

    pickle.dump(flow, open(saveFileAt + '/Evaluation_Arrays/' + modelName + '_' + m_id + '.sav', 'wb'))
    print('npy file is at : ' + saveFileAt + '/Evaluation_Arrays/' + modelName + '_' + m_id + '_slowFeatures.npy')
    np.save(saveFileAt + '/Evaluation_Arrays/' + modelName + '_' + m_id + '_slowFeatures.npy', slowFeatures)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: python trainAndExecuteNetwork")
        print("     image folder        # path to the folder containing images")
        print("     total Number of Images in the folder containing images     ")
        print("     image suffix        # suffix of the image file")
        print("     Model Name          # Save the model with the specified name")
        print("\nExample: python trainAndExecuteNetwork.py images/ 1600 .png train\n")
        print(
            "\n python trainAndExecuteNetwork.py '/hri/localdisk/ThesisProject/Kaushik/Simulator_data/SingleMarker"
            "/Sample_Extracted_train/' 890 '.png' 'train'")
        sys.exit()
    trainAndExecute(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]))
