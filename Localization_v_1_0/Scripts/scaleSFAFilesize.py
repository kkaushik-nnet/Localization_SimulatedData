import numpy as np
import pandas as pd
import sys


def scaleFile(outputPath, trainSetCoordsPath, testSetCoordsPath, aruco, train_test_data):
    m_id = str(aruco)
    # Train Set
    # Slow Feature Values
    trainSetPath = outputPath + '/' + 'train_' + m_id + '_slowFeatures.npy'

    # Marker detection results
    detectionResultsTrainSetPath = outputPath + '/' + 'result_train.csv'

    # Load Files
    trainSet = np.load(trainSetPath)
    trainSetCoords = np.loadtxt(trainSetCoordsPath, delimiter=',', usecols=(4, 5))
    detectionResultsTrainSet = pd.read_csv(detectionResultsTrainSetPath)

    markerSquaresTrainSet = detectionResultsTrainSet[
        [m_id + '_bb_x1', m_id + '_bb_y1', m_id + '_bb_x2', m_id + '_bb_y2', m_id + '_bb_x3', m_id + '_bb_y3', m_id +
         '_bb_x4', m_id + '_bb_y4']].values

    # Keep only the coordinates with valid marker detection (Train/Test Set)
    trainCoords = []
    n = 0
    for i in range(trainSetCoords.shape[0]):
        if detectionResultsTrainSet.iloc[i, 1:].sum() != 0:
            if markerSquaresTrainSet[i].sum() == 0:
                trainSet = np.insert(trainSet, n, np.zeros((1, 8)), 0)
            n = n + 1
            trainCoords.append(trainSetCoords[i, :])
    trainCoords = np.array(trainCoords)

    print(m_id, " _trainSet :", trainSet.shape)
    print(m_id,"_trainCoords :",trainCoords.shape)
    '''
    for i in range(trainSetCoords.shape[0]):
        if detectionResultsTrainSet.iloc[i, 1:].sum() != 0:
            if markerSquaresTrainSet[i].sum() == 0:
                trainSet = np.insert(trainSet, n, np.zeros((1, 8)), 0)
            n = n + 1
    '''
    np.save('data_'+m_id,trainSet)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("usage: python executeTestSet")
        print("     image folder        # path to the folder containing images")
        print("     total Number of Images in the folder containing images     ")
        print("     image suffix        # suffix of the image file")
        print("     Model Name          # Trained model name")
        print("\nExample: python executeTestSet.py images/ 1600 .png train\n")
        sys.exit()
    scaleFile(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
