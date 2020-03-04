import numpy as np
import pandas as pd
import sys


def scaleFile(outputPath, trainSetCoordsPath, testSetCoordsPath, aruco, train_test_data):
    m_id = aruco
    # Train Set
    # Slow Feature Values
    trainSetPath = outputPath + '/Evaluation_Arrays/' + 'train_' + m_id + '_slowFeatures.npy'

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
                trainCoords.append(0)
            else:
                trainCoords.append(1)
            n = n + 1
    trainCoords = np.array([trainCoords])
    np.save(outputPath+'/Evaluation_Arrays/data_'+m_id+'_train',trainSet)
    np.save(outputPath+'/Evaluation_Arrays/coords_'+m_id+'_train',trainCoords)

    if train_test_data:
        testSetPath = outputPath + '/Evaluation_Arrays/' + 'test_' + m_id + '_slowFeatures.npy'
        detectionResultsTestSetPath = outputPath + '/' + 'result_test.csv'
        testSet = np.load(testSetPath)
        testSetCoords = np.loadtxt(testSetCoordsPath, delimiter=',', usecols=(4, 5))
        detectionResultsTestSet = pd.read_csv(detectionResultsTestSetPath)
        markerSquaresTestSet = detectionResultsTestSet[
            [m_id + '_bb_x1', m_id + '_bb_y1', m_id + '_bb_x2', m_id + '_bb_y2', m_id + '_bb_x3', m_id + '_bb_y3',
             m_id +
             '_bb_x4', m_id + '_bb_y4']].values
        n = 0
        testCoords = []
        for i in range(testSetCoords.shape[0]):
            if detectionResultsTestSet.iloc[i, 1:].sum() != 0:
                if markerSquaresTestSet[i].sum() == 0:
                    testSet = np.insert(testSet, n, np.zeros((1, 8)), 0)
                n = n + 1
        testCoords = np.array([testCoords])
        np.save(outputPath+'/Evaluation_Arrays/data_' + m_id+'_test', testSet)
        np.save(outputPath+'/Evaluation_Arrays/coords_' + m_id + '_test', testCoords)

    print(m_id, " _trainSet :", trainSet.shape)
    print(m_id,"_trainCoords :",trainCoords.shape)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("usage: python executeTestSet")
        print("     image folder        # path to the folder containing images")
        print("     total Number of Images in the folder containing images     ")
        print("     image suffix        # suffix of the image file")
        print("     Model Name          # Trained model name")
        print("\nExample: python executeTestSet.py images/ 1600 .png train\n")
        sys.exit()
    scaleFile(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
