import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from os import walk
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

testSet = []
testCoords = []


def evaluate_individual_performances(outputPath, trainSetCoordsPath, testSetCoordsPath, detection_train_csv,
                                     detection_test_csv,label_id):
    global testSet, testCoords
    tags = ['Building_Corner', 'Hut', 'Garden_Entry']
    label_name = tags[label_id]

    DEG = 2
    # Train Set Slow Feature Values
    trainSetPath = outputPath + 'train_' + label_name + '_slowFeatures.npy'
    trainSet = np.load(trainSetPath)

    # Test Set Slow Feature Values
    testSetPath = outputPath + 'test_' + label_name + '_slowFeatures.npy'
    testSet = np.load(testSetPath)

    # Load Coordinates
    trainSetCoords = np.loadtxt(trainSetCoordsPath, delimiter = ',', usecols = (4, 5))
    testSetCoords = np.loadtxt(testSetCoordsPath, delimiter = ',', usecols = (4, 5))

    # DetectionResults
    detectionResultsTrainSetPath = detection_train_csv
    detectionResultsTestSetPath = detection_test_csv
    # '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/01/Detection_Results.csv'

    detectionResultsTrainSet = pd.read_csv (detectionResultsTrainSetPath)
    detectionResultsTestSet = pd.read_csv (detectionResultsTestSetPath)

    '''
    # saving the exact file numbers present in the folder
    f_train = []
    for (dir_path, dir_names, filenames_train) in walk(imageSource_train):
        f_train.extend(filenames_train)
        break
    # clearing the part of omni_*.jpg from the string and converting it into int
    fileNumbers_train = []
    for i in range(0, len(f_train)):
        fileNumbers_train.append(int(f_train[i][5:-4]))
    fileNumbers_train.sort()
    print(len(fileNumbers_train))
    print(fileNumbers_train)
    trainCoords = [trainSetCoords[fileNumbers_train[i]] for i in range(0,len(fileNumbers_train)-1)]
    trainCoords = np.array(trainCoords)
    '''

    df = detectionResultsTrainSet[['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'confidence']].values
    df = pd.DataFrame(df)
    df_g = df[df[5] == label_id]
    im_array = np.array(df_g[0])
    img =[int(im_array[i][5:-4]) for i in range(0,len(im_array))]
    img = np.unique(img)
    img.sort()
    train_order = [img [i] - img [0] for i in range (0, len (img))]
    '''
    Coords_order = []
    for i in range(0,len(img)-1):
        if img [i + 1] - img [i] == 1:
            Coords_order.append (i)
    Coords_order.append (len (img))
    #  Coords_order = [img[i]-img[0] for i in range(0,len(img)) if img[i+1]-img[i]==1]
    #  trainCoords.append(len(img))
    '''
    trainCoords = [trainSetCoords[train_order[i]] for i in range(0,len(train_order))]
    trainCoords = np.array (trainCoords)
    print("Train Set Coords ", trainCoords.shape)
    print("Train Set SlowFeatures", trainSet.shape)

    '''
    # saving the exact file numbers present in the folder
    f_test = []
    for (dir_path, dir_names, filenames_test) in walk(imageSource_test):
        f_test.extend(filenames_test)
        break

    # clearing the part of .jpg from the string and converting it into int
    fileNumbers_test = []
    for i in range(0, len(f_test)):
        fileNumbers_test.append(int(f_test[i][5:-4]))
    fileNumbers_test.sort()
    print(len(fileNumbers_test))
    print(fileNumbers_test)
    testCoords = [testSetCoords[fileNumbers_test[i]] for i in range(0, len(fileNumbers_test))]
    testCoords = np.array(testCoords)
    '''

    df = detectionResultsTestSet[['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'confidence']].values
    df = pd.DataFrame(df)
    df_label = df[df[5] == label_id]
    im_array = np.array(df_label[0])
    img =[int(im_array[i][5:-4]) for i in range(0,len(im_array))]
    img = np.unique(img)
    img.sort()
    test_order = [img [i] - img [0] for i in range (0, len (img))]

    testCoords = [testSetCoords[test_order[i]] for i in range(0,len(test_order))]
    testCoords = np.array (testCoords)

    print("Test Set Coords ", testCoords.shape)
    print("Test Set SlowFeatures", testSet.shape)

    # Perform Regression
    #####################################################
    polyRegressor = PolynomialFeatures(degree = DEG)
    fig, ax = plt.subplots()

    polyFeatureTrainingSet = polyRegressor.fit_transform(trainSet)
    polyFeaturesTestSet = polyRegressor.fit_transform(testSet)

    regressor_x = LinearRegression()
    regressor_x.fit(polyFeatureTrainingSet, trainCoords[:, 0])  # x-coordinates

    regressor_y = LinearRegression()
    regressor_y.fit(polyFeatureTrainingSet, trainCoords[:, 1])  # y-coordinates

    predicted_X = regressor_x.predict(polyFeaturesTestSet)
    predicted_Y = regressor_y.predict(polyFeaturesTestSet)

    prediction_X = predicted_X.reshape(predicted_X.shape[0], 1)
    prediction_Y = predicted_Y.reshape(predicted_Y.shape[0], 1)
    predictedCoordinates = np.hstack([prediction_X, prediction_Y])
    # Calculate Error
    ######################################################
    MAE = mean_absolute_error(testCoords, predictedCoordinates)
    print("\nMean Absolute Error: %f [m]\n" % MAE)
    # Visualize
    ######################################################
    ax.plot(testCoords[:, 0], testCoords[:, 1], 'b.', lw = 1, label = 'Ground truth')
    ax.plot(prediction_X, prediction_Y, 'r.', lw = 1, label = 'Estimation')

    plt.title('Mean Absolute Error: ' + "{:.4f}".format(MAE), fontweight = 'bold')
    legend = ax.legend(loc = 'lower right', shadow = True)
    plt.xlabel('X [m]', fontsize = 16)
    plt.ylabel('Y [m]', fontsize = 16)
    plt.tick_params(top = 'off', bottom = 'on', left = 'on', right = 'off', labelleft = 'on',
                      labelbottom = 'on')
    plt.rc('font', weight = 'bold')
    plt.rc('legend', **{'fontsize': 8})
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.tick_params(labelsize = 12)
    plt.savefig(outputPath + '/' + label_name + '_result.pdf', dpi = 1200, bbox_inches = 'tight')  # Save?
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("usage: python executeTestSet")
        print("     image folder        # path to the folder containing images")
        print("     total Number of Images in the folder containing images     ")
        print("     image suffix        # suffix of the image file")
        print("     Model Name          # Trained model name")
        print("\nExample: python executeTestSet.py images/ 1600 .png train\n")
        sys.exit()
    evaluate_individual_performances(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                                       int(sys.argv[6]))
