import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from Localization_v_1_0.Individual_Scripts.scaleSFAFilesize import scaleFile
import csv
# Globally declared arrays
testSet = []
testSet_1 = []
testSet_2 = []
test_flag = []
trainSet = []
'''
outputPath = '/hri/localdisk/ThesisProject/Kaushik/Kaushik/Testing_Sample_Script/10032020-083202/'
trainSetCoordsPath = '/hri/localdisk/ThesisProject/Kaushik/Kaushik/Testing_Sample_Script/10032020-083202/coordinates_train.txt'
testSetCoordsPath = '/hri/localdisk/ThesisProject/Kaushik/Kaushik/Testing_Sample_Script/10032020-083202/coordinates_test.txt'
aruco = [160]
m_id = str(aruco[0])
train_test_data = True
'''


def evaluate_distinct_data_performance(outputPath, trainSetCoordsPath, testSetCoordsPath,train_detection,test_detection,
                                       train_folder,test_folder):
    global testSet, testSet_1, testSet_2, test_flag, trainSet
    # Regression Degree
    DEG = 2
    tags = [ 'Building_Corner','Hut', 'Garden_Entry']
    # csv_path = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/Results/'
    csv_path = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/Jitter/'
    # label_name = tags[label_id]
    type_of_aruco_data = ['concatenated']

    for j in range(0,len(tags)):
        scaleFile(outputPath, train_detection,test_detection, j)

    trainSetPath_1 = outputPath + 'data_' + tags[0] + '_train.npy'
    trainSetPath_2 = outputPath + 'data_' + tags[1] + '_train.npy'
    trainSetPath_3 = outputPath + 'data_' + tags[2] + '_train.npy'
    trainSet_1 = np.load(trainSetPath_1)
    trainSet_2 = np.load(trainSetPath_2)
    trainSet_3 = np.load(trainSetPath_3)

    trainSet = np.concatenate((trainSet_1,trainSet_2,trainSet_3),axis =1)
    # Load Files
    trainSetCoords = np.loadtxt(trainSetCoordsPath, delimiter=',', usecols=(4, 5))

    print("\nActual Train coordinates\n ")
    print("Train Set Coords: ", trainSetCoords.shape)
    print("Train Set Coords after marker detection: ", trainSet.shape)

    # Test set
    testSetPath_1 = outputPath + '/data_' + tags [ 0 ] + '_test.npy'
    testSetPath_2 = outputPath + '/data_' + tags [ 1 ] + '_test.npy'
    testSetPath_3 = outputPath + '/data_' + tags [ 2 ] + '_test.npy'

    testSet_1 = np.load ( testSetPath_1 )
    testSet_2 = np.load ( testSetPath_2 )
    testSet_3 = np.load ( testSetPath_3 )

    testSet = np.concatenate ( (testSet_1 , testSet_2 , testSet_3) , axis = 1 )

    testSetCoords = np.loadtxt ( testSetCoordsPath , delimiter = ',' , usecols = (4 , 5) )

    print ("\nActual Test coordinates\n ")
    print ("Test Set Coords: ", testSetCoords.shape)
    print ("Test Set Coords after marker detection: ", testSet.shape)

    evaluate_multi_style_data_model ( outputPath , DEG , trainSet , trainSetCoords ,testSetCoords,testSet ,
                                      type_of_aruco_data [0], csv_path, train_folder, test_folder)


def evaluate_multi_style_data_model(outputPath, DEG, train_Set, trainCoords, testCoords, test_Set,
                                    data_style, csv_path, train_folder,test_folder):
    # Perform Regression
    #####################################################
    polyRegressor = PolynomialFeatures(degree=DEG)
    fig, ax = plt.subplots()

    polyFeatureTrainingSet = polyRegressor.fit_transform ( train_Set )
    polyFeaturesTestSet = polyRegressor.fit_transform ( test_Set )

    regressor_x = LinearRegression ( )
    regressor_x.fit ( polyFeatureTrainingSet , trainCoords [ : , 0 ] )  # x-coordinates

    regressor_y = LinearRegression ( )
    regressor_y.fit ( polyFeatureTrainingSet , trainCoords [ : , 1 ] )  # y-coordinates

    predicted_X = regressor_x.predict ( polyFeaturesTestSet )
    predicted_Y = regressor_y.predict ( polyFeaturesTestSet )

    prediction_X = predicted_X.reshape ( predicted_X.shape [ 0 ] , 1 )
    prediction_Y = predicted_Y.reshape ( predicted_Y.shape [ 0 ] , 1 )
    predictedCoordinates = np.hstack ( [ prediction_X , prediction_Y ] )
    # Calculate Error
    ######################################################
    MAE = mean_absolute_error ( testCoords , predictedCoordinates)
    print ("\nMean Absolute Error: %f [m]\n" % MAE)
    row_val = str(train_folder)+','+str(test_folder),str(MAE)
    with open(csv_path+'error.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row_val)

    # Visualize
    ######################################################
    ax.plot (testCoords [:, 0], testCoords [:, 1], 'b.', lw = 1, label = 'Ground truth')
    ax.plot (prediction_X, prediction_Y, 'r.', lw = 1, label = 'Estimation')

    plt.title('Mean Absolute Error: ' + "{:.4f}".format(MAE), fontweight='bold')
    legend = ax.legend(loc='lower right', shadow=True)
    plt.xlabel('X [m]', fontsize=16)
    plt.ylabel('Y [m]', fontsize=16)
    plt.tick_params(top='off', bottom='on', left='on', right='off', labelleft='on', labelbottom='on')
    plt.rc('font', weight='bold')
    plt.rc('legend', **{'fontsize': 8})
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=12)
    plt.savefig(outputPath + '/' + data_style + '_marker_result.pdf', dpi=1200, bbox_inches='tight')  # Save?
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
    evaluate_distinct_data_performance(sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4], sys.argv[5],
                                       int(sys.argv[6]), int(sys.argv[7]))
