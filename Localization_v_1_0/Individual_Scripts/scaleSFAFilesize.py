import numpy as np
import pandas as pd
import sys
from os import walk

'''
outputPath = '/hri/localdisk/ThesisProject/LabledImageResults/'
sourcePath = '/hri/localdisk/ThesisProject/'
label_id = 0 
train_test_data = False

trainSetCoordsPath = '/hri/localdisk/ThesisProject/MO_New/coordinates_wp0_Kodak.txt'
testSetCoordsPath = '/hri/localdisk/ThesisProject/M1/coordinates_wp0_Kodak.txt'
'''


def scaleFile(outputPath, train_detection, test_detection, label_id) :
    tags = [ 'Building_Corner' , 'Hut' , 'Garden_Entry' ]
    label_id = int(label_id)
    label_name = tags [ label_id ]

    # Train Set
    # Slow Feature Values
    trainSetPath = outputPath + 'train_' + label_name + '_slowFeatures.npy'

    # Load Files
    trainSet = np.load ( trainSetPath )
    '''
    imageSource_train = sourcePath + '/Extract_' + label_name + '/'

    f_train = [ ]

    for (dir_path , dir_names , filenames_train) in walk ( imageSource_train ) :
        f_train.extend ( filenames_train )
        break

    # clearing the part of .jpg from the string and converting it into int
    fileNumbers_train = [ ]

    for i in range ( 0 , len ( f_train ) ) :
        fileNumbers_train.append ( int ( f_train [ i ] [ :-4 ] ) )

    fileNumbers_train.sort ( )
    del f_train
    f_train = fileNumbers_train
    '''
    detectionResultsTrainSetPath = train_detection
    detectionResultsTrainSet = pd.read_csv (detectionResultsTrainSetPath)
    df = detectionResultsTrainSet[['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'confidence']].values
    df = pd.DataFrame(df)
    df_g = df[df[5] == label_id]
    im_array = np.array(df_g[0])
    img =[int(im_array[i][5:-4]) for i in range(0,len(im_array))]
    img = np.unique(img)
    img.sort()
    train_order = [img [i] - img [0] for i in range (0, len (img))]
    train_order_len = len(train_order)
    print(train_order_len)
    for i in range (0 , train_order_len) :
        if i != train_order[ i ] :
            train_order = np.insert ( train_order , i , i )
            print(i)
            trainSet = np.insert ( trainSet , i , np.zeros ( (1 , 8) ) , 0 )
            i = i - 1
    '''
    for i in range ( 0 , len ( fileNumbers_train ) ) :
        if i != f_train [ i ] :
            f_train = np.insert ( f_train , i , i )
            trainSet = np.insert ( trainSet , i , np.zeros ( (1 , 8) ) , 0 )
            i = i - 1
    '''
    np.save ( outputPath + 'data_' + label_name + '_train.npy' , trainSet )
    print ( label_name , " _trainSet :" , trainSet.shape )

    '''
#################################################################################

    imageSource_test = sourcePath + '/Extract_' + label_name + '/'

    f_test = [ ]
    for (dir_path , dir_names , filenames_test) in walk ( imageSource_test ) :
        f_test.extend ( filenames_test )
        break

    # clearing the part of .jpg from the string and converting it into int
    fileNumbers_test = [ ]

    for i in range ( 0 , len ( f_test ) ) :
        fileNumbers_test.append ( int ( f_test [ i ] [ :-4 ] ) )

    fileNumbers_test.sort ( )
    del f_train
    f_test = fileNumbers_test
    '''
    testSetPath = outputPath + 'test_' + label_name + '_slowFeatures.npy'
    testSet = np.load ( testSetPath )

    detectionResultsTestSetPath = test_detection
    detectionResultsTestSet = pd.read_csv ( detectionResultsTestSetPath )

    df = detectionResultsTestSet[['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'confidence']].values
    df = pd.DataFrame(df)
    df_label = df[df[5] == label_id]
    im_array = np.array(df_label[0])
    img =[int(im_array[i][5:-4]) for i in range(0,len(im_array))]
    img = np.unique(img)
    img.sort()
    test_order = [img [i] - img [0] for i in range (0, len (img))]
    test_order_len = len(test_order)
    for i in range (0 , test_order_len) :
        if i != test_order[ i ] :
            test_order = np.insert ( test_order , i , i )
            testSet = np.insert (testSet, i , np.zeros ( (1 , 8) ) , 0 )
            i = i - 1
    '''
    for i in range ( 0 , len ( fileNumbers_test ) ) :
        if i != f_test [ i ] :
            f_test = np.insert ( f_test , i , i )
            testSet = np.insert ( testSet , i , np.zeros ( (1 , 8) ) , 0 )
            i = i - 1
    '''
    np.save ( outputPath + '/data_' + label_name + '_test.npy' , testSet )
    print ( label_name , " _testSet :" , testSet.shape )


if __name__ == "__main__" :
    if len ( sys.argv ) != 5 :
        print ( "usage: python executeTestSet" )
        print ( "     image folder        # path to the folder containing images" )
        print ( "     total Number of Images in the folder containing images     " )
        print ( "     image suffix        # suffix of the image file" )
        print ( "     Model Name          # Trained model name" )
        print ( "\nExample: python executeTestSet.py images/ 1600 .png train\n" )
        sys.exit ( )
    scaleFile ( sys.argv [ 1 ] ,  sys.argv [ 2 ],sys.argv [ 3 ],int(sys.argv [ 4 ]))
