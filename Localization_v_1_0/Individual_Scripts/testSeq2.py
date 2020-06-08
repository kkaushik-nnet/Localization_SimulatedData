import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import random

# Path to images
# train_folder = '03'
# train_folder_e = '303'
output_path_no_jitter = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/'
output_path_jitter = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/Jitter/'
'''
train = ['09']
train_e = ['909']
'''

train = [ '07' , '08' , '09' ]
train_e = [ '707' , '808' , '909' ]
'''
train = ['010','011']
train_e = ['1010','1111']
'''
for i in range ( 0 , len ( train ) ) :
    range_val = 10
    seq_gap = 2
    x_limit_l = 0
    x_limit_u = range_val
    y_limit_l = -range_val
    y_limit_u = range_val

    train_folder = train [ i ]
    train_folder_e = train_e [ i ]

    basePath = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/' + train_folder + '/WP_1200/'
    detectionResultsPath = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/' + train_folder + '/Detection_Results.csv'
    '''
    basePath = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/'+train_folder+'/panoramicImages/'
    detectionResultsPath = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/'+train_folder+'/Detection_Results.csv'
    '''
    outputPath = output_path_jitter + train_folder_e + '/'
    # Path for saving Extracted Images
    savePath = [ output_path_jitter + train_folder_e + '/Extract_Building_Corner/' ,
                 output_path_jitter + train_folder_e + '/Extract_Hut/' ,
                 output_path_jitter + train_folder_e + '/Extract_Garden_Entry/' ]

    # Read image names in ascending order
    imageList = os.listdir ( basePath )
    imageList.sort ( key = lambda fileName : int ( filter ( str.isdigit , fileName ) ) )

    # Read Detection Results
    coords = pd.read_csv ( detectionResultsPath ,
                           usecols = [ 'image' , 'xmin' , 'ymin' , 'xmax' , 'ymax' , 'label' , 'confidence' ] )

    # Object tags
    tags = [ 'Building_Corner' , 'Hut' , 'Garden_Entry' ]
    colors = [ (255 , 0 , 0) , (0 , 255 , 0) , (0 , 0 , 255) ]

    trackJitter = [ ]
    visualizeBoundingBoxes = True
    counter = 0
    imageSuffix = '.jpg'

    ###################### Making a sequence for jitter #############

    x0 = x_limit_l
    x1 = x0 + seq_gap
    x0_array = [ ]

    y0 = y_limit_l
    y1 = y0 + seq_gap
    y0_array = [ ]
    for i in range ( len ( imageList ) ) :
        img = cv2.imread ( basePath + imageList [ i ] , 0 )
        labels = coords [ coords [ 'image' ] == imageList [ i ] ]

        if (x1 < x_limit_u) & (x1 > x_limit_l) :
            if (x1 - x0) == seq_gap :
                x1 = x1 + seq_gap
                x0 = x0 + seq_gap
            elif (x1 - x0) == -seq_gap :
                x1 = x1 - seq_gap
                x0 = x0 - seq_gap
            x0_array.append ( x0 )
        if x1 == x_limit_u :
            x1 = x1 - seq_gap
            x0 = x0 + seq_gap
            x0_array.append ( x0 )
        if x1 == x_limit_l + seq_gap :
            x1 = x1 + seq_gap
            x0 = x0 - seq_gap
            x0_array.append ( x0 )

        if (y1 < y_limit_u) & (y1 > y_limit_l) :
            if (y1 - y0) == seq_gap :
                y1 = y1 + seq_gap
                y0 = y0 + seq_gap
            elif (y1 - y0) == -seq_gap :
                y1 = y1 - seq_gap
                y0 = y0 - seq_gap
            y0_array.append ( y0 )
        if y1 == y_limit_u :
            y1 = y1 - seq_gap
            y0 = y0 + seq_gap
            y0_array.append ( y0 )
        if y1 == y_limit_l + seq_gap :
            y1 = y1 + seq_gap
            y0 = y0 - seq_gap
            y0_array.append ( y0 )

        # Set Jitter Value
        jitter_x = x0_array [ -1 ]
        jitter_y = y0_array [ -1 ]
        print ( '  - jitter_x :' + str ( jitter_x ) + ' , jitter_y : ' + str ( jitter_y ) )
