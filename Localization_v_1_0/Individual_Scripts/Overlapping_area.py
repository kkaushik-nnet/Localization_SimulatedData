import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import random
import math as m
# Path to images
train_folder = '03'
train_folder_e = '303'
output_path_no_jitter = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/'
output_path_jitter = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/Jitter/'

basePath = '/hri/storage/user/haris/Kaushik/panoramicImages/Test_Image_Detection_Results/'
detectionResultsPath = '/hri/storage/user/haris/Kaushik/panoramicImages/hut_csvs/Test_Results_csv/Detection_Results_.csv'

Area_csv = '/hri/storage/user/haris/Kaushik/panoramicImages/hut_csvs/Test_Results_csv/'

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


for i in range ( len ( imageList ) ) :
    img = cv2.imread ( basePath + imageList [ i ] , 0 )
    labels = coords [ coords [ 'image' ] == imageList [ i ] ]

    y_max_label_0 = 0
    y_max_label_1 = 0
    x_max_label_0 = 0
    x_max_label_1 = 0

    y_min_label_0 = 0
    y_min_label_1 = 0
    x_min_label_0 = 0
    x_min_label_1 = 0

    # print ( imageList [ i ] + '  - ')
    # print ( )

    for l in range ( len ( tags ) ) :
        label = labels [ labels [ 'label' ] == l ]
        # jitter = 40
        if len ( label ) == 0 :  # Object not detected in the current image.
            trackJitter.append ( [ l , 0 , 0 , 0 ] )  # objectId, xCenter, yCenter , numberofPixels
            # print(labels,': ',imageList[i])
            continue
        elif len ( label ) > 1 :  # Double detections: Reject bounding boxes with low confidence.
            label = label [ label [ 'confidence' ] == label [ 'confidence' ].max ( ) ]
            print ( '' )
            # print ( imageList [ i ] + '  ########## ' )
            # print ( imageList [ i ] + '  - ' + savePath [ l ] )
        # Add jitter after each term
        x1 , y1 , x2 , y2 = int ( label [ 'xmin' ].iloc [ 0 ] ), int (
            label [ 'ymin' ].iloc [ 0 ] ) , int (
            label [ 'xmax' ].iloc [ 0 ] ) , int ( label [ 'ymax' ].iloc [ 0 ] )

        trackJitter.append ( [ l , (x1 + x2) / 2 , (y1 + y2) / 2 ,
                               (x2 - x1) * (y2 - y1) ] )  # objectId, xCenter, yCenter , numberofPixels

        if label['label'].iloc [ 0 ] == 0 :
            x_min_label_0 = x1
            x_max_label_0 = x2
            y_min_label_0 = y1
            y_max_label_0 = y2
        elif label['label'].iloc [ 0 ] == 1 :
            x_min_label_1 = x1
            x_max_label_1 = x2
            y_min_label_1 = y1
            y_max_label_1 = y2

    if labels [ 'label' ].nunique ( ) == 2 :
        y_max = max ( y_max_label_0 , y_max_label_1 )
        y_min = min ( y_min_label_0 , y_min_label_1 )
        x_max = max ( x_min_label_1 , x_max_label_0 )
        x_min = min ( x_min_label_1 , x_max_label_0 )

        # diagonal of a square is d
        d = m.sqrt ( pow ( x_max - x_min , 2 ) + pow ( y_max - y_min , 2 ) )

        # area is a
        a = m.sqrt ( pow ( d , 2 ) / 2 )
        if x_min_label_1 < x_max_label_0 :
            a = -a
    else:
        a = 0
    print ( imageList [ i ] + '  - ' +str(a))

    row_val = str(imageList [ i ]),str(a)
    with open(Area_csv+'Area.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row_val)
    '''
    # saving marker boundary box values into to extract the object inbetween
        marker = img [ y1 :y2 , x1 :x2 ]

        # resizing the object to 120*120
        marker = cv2.resize ( marker , (120 , 120) )

        # print ( imageList [ i ] , ', label ',l)

        # Save the labeled images to their respective paths
        cv2.imwrite ( savePath [ l ] + imageList [ i ] , marker )
    '''