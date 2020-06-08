import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to images
basePath = '/hri/localdisk/ThesisProject/MO_New/panoramicImages/'
detectionResultsPath = '/hri/localdisk/ThesisProject/MO_New/Detection_Results.csv'

# Path for saving Extracted Images
savePath = ['/hri/localdisk/ThesisProject/MO_New/Extract_Building_Corner/','/hri/localdisk/ThesisProject/MO_New/Extract_Hut/','/hri/localdisk/ThesisProject/MO_New/Extract_Garden_Entry/']
# Read image names in ascending order
imageList = os.listdir ( basePath )
imageList.sort ( key = lambda fileName : int ( filter ( str.isdigit , fileName ) ) )

# Read Detection Results
coords = pd.read_csv ( detectionResultsPath ,
                       usecols = [ 'image' , 'xmin' , 'ymin' , 'xmax' , 'ymax' , 'label' , 'confidence' ] )

# Object tags
tags = [ 'Building_Corner','Hut', 'Garden_Entry']
colors = [ (255 , 0 , 0) , (0 , 255 , 0) , (0 , 0 , 255) ]

trackJitter = [ ]
visualizeBoundingBoxes = True
counter = 0
imageSuffix = '.jpg'
for i in range ( len ( imageList ) ) :
    img = cv2.imread ( basePath + imageList [ i ] , 0 )
    labels = coords [ coords [ 'image' ] == imageList [ i ] ]

    print ( ' ' )
    print ( imageList [ i ] )
    for l in range ( len ( tags ) ) :
        label = labels [ labels [ 'label' ] == l ]

        if len ( label ) == 0 :  # Object not detected in the current image.
            trackJitter.append ( [ l , 0 , 0 , 0 ] )  # objectId, xCenter, yCenter , numberofPixels
            continue
        elif len ( label ) > 1 :  # Double detections: Reject bounding boxes with low confidence.
            label = label [ label [ 'confidence' ] == label [ 'confidence' ].max ( ) ]

        x1 , y1 , x2 , y2 = int ( label [ 'xmin' ].iloc [ 0 ] ) , int ( label [ 'ymin' ].iloc [ 0 ] ) , int (
        label [ 'xmax' ].iloc [ 0 ] ) , int ( label [ 'ymax' ].iloc [ 0 ] )
        trackJitter.append ( [ l , (x1 + x2) / 2 , (y1 + y2) / 2 ,
                               (x2 - x1) * (y2 - y1) ] )  # objectId, xCenter, yCenter , numberofPixels
        marker = img[y1:y2, x1:x2]
        marker = cv2.resize(marker, (120, 120))
        print ( imageList [ i ] , ', label ',l)
        cv2.imwrite(savePath[l] + str(i) + imageSuffix, marker)

