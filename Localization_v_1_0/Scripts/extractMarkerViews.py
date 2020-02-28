import numpy as np
import pandas as pd
import sys
import os
import shutil
import math
import cv2
from tqdm import tqdm


def align_view(img, point, center, target_alpha=np.pi / 2):
    uv = point - center
    alpha = np.arctan2(uv[1], uv[0])
    delta = alpha - target_alpha
    delta %= np.pi * 2

    rows, cols = img.shape
    R = cv2.getRotationMatrix2D((point[0], point[1]), np.rad2deg(delta), 1)
    aligned_img = cv2.warpAffine(img, R, (cols, rows))
    return aligned_img


def extractMarkerViews(markerId, imageFileName, imagePath, imageSuffix
                       , numImages, savePath, extractedImageWidth=120, extractedImageHeight=120):
    m_id = markerId
    detection_results = pd.read_csv(imageFileName)
    marker_squares = detection_results[
        [m_id + '_bb_x1', m_id + '_bb_y1', m_id + '_bb_x2', m_id + '_bb_y2', m_id + '_bb_x3', m_id + '_bb_y3', m_id +
         '_bb_x4', m_id + '_bb_y4']].values

    w = extractedImageWidth
    h = extractedImageHeight

    # Create output folder
    if os.path.isdir(savePath):
        shutil.rmtree(savePath)
    os.mkdir(savePath)

    counter = 0
    for i in tqdm(range(numImages)):
        # print(i)

        if marker_squares[i].sum() != 0:
            # img = cv2.imread(imagePath + str(i) + imageSuffix, 0)
            img = cv2.imread(detection_results[['image']].values[i][0], 0)

        points = np.zeros((4, 2))
        points[0] = marker_squares[i, 0:2]
        points[1] = marker_squares[i, 2:4]
        points[2] = marker_squares[i, 4:6]
        points[3] = marker_squares[i, 6:8]

        # print(points)
        # print('')

        cx = int(points[:, 0].mean() + 0.5)
        cy = int(points[:, 1].mean() + 0.5)

        side = (math.sqrt((points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2))

        # print("cx: %d, cy: %d" % (cx, cy))
        # print("points[2][0] %d, points[2][1] %d" % (points[2][0], points[2][1]))
        # print("Width : %d" % (math.sqrt((points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2)))
        img = align_view(img, np.array([cx, cy]), np.array([1440, 1440]))

        marker = img[int(cy - 10 + side / 2):int(cy + 10 + 3 * side / 2),
                 int(cx - 10 - side / 2):int(cx + 10 + side / 2)]
        # marker = img[int(cy - 10 - side / 2):int(cy + 10 + side/2 ) , int(cx - 10 - 3*side / 2 ):int(cx - 10 - side
        # / 2 )]

        # Display the extracted image?
        # temp = cv2.resize(img, (1000,1000))
        # cv2.imshow('marker', temp)
        # if cv2.waitKey() & 255 == 27:
        # break

        if marker.shape[0] != 0 and marker.shape[1] != 0:
            marker = cv2.resize(marker, (120, 120))
            cv2.imwrite(savePath + str(counter) + imageSuffix, marker)
            counter += 1


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("usage: python extractMarkerViews")
        print("     markerID                        # ")
        print("     marker detection results        # file name")
        print("     image path                      # path to image folder")
        print("     image suffix                    # ")
        print("     total number of images          # ")
        print("     outputPath                      # Save extracted views to the specified loc.")

        print("\nExample: python extractMarkerViews.py 144 result images/ .png 11 extracted/ \n")
        print(
            "\n python extractMarkerViews.py 144 "
            "'/hri/localdisk/ThesisProject/Kaushik/Version_1_2/Landmark_based_localization/Output/result_train.csv' "
            "'/hri/localdisk/ThesisProject/Kaushik/Simulator_data/SingleMarker/train/' .png 11 "
            "'/hri/localdisk/ThesisProject/Kaushik/Simulator_data/SingleMarker/Sample_Extracted_train/' \n")

        sys.exit()
    extractMarkerViews(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
