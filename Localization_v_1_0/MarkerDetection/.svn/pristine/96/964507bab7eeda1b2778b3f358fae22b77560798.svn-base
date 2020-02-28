import pandas as pd
import numpy as np
import cv2


marker_ids = [72, 136, 140, 272]

marker_info = pd.read_csv('marker_info.csv')

# print columns of marker info file
print "\ndata columns:"
print marker_info.columns


# remove path from image filenames contained in column 'image'
marker_info['image'] = marker_info['image'].apply(lambda x: x.split('/')[-1])


# read column with title 'image' and get the data as numpy array
image_files = marker_info['image'].values

print "\nimage file names:"
print image_files


# extract translation of marker 136 and get the data as numpy array
translation_columns_136 = [str(marker_ids[1]) + '_x', str(marker_ids[1]) + '_y', str(marker_ids[1]) + '_z']
translation_136 = marker_info[translation_columns_136].values

print "\ntranslation vectors:"
print translation_136


# get mask for valid estimations
valid_mask = abs(translation_136).sum(axis=1) != 0

print "\nmask for valid pose estimates:"
print valid_mask


# extract rotation of marker 136 and get the data as numpy array
rotation_columns_136 = [str(marker_ids[1]) + '_r1', str(marker_ids[1]) + '_r2', str(marker_ids[1]) + '_r3']
rotation_136 = marker_info[rotation_columns_136].values

print "\nrotation vectors:"
print rotation_136


# convert first rotation vector to rotation matrix
R, _ = cv2.Rodrigues(rotation_136[0])

print "\nrotation vector 1:"
print rotation_136[0]
print "\nrotation matrix 1:"
print R