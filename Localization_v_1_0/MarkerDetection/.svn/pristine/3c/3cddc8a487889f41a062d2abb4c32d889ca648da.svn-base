#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path
import sys
import numpy as np


def create_image_list(path_directory, output_file, image_suffix):
    if not os.path.exists(path_directory):
        print 'error: directory ' + path_directory + ' does not exist'
        sys.exit()
    
    if not path_directory[-1] == '/':
        path_directory += '/'
    
    # create an array of all files in directory with image suffix
    image_files = np.array([f for f in os.listdir(path_directory) if f.endswith(image_suffix)])
    if len(image_files) == 0:
        print 'error: ' + path_directory + ' does not contain any image files with suffix ' + image_suffix
        sys.exit()
        
    # extract image ids from filenames - should handle all cases except other numbers than the id occur in filename
    image_ids = []
    for i in xrange(len(image_files)):
        id_as_string = ''
        for j in xrange(len(image_files[i])):
            if image_files[i][j].isdigit():
                id_as_string += image_files[i][j]
        if id_as_string == '':
            print 'error: image ' + image_files[i] + ' does not contain an id'
            sys.exit()
        image_ids.append(int(id_as_string))
    
    # sort image file names by their id
    sorted_idx = np.argsort(image_ids)
    image_files = image_files[sorted_idx]
    
    # write image list to output file
    with open(output_file, 'w') as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<opencv_storage>\n')
        fh.write('<images>\n')
        for f in image_files:
            fh.write('"' + path_directory + f + '"\n')
        fh.write('</images>\n')
        fh.write('</opencv_storage>\n')
    

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "usage: python create_image_list"
        print "     image folder        # path to the folder containing images"
        print "     output file name    # name of the resulting file containing the image list"
        print "     image suffix        # suffix of the image file"
        sys.exit()
    create_image_list(sys.argv[1], sys.argv[2], sys.argv[3])