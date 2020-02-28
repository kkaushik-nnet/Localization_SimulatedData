
from createSFANetwork import *
import numpy as np
import pickle
import sys
from PIL import Image
import os

root  = '/hri/localdisk/ThesisProject/Kaushik/Kaushik/Extracted_2801/'


def train_test_splitting(root):
    
    X = numpy.array([])
    y = list()

    for path, subdirs, files in os.walk(root):
        for name in files:
           img_path = os.path.join(path,name)
           correct_cat = categories[img_path]
           img_pixels = list(Image.open(img_path).gtedata())
           X = numpy.vstack((X, img_pixels))
           y.append(correct_cat)
           print(y)

    X_train, X_test, y_train, y_test = train_testsplit(X,y)
    print(y)
       	



