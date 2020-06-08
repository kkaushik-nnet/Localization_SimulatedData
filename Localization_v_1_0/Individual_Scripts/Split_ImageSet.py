import numpy as np
import random as rd
from os import walk
import shutil
import pandas as pd
import csv

complete_hut_data = '/hri/localdisk/ThesisProject/MO_New/Extract_Hut/'
training_Hut = '/hri/localdisk/ThesisProject/MO_New/Hut_data_train/'
testing_Hut = '/hri/localdisk/ThesisProject/MO_New/Hut_data_test/'
f = []
for (dir_path, dir_names, filenames) in walk( complete_hut_data ):
    f.extend(filenames)
    break

g = []
for i in range(0,len(f)):
    g.append(int(f[i][:-4]))


def move_coordinates_files(folder_path, train_folder, test_folder):
    # Train set
    for i in range(0,len(train_set)):
        shutil.copy( folder_path + train_set[i], train_folder)
    # Test set
    for i in range(0,len(test_set)):
        shutil.copy( folder_path + test_set[i], test_folder)


train_set = f
test_set = []
l = len(train_set)
r = int(np.floor(30*l/100))
for i in range(0,r):
    index = rd.randrange(1,len(train_set))
    test_set.append(train_set[index-1])
    del train_set[index-1]

move_coordinates_files(complete_hut_data, training_Hut, testing_Hut)

data = '/home/kkaushik/Interpretations/coordinates.txt'
data = pd.read_csv ( data , header = 0 )
for i in range ( 0 , len ( data [ 'filename' ] ) ) :
    data.at [ i , 'filename' ] = i
    int ( data [ 'filename' ] [ i ] )

with open('coordinates_train.csv', 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(data.columns)
    for i in range(0, len(train_set)):
        writer.writerow()

print(np.sort(train_set))
print('############################################################################################3')
print(np.sort(test_set))
print(len(test_set))
print(len(train_set))
print(training_Hut+ train_set[1])
print(g)
