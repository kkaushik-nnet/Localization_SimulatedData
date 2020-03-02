import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import pandas as pd


def removeDuplicateRows(filename):
    with open(filename, 'r') as csvfile:
        Set = list(csv.reader(csvfile))

    print("Total number of rows: {}".format(len(Set)))
    x = Set[1][4]
    y = Set[1][5]
    counter = 1
    duplicates = 0

    for i in range(1, len(Set) - 1):
        if x == Set[counter + 1][4] and y == Set[counter + 1][5]:
            Set.remove(Set[counter + 1])
            duplicates += 1
        else:
            x = Set[counter + 1][4]
            y = Set[counter + 1][5]
            counter += 1

    print("Total number of duplicate rows: {}".format(duplicates))
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(Set)


def duplicates_deletion():
    # Set paths
    filePath_wp0 = 'coordinates_wp0_Kodak.txt'
    # filePath_wp1 = 'coordinates_wp1_Kodak.txt'

    coordinate = np.loadtxt(filePath_wp0, delimiter=',', usecols=(4, 5))
    # coordinate1 = np.loadtxt(filePath_wp1,delimiter=',',usecols=(4,5))

    removeDuplicateRows(filePath_wp0)
    # removeDuplicateRows(filePath_wp1)

    plt.plot(coordinate[:, 0], coordinate[:, 1], 'b-', label='working_period_0')
    # plt.plot(coordinate1[:,0],coordinate1[:,1],'r-',label = 'working_period_1')
    plt.legend()
    plt.axis('equal')
    # plt.savefig('Recording_06.png')
    plt.show()

