import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plot
import seaborn
outputPath = 'D:/Localization_SimulatedData-master/Localization_v_2_0/gitUpload_160_side'
trainSetCoordsPath = 'D:/Localization_SimulatedData-master/Localization_v_2_0/gitUpload_160_side/coordinates_train.txt'
testSetCoordsPath = 'D:/Localization_SimulatedData-master/Localization_v_2_0/gitUpload_160_side/coordinates_test.txt'
m_id = '160'
train_test_data = True
test_160 = 'D:/Localization_SimulatedData-master/Localization_v_2_0/gitUpload_160_side/result_test.csv'
train_160 = 'D:/Localization_SimulatedData-master/Localization_v_2_0/gitUpload_160_side/result_train.csv'

trainSetCSV = train_160

detectionResultsTrainSet = pd.read_csv (trainSetCSV)
data = detectionResultsTrainSet[[m_id + '_bb_x1' , m_id + '_bb_y1' , m_id + '_bb_x2' , m_id + '_bb_y2' , m_id + '_bb_x3' , m_id + '_bb_y3' , m_id +
      '_bb_x4' , m_id + '_bb_y4']].values
	  
dataset = pd.DataFrame({'x1': data[:, 0],'y1': data[:, 1],'x2': data[:, 2],'y2': data[:, 3],'x3': data[:, 4],'y3': data[:, 5],'x4': data[:, 6],'y4': data[:, 7]})
dataset
dataset['C'] = np.arange(len(dataset))
dataset['total_sum'] = dataset[["x1", "y1", "x2","y2","x3", "y3", "x4","y4"]].sum(axis=1)
dataset.loc[dataset['total_sum']>0,'bool']=1
dataset['cum_sum'] = dataset['bool'].cumsum()
dataset

dataset.plot( x='C', y='cum_sum')

plt.hist(dataset['total_sum'], 20,
         density=True,
         histtype='bar',
         facecolor='b',
         alpha=0.5)

plt.show()
	  