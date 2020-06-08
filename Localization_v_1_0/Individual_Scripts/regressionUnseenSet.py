import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pylab


# sns.set()
# sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})

def calArrayMean(num , array , k) :
    X_mean = [ ]
    for i in range ( num ) :
        offset = i * k
        arr = array [ offset : (offset + k) ]
        mean_ = arr.mean ( axis = 0 )
        X_mean.append ( mean_ )

    X_mean = np.array ( X_mean )
    return X_mean


###################################################
# Load SFA matrix
trainingSet = np.load ( '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/Results/Results_2_1/train_Garden_Entry_slowFeatures.npy' )
testSet = np.load ( '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/Results/Results_2_1/test_Garden_Entry_slowFeatures.npy' )

print ( trainingSet.shape )
print ( testSet.shape )

combinedCoordinates = np.loadtxt ( '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/02/coordinates_wp0.txt' , delimiter = ',' ,
                                   usecols = (4 , 5) )

## Load test coordinates
# testCoordinates = np.loadtxt('coordinates_test.txt')
testCoordinates = np.loadtxt ( '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/01/coordinates_wp0.txt' , delimiter = ',' ,
                               usecols = (4 , 5) )

## Perform Regression
polyRegressor = PolynomialFeatures ( degree = 2 )
###################################################

polyFeatureTrainingSet = polyRegressor.fit_transform ( trainingSet )
polyFeaturesTestSet = polyRegressor.fit_transform ( testSet )

regressor_x = LinearRegression ( )
regressor_x.fit ( polyFeatureTrainingSet , combinedCoordinates [ : , 0 ] )  # x-coordinates

regressor_y = LinearRegression ( )
regressor_y.fit ( polyFeatureTrainingSet , combinedCoordinates [ : , 1 ] )  # Y_coordinates

predicted_X = regressor_x.predict ( polyFeaturesTestSet )
predicted_Y = regressor_y.predict ( polyFeaturesTestSet )

RMSE_x = mean_squared_error ( testCoordinates [ : , 0 ] , predicted_X ) ** 0.5
RMSE_y = mean_squared_error ( testCoordinates [ : , 1 ] , predicted_Y ) ** 0.5

prediction_X = predicted_X.reshape ( predicted_X.shape [ 0 ] , 1 )
prediction_Y = predicted_Y.reshape ( predicted_Y.shape [ 0 ] , 1 )
predictedCoordinates = np.hstack ( [ prediction_X , prediction_Y ] )

distance = np.linalg.norm ( predictedCoordinates - testCoordinates , axis = 1 )
MAE = distance.mean ( )
print ( "Mean Euclidean Distance: " + str ( MAE ) )
print ( "Median Error: " + str ( np.median ( distance ) ) )

'''
## Visualize
fig, ax = plt.subplots()
ax.axis('equal')
ax.plot(testCoordinates[:,0], testCoordinates[:,1], 'b.', lw=1, label='Ground truth')
ax.plot(prediction_X, prediction_Y, 'r.', lw=1 ,label='Estimation')

plt.title('Test Set -- '+'Mean Euclidean Distance: ' + "{:.2f}".format(MAE), fontsize=16)#,fontweight='bold')
plt.title('(Median | Mean Performance: ' + "{:.2f}".format(np.median(distance)) + " , "+ "{:.2f}".format(np.mean(distance))+ ' [m])')
legend = ax.legend(loc='lower left', shadow=True)
plt.xlabel('X ')# + '(RMSE: ' + "{:.2f}".format(RMSE_x) + ')')
plt.ylabel('Y ')# + '(RMSE: ' + "{:.2f}".format(RMSE_y) + ')')
plt.tick_params(top='off', bottom='on', left='on', right='off', labelleft='on', labelbottom='on')
plt.rc('font', weight='bold')
plt.rc('legend',**{'fontsize':8})
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(labelsize=12)
plt.savefig('Different_holo.png')
plt.savefig('Simulator.pdf', dpi=1200, bbox_inches='tight')
plt.show()
'''

# fig2, ax2 = plt.subplots()
# ax2.set_aspect('equal')
# plt.plot(testCoordinates[:,0], testCoordinates[:,1], 'b.', label='Test Pts')
# plt.plot(prediction_X, prediction_Y, 'r.', label='Predicted Pts')
# ax2.legend(loc='lower left', shadow=True)
# plt.xlabel('X ')
# plt.ylabel('Y ')
# plt.title('Mean Error [m]: ' + "{:.2f}".format(MAE))
# coordinateList = list()
# for j in range(testCoordinates.shape[0]):
# coordinateList.append((testCoordinates[j, 0], prediction_X[j]))
# coordinateList.append((testCoordinates[j, 1], prediction_Y[j]))
# coordinateList.append('k')

# pylab.plot(*coordinateList)
##plt.savefig('BC.png')
# plt.show()
