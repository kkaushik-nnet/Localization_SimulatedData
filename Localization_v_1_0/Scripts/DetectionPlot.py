import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd


def detection_plot(outputPath,trainSetCoordsPath,detectionResultsTrainSetPath,m_id,var_type):

    trainSetCoords = np.loadtxt(trainSetCoordsPath, delimiter=',', usecols=(4, 5))

    detectionResultsTrainSet = pd.read_csv(detectionResultsTrainSetPath)
    markerSquaresTrainSet = detectionResultsTrainSet[
        [m_id + '_bb_x1', m_id + '_bb_y1', m_id + '_bb_x2', m_id + '_bb_y2', m_id + '_bb_x3', m_id + '_bb_y3', m_id +
         '_bb_x4', m_id + '_bb_y4']].values

    # Keep only the coordinates with valid marker detection (Train/Test Set)
    #########################################################################

    trainCoords = []
    unDetectedCoords = []
    for i in range(trainSetCoords.shape[0]):
        if (markerSquaresTrainSet[i][0:4].sum() != 0) and (markerSquaresTrainSet[i][4:8].sum() != 0):
            trainCoords.append(trainSetCoords[i, :])
        else:
            unDetectedCoords.append(trainSetCoords[i, :])

    trainCoords = np.array(trainCoords)
    unDetectedCoords = np.array(unDetectedCoords)
    # testCoords = np.array(testCoords)

    print("\nActual coordinates\n ")
    print("Train Set: (%d,%d) " % trainSetCoords.shape)

    print("\nAfter removing coordinates with no marker detection\n ")

    print("Train Set: (%d,%d) " % trainCoords.shape)

    percent = str(round(100 * len(unDetectedCoords) / len(trainCoords), 2))
    ## Visualize
    ######################################################
    fig, ax = plt.subplots()

    ax.plot(trainCoords[:, 0], trainCoords[:, 1], 'b-', lw=1, label='Ground truth')
    if len(unDetectedCoords) == 0:
        print("All coordinates detected")
    else:
        ax.plot(unDetectedCoords[:, 0], unDetectedCoords[:, 1], 'r.', label='Missed_' + m_id + '  ' + percent + '%')
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel('X [m]', fontsize=16)
    plt.ylabel('Y [m]', fontsize=16)
    plt.tick_params(top='off', bottom='on', left='on', right='off', labelleft='on', labelbottom='on')
    # plt.rc('font', weight='bold')
    plt.rc('legend', **{'fontsize': 6})
    plt.gca().set_aspect('equal')  # , adjustable='box'
    plt.tick_params(labelsize=12)
    plt.savefig(outputPath+'/' + 'single_marker_result_'+m_id+'_'+var_type+'.jpg', dpi=1200, bbox_inches='tight')
    plt.show()
