import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def train_test_split_sfa(npy_file_path,coordinate_x_y,degree):
    polyRegressor = PolynomialFeatures(degree=degree)
    polyFeatureTrainingSet = polyRegressor.fit_transform(npy_file_path)
    X_train,X_test,y_train,y_test = train_test_split(polyFeatureTrainingSet,coordinate_x_y,test_size=0.33)

    df_1 = pd.DataFrame(npy_file_path)
    df_2 = pd.read_csv('/hri/localdisk/ThesisProject/Kaushik/Simulator_data/Coordinates/coordinates.txt', sep=" ",header=None)

    print(y_test)