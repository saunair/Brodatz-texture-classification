import xgboost as xgb
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from keras.models import Sequential
import numpy
import pandas

import sys
sys.setrecursionlimit(5000)
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataframe = pandas.read_csv("feature.txt", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
data_in = xgb.DMatrix('feature.txt')

