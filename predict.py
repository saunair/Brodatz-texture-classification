from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

clf = joblib.load('filename.pkl')
