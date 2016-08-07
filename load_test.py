
#import pandas
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
import numpy
import pandas
from keras.utils.np_utils import to_categorical

def baseline_model():
	model = Sequential()
	model.add(Dense(1400, input_dim=88, init='glorot_normal', activation='tanh'))
	model.add(Dropout(0.3))
	model.add(Dense(1400, init='uniform', activation='tanh'))
	model.add(Dropout(0.3))
	model.add(Dense(1400, init='uniform', activation='tanh'))
	model.add(Dropout(0.3))
#	model.add(Dense(1300, init='uniform', activation='tanh'))
#	model.add(Dropout(0.2))
#	model.add(Dense(1000, init='uniform', activation='tanh'))
#	model.add(Dropout(0.3))
	model.add(Dense(3, init='uniform', activation='sigmoid'))

	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=sgd , metrics=['accuracy'])
	return model
model = baseline_model()
model.load_weights('model2.h5')

dataframe = pandas.read_csv("feature.txt", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[1:,0:88].astype(float)
Y = dataset[1:,88]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

score = model.evaluate(X, dummy_y)
print score


