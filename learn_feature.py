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
from keras.models import model_from_json

# fix random seed for reproducibility
import sys
sys.setrecursionlimit(6000)
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataframe = pandas.read_csv("feature.txt", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[1:,0:88].astype(float)
Y = dataset[1:,88]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
#encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

#encode class values as integers
#create model
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
# Fit the model

model = baseline_model()
model.fit(X_train, dummy_y,nb_epoch=150, batch_size=8000, verbose = 1, validation_split=0.20)

#evaluate the model
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=20, batch_size=2000, verbose=1)
#kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=seed)
#encoder = LabelEncoder()
#encoder.fit(Y_test)
encoded_Y_test = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)

#results = cross_val_score(estimator, X, dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

score = model.evaluate(X_test, dummy_y_test)
print score

model_json = model.to_json()
with open("model.json", "w") as json_file:
    		json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("model2.h5")
		print("Saved model to disk")
joblib.dump(model, 'neural_model2.pkl')
 
