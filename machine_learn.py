import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn import datasets
digits = datasets.load_digits()
print digits.data
print digits.target


#test(predictor) of test_dataset
X, y = digits.data[:-10], digits.target[:-10]
print y

model = svm.SVC(kernel = 'rbf' , C= 100, gamma= 0.001)
model.fit(X,y)
print model.score(X,y)

plt.imshow(digits.images[-5], cmap= plt.cm.gray_r, interpolation = 'nearest')

predicted = model.predict(digits.data[-5])
print predicted
