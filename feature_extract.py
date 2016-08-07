import matplotlib.pyplot as plt
from scipy import stats
import math
import matplotlib.image as mpimg
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from random import randint

import numpy as np
from scipy import stats
PATCH_SIZE = 20
training_length = 50000
#angles = [0.0, 45.0, 90.0 , 135.0]
#angles = [float(x) for x in angles]
#angles = [math.radians(x) for x in angles]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/2]
np.set_printoptions(precision=10)

def reshape_list(arr):
	return np.reshape(arr, arr.size)

def feature_glcp(x,y,I, names):
	temp = np.array([])
	for h in angles:
		square = I[vertil:vertil + PATCH_SIZE,
			horiz:horiz + PATCH_SIZE]
		glcm = greycomatrix(square, [1 ,2, 3], [h], 256, symmetric=True, normed = True)
		dissim = (greycoprops(glcm, 'dissimilarity'))
		correl = (greycoprops(glcm, 'correlation'))
		energy = (greycoprops(glcm, 'energy'))
		contrast= (greycoprops(glcm, 'contrast'))
		homogen= (greycoprops(glcm, 'homogeneity'))
		asm =(greycoprops(glcm, 'ASM'))
		glcm = glcm.flatten()
		statistics = stats.describe(glcm)
		temp1 = [statistics.mean ,statistics.variance ,statistics.skewness ,statistics.kurtosis]
		check = np.append(dissim, correl)
		check = np.append(check, energy)
		check = np.append(check,contrast) 
		check = np.append(check, homogen)
		check = np.append(check, asm)
		check = np.append(check,temp1)
		temp = np.append(temp,check)
	temp = np.append(temp,names)	
	return temp

A = np.zeros((89,))	
for names in range(1,4):
	I = mpimg.imread("textures/" +  "1." + str(names) + ".0" + str(names)+ ".tiff")
	x_len, y_len = I.shape
	x_axes = []
	y_axes= []
	feature = []
	temp = []
	for data in range(0, training_length):
		x_axes.append(randint(0,x_len-1));
		y_axes.append(randint(0, y_len-1));

	for pixels in range(0,training_length):
		vertil = y_axes[pixels]
		horiz  = x_axes[pixels]
		temp = feature_glcp(horiz,vertil,I,names)
		A = np.row_stack((A, temp))

np.savetxt("feature.txt",A,delimiter=',')
