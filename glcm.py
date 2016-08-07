import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import numpy as np
from scipy import stats
PATCH_SIZE = 21

np.seterr(divide = 'ignore', invalid = 'ignore')
# open the camera image
image = data.camera()

#plt.imshow(I, cmap = 'Greys_r')

# select some patches from grassy areas of the image
grass_locations = [(474, 291), (440, 433), (466, 18), (462, 236)]
grass_patches = []
#patches are rectangular right bottom of images
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])


# select some patches from sky areas of the image
sky_locations = [(54, 48), (21, 233), (90, 380), (195, 330)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
energy = []
contrast = []
homogen = []
asm = []

angles = [0.0, 45.0, 90.0 , 135.0]
angles = [float(x) for x in angles]
angles = [math.radians(x) for x in angles]
for h in (angles):
	for patch in (sky_patches + grass_patches):
    		glcm = greycomatrix(patch, [5], [h], 256, symmetric=True, normed = True)
    		xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    		ys.append(greycoprops(glcm, 'correlation')[0, 0])
		energy.append(greycoprops(glcm, 'energy'))
		contrast.append(greycoprops(glcm, 'contrast'))
		homogen.append(greycoprops(glcm, 'homogeneity'))
		asm.append(greycoprops(glcm, 'ASM'))

#print glcm[3,4]
xsn = np.asarray(xs)
ysn = np.asarray(ys)
energyn =np.asarray(energy)
contrastn = np.asarray(contrast)
homogenn = np.asarray(homogen)
asmn = np.asarray(asm)



print xsn.shape, ysn.shape, energyn.shape, contrastn.shape, homogenn.shape, asmn.shape, "shapes"
np.savetxt("test.txt",glcm)
print glcm.shape
print glcm.nonzero()
total = glcm.sum()
#imp feature!!
glcp = glcm/total
stats.describe(glcp)

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(angles)*len(grass_patches)], ys[:len(angles)*len(grass_patches)], 'go',
        label='Grass')
ax.plot(xs[len(angles)*len(grass_patches):], ys[len(angles)*len(grass_patches):], 'bo',
        label='Sky')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLVM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Grass %d' % (i + 1))

for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Sky %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()
