import numpy as np
from skimage import exposure
from skimage.transform import resize
from skimage.feature import hog

def histogramStretching(img):
    p2 = np.percentile(img, 0)
    p98 = np.percentile(img, 85)
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

def histogramEqualization(img):
    imgEq = exposure.equalize_hist(img)
    return imgEq

def adaptiveHistogramEqualization(imgs):
    imgEq = exposure.equalize_adapthist(imgs, clip_limit=0.1)
    return imgEq

def histogramOrientedGradients(img):
    img = resize(img, (300, 300))
    img = histogramEqualization(img)
    data = hog(img, orientations=8, pixels_per_cell=(12, 12), cells_per_block=(1, 1), multichannel=True)
    return data

def applyHistogramNormalization(imgs, methods=[1]):
    normMethods = (histogramStretching, histogramEqualization, adaptiveHistogramEqualization)
    for method in methods:
        imgs = [ normMethods[method](img) for img in imgs ]
    return np.array(imgs)

def applyHOG(x):
    x = [histogramOrientedGradients(img) for img in x]
    return x

def meanNorm(imgs):
    means = np.mean(imgs, axis=(1, 2, 3))
    normalized = np.array([img for img in imgs])
    for i in range(0, len(means)):
        normalized[i] -= means[i]
    return normalized

def stdNorm(imgs):
    stds = np.std(imgs, axis=(1, 2, 3))
    normalized = np.array([img for img in imgs])
    for i in range(0, len(stds)):
        normalized[i] /= stds[i]
    return normalized

def minMaxNorm(imgs):
    low = np.min(imgs, axis=(1, 2, 3))
    high = np.max(imgs, axis=(1, 2, 3))
    normalized = np.array([img for img in imgs])
    for i in range(0, len(imgs)):
        normalized[i] = (normalized[i] - low[i]) / (high[i] - low[i])
    return normalized * 2 - 1

def applyNormalization(imgs):
    imgs = imgs.astype(np.float32)
    imgs = meanNorm(imgs)
    imgs = stdNorm(imgs)
    imgs = minMaxNorm(imgs)
    return imgs

def resizeMany(x, shape=(32, 32)):
    return np.array([ resize(img, shape) for img in x ])

def shuffle(x, y):
    permutation = np.random.permutation(x.shape[0])
    x = np.take(x, permutation, axis=0)
    y = np.take(y, permutation, axis=0)
    return x, y

def preprocessMany(x):
    x = histogramEqualization(x)
    x = applyNormalization(x)
    return np.array(x)