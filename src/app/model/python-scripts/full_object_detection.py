import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from keras.utils.np_utils import to_categorical
from object_classification import CommitteeOfCNNs
from utils import ModelSaver, showScaledImage
from objectdetection import ObjectDetector
from preprocessing import preprocessMany, applyHistogramNormalization, applyHOG, resizeMany, applyNormalization
import time
import cv2

modelsPath = 'src/assets/models'

classifier = CommitteeOfCNNs()
classifier.load(modelsPath)

objectDetector = ObjectDetector()
objectDetector.load(modelsPath)


def cropImage(img):
    w, h = img.shape[1], img.shape[0]
    wh = min(w, h)
    x1, y1 = (w - wh) / 2, (h - wh) / 2
    x2, y2 = x1 + w, y1 + h
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    return img[y1:y2, x1:x2], x1, y1


def fullPrediction(image):
    image, dx, dy = cropImage(image)
    results = objectDetector.predict(image)
    if (len(results) == 0):
        return [], [], []
    bboxes, certainties = extendBoundingBoxes(image, results)
    images = prepareImages(image, bboxes)
    predictions = []
    for i in range(0, len(results)):
        pred = classifier.predict([images[0][i], images[1][i], images[2][i], images[3][i], images[4][i]])
        predictions.append(pred)

    bboxes[..., 0:2] += dx
    bboxes[..., 2:] += dy
    return bboxes, predictions, certainties
    
def extendBoundingBoxes(image, bboxes, value = .1):
    h, w = image.shape[:2]
    results = []
    for bbox in bboxes:
        x1, x2, y1, y2 = bbox[1:]
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * value, bh * value
        x1 = max(0, x1 - dx)
        x2 = min(w, x2 + dx)
        y1 = max(0, y1 - dy)
        y2 = min(h, y2 + dy)
        results.append([x1, x2, y1, y2])
    return np.array(results), bboxes[..., 0]

def prepareImages(image, bboxes):
    imgs = []
    for bbox in bboxes:
        x1, x2, y1, y2 = bbox.astype('int')
        imgs.append(image[y1:y2, x1:x2])
    imgs = np.array(imgs)
    # Baj van, ha nem 1:1-es az eredeti kép!
    resized = resizeMany(imgs, (40, 40))
    normalized = [ applyHistogramNormalization(resized, [method]) for method in range(0, 3) ]
    preprocessed = [ applyNormalization(imgs) for imgs in np.array(normalized) ]
    preprocessed.append(applyHOG(imgs))
    preprocessed.insert(0, resized)
    return preprocessed

def loadImage(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = im.astype('float32')
    im = im / 255.0
    return im