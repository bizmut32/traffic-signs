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

def fullPrediction(image):
    image = image / 255.0
    results = objectDetector.predict(image)
    bboxes = extendBoundingBoxes(image, results)
    images = prepareImages(image, bboxes)
    predictions = []
    for i in range(0, len(results)):
        pred = classifier.predict([images[0][i], images[1][i], images[2][i], images[3][i], images[4][i]])
        predictions.append(pred)
    return bboxes, predictions
    
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
    return np.array(results)

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
    im = cv2.imread(path)
    return im.astype('float32')
