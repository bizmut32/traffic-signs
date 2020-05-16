import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import math
import time
from keras.utils import Sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import callbacks
import keras.backend as K
import tensorflow as tf
from os import listdir
from skimage import io, color

from utils import ModelSaver
from preprocessing import histogramEqualization, histogramStretching, applyNormalization, adaptiveHistogramEqualization, resize, resizeMany, applyHOG
from cnn import CommitteeOfCNNs

def getImages():
    path = 'data/signs'
    return ModelSaver(path).load()

def getParisDatasetImages():
    path = 'data/val/'
    def openImage(path):
        img = io.imread(path)
        return img.astype(np.uint8)

    def loadImagesFromFolder(path):
        folder = listdir(path)
        images = []
        imageNames = []
        for imgPath in folder:
            img = openImage(path + imgPath)
            images.append(img)
            imageNames.append(imgPath)
        return np.asarray(images), np.array(imageNames)
    return loadImagesFromFolder(path)

def getLabels():
    return ModelSaver('data/labels').load()

def getRandomImageOfSign(image, sign, maxWidthScale):
    prevx, prevy, w, h = int(sign['x']), int(sign['y']), int(sign['w']), int(sign['h'])
    maxW = int(min(w, h) * maxWidthScale)
    newW = random.randint(max(w, h), maxW)
    minX, maxX, minY, maxY = prevx - (newW - w) / 2, prevx + (newW - w) / 2, prevy - (newW - h) / 2, prevy + (newW - h) / 2
    x, y = random.randint(int(minX), int(maxX)), random.randint(int(minY), int(maxY))
    x1, y1, x2, y2 = x - int(newW / 2), y - int(newW / 2), x + int(newW / 2), y + int(newW / 2)
    x1, y1 = max(x1, 0), max(y1, 0)
    return image[y1:y2, x1:x2], w / newW, h / newW, (prevx - x1) / newW, (prevy - y1) / newW

def getRandomImageOfBlank(image, signs, boundaries, maxWidthScale):
    imgW, imgH = image.shape[1], image.shape[0]
    minW, maxW, = signs[0]['w'], min(signs[0]['w'] * maxWidthScale * 1.5, imgW)
    while True:
        w = random.randint(int(minW), int(maxW))
        x, y = random.randint(0, imgW - w), random.randint(0, imgH - w)
        x2, y2 = x + w, y + w
        if np.max(np.array([ iou(sign, x, x2, y, y2) for sign in signs ])) < 0.3:
            break
    return image[y:y2, x:x2]

def getRandomImageOfParisDataset(image):
    imgW, imgH = int(image.shape[1] / 2), image.shape[0]
    minW, maxW = int(imgW / 5), int(imgW / 2)
    w = random.randint(minW, maxW)
    x, y = random.randint(0, imgW - w), random.randint(0, imgH - w)
    x2, y2 = x + w, y + w
    return image[y:y2, x:x2]

def iou(sign, bx1, bx2, by1, by2):
    ax, ay, w, h = sign['x'], sign['y'], sign['w'], sign['h']
    ax1, ax2, ay1, ay2 = ax - w / 2, ax + w / 2, ay - h / 2, ay + h / 2
    x1, x2 = max(ax1, bx1), min(ax2, bx2)
    y1, y2 = max(ay1, by1), min(ay2, by2)
    intersection = (x2 - x1) * (y2 - y1)
    areaA, areaB = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
    union = areaA + areaB - intersection
    iou = intersection / union
    return max(intersection/ areaA, intersection / areaB)

def generateTrainingDataset(images, signs, boundaries, multplier=100, maxWidthScale = 3):
    x = []
    y = []
    for i in range(0, len(images)):
        for sign in signs[i]:
            for _ in range(0, multplier):
                imageOfSign = getRandomImageOfSign(images[i], sign, maxWidthScale)
                x.append(resize(imageOfSign[0], (48, 48)))
                y.append([1, imageOfSign[1], imageOfSign[2], imageOfSign[3], imageOfSign[4]])
                
                x.append(resize(getRandomImageOfBlank(images[i], signs[i], boundaries[i], maxWidthScale), (48, 48)))
                y.append([0, 0, 0, 0, 0])
    
    return np.array(x, dtype='float32'), np.array(y, dtype='float32')

def appendFurtherImages(images, px, py, multiplier = 100):
    size = min(len(images), int(len(py) / 7 / multiplier))
    x, y = [], []
    for i in range(0, size):
        for _ in range(0, multiplier):
            x.append(resize(getRandomImageOfParisDataset(images[i]), (48, 48)))
            y.append([0, 0, 0, 0, 0])
    x, y = np.array(x), np.array(y)
    return np.concatenate((px, x), axis=0), np.concatenate((py, y), axis=0)

class Cnn:
    def __init__(self, preprocessor):
        model = Sequential()
        model.add(Conv2D(48, kernel_size=4, padding='same', activation='relu', input_shape=(48, 48, 3)))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(24, kernel_size=4, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(5, activation='sigmoid'))

        self.model = model
        self.preprocessor = preprocessor

        self.model.compile(optimizer='adam', loss=self.loss())
    
    def loss(self):
        def loss_fn(true, pred): 
            mask = K.cast(K.less(true[..., 0], 0.5), 'float32')
            K.expand_dims(mask, -1)
            return K.sum(K.square(true[..., 0] - pred[..., 0])) * mask + K.sum(K.square(true - pred)) * (1 - mask)
        return loss_fn

    
    def clone(self):
        cnn = Cnn(self.preprocessor)
        cnn.model = self.model
        return cnn
    
    def save(self, path):
        path = 'data/advanced_%s.h5'%path
        self.model.save_weights(path)
        
    def load(self, path):
        path = 'data/advanced_%s.h5'%path
        self.model.load_weights(path)
        return self

    def train(self, x, y, xValid, yValid, epochs):
        start = time.time()
        x = [self.preprocessor(x)]
        xValid = self.preprocessor(xValid)
        end = time.time()
        print('Preprocessing took', int(end - start), 's')

        start = time.time()
        self.history = self.model.fit(
            x, y, epochs = epochs, batch_size = 12, 
            validation_data = (xValid, yValid), verbose=2)
        end = time.time()
        print('Training model took', int(end - start), 's')

class ObjectDetector: 
    def __init__(self, cnns=[]):
        self.names = ['eq', 'stretch', 'adeq']
        self.cnns = cnns
        
    def load(self):
        for name in self.names:
            model = self.model()
            model.load_weights('models/advanced2_%s.h5'%name)
            self.cnns.append(model)
    
    def save(self):
        for i in range(0, len(self.names)):
            name = self.names[i]
            cnn = self.cnns[i]
            cnn.save_weights('models/advanced2_%s.h5'%name)

    def model(self):
        model = Sequential()
        model.add(Conv2D(48, kernel_size=4, padding='same', activation='relu', input_shape=(48, 48, 3)))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(24, kernel_size=4, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(5, activation='sigmoid'))
        return model
        
    def predict(self, image_eq, image_stretch, image_adeq):
        boxes = np.empty((0, 4), dtype='int')
        images = [image_eq, image_stretch, image_adeq]
        i = 1
        while i <= 8:
            boxes = np.concatenate((boxes, self.getBoxes(images[0], i)), axis=0)
            i *= 2
        predictions = self.scanBoxes(images, boxes)
        result = self.nonMaxSupression(predictions)
        
        return np.array(result)
            
    def getBoxes(self, image, i):
        boxes = []
        boxSize = int(image.shape[0] / i)
        numberOfBoxes = 2 * i - 1
        x, y = 0, 0
        for i in range(0, numberOfBoxes):
            for j in range(0, numberOfBoxes):
                boxes.append([x, x + boxSize, y, y + boxSize])
                x += boxSize / 2
            y += boxSize / 2
            x = 0
        return np.array(boxes, dtype='int')
    
    def showPrediction(self, eq, stretch, adeq):
        start = time.time()
        predictions = self.predict(eq, stretch, adeq)
        end = time.time()
        print('Detection took', int((end - start) * 100) / 100.0, 's')
        print(predictions)
        fig, ax = plt.subplots(1)
        img = eq
        low, high = np.min(img), np.max(img)
        img = (img - low) / (high - low)
        ax.imshow(img.astype('float32'), cmap=plt.get_cmap('gray_r'))
        for pred in predictions:
            x1, x2, y1, y2 = pred[1:]
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x, y),w, h,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        plt.show()
            
    def scanBoxes(self, images, boxes):
        imgs = []
        start = time.time()
        for box in boxes:
            x1, x2, y1, y2 = box
            cropped = [image[y1:y2, x1:x2] for image in images]
            imgs.append(resizeMany(cropped, (48, 48)))
        imgs = np.transpose(np.array(imgs), axes=(1, 0, 2, 3, 4))
        end = time.time()
        print('Image resizing took', int((end - start) * 100) / 100, 's')
        
        start = time.time()
        predictions = self.getPredictions(imgs)
        end = time.time()
        print('Prediction took', int((end - start) * 100) / 100, 's')

        for i in range(0, len(boxes)):
            x1, x2, y1, y2 = boxes[i]
            boxSize = x2 - x1
            pred = predictions[i]
            x = boxSize * pred[3] + x1
            y = boxSize * pred[4] + y1
            w = boxSize * pred[1]
            h = boxSize * pred[2]
            pred[1] = x - w / 2
            pred[2] = x + w / 2
            pred[3] = y - h / 2
            pred[4] = y + h / 2
        return predictions

    def getPredictions(self, images):
        predictions = np.array([ self.cnns[i].predict(images[i]) for i in range(0, 3) ])
        predictions = np.transpose(predictions, axes=(1, 0, 2))
        finalPredictions = []
        for preds in predictions:
            if np.sum(preds[..., 0]) < 0.7:
                finalPredictions.append([0, 0, 0, 0, 0])
            else:
                truePreds = preds[preds[..., 0] > 0.5]
                pred = [ np.sum(preds[..., 0]) / 3, *(np.sum(truePreds[..., 1:], axis=0) / len(truePreds)) ]
                finalPredictions.append(pred)
        return np.array(finalPredictions)

    def nonMaxSupression(self, predictions):
        predictions = predictions[predictions[..., 0] > 0.8]
        results = []
        while len(predictions) > 0:
            bestMatch = predictions[np.argmax(predictions[..., 0])]
            mask = self.findHighIOUs(predictions, bestMatch)
            # matches = predictions[mask]
            predictions = predictions[~mask]
            results.append(bestMatch)
        return np.array(results)
            
    def findHighIOUs(self, predictions, box):           
        matches = []
        ax1, ax2, ay1, ay2 = box[1:]
        for pred in predictions:
            bx1, bx2, by1, by2 = pred[1:]
            if self.iou(ax1, ax2, ay1, ay2, bx1, bx2, by1, by2) > 0.33:
                matches.append(True)
            else:
                matches.append(False)
        return np.array(matches)
    
    def mergeBoundingBoxes(self, boxes):
        x1, x2, y1, y2 = np.average(boxes[..., 1:5], axis=0, weights = boxes[..., 0])
        return np.array([x1, x2, y1, y2])
    
    def clone(self):
        this = ObjectDetector(self.cnn)
        return this
    
    def iou(self, ax1, ax2, ay1, ay2, bx1, bx2, by1, by2):
        x1, x2 = max(ax1, bx1), min(ax2, bx2)
        y1, y2 = max(ay1, by1), min(ay2, by2)
        intersection = (x2 - x1) * (y2 - y1)
        areaA, areaB = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
        union = areaA + areaB - intersection
        # iou = intersection / union
        return max([intersection / areaA, intersection / areaB ])

class TrafficSignDetector():
    def __init__(self, load = True):
        if (load):
            self.classifier = CommitteeOfCNNs()
            self.classifier.load()

            self.detector = ObjectDetector()
            self.detector.load()

    def detect(self, image):
        start = time.time()
        imgEq, imgStretch, imgAdEq = self.getPreprocessed(image)
        end = time.time()
        print('Image preprocessing took', int((end - start) * 100) / 100, 's')
        
        start = time.time()
        objects = self.detector.predict(imgEq, imgStretch, imgAdEq)
        end = time.time()
        print('Detection took', int((end - start) * 100) / 100, 's')
        self.extendBoundingBoxes(objects, 0.15)
        images = self.prepareImages(objects, image, imgEq, imgStretch, imgAdEq)

        start = time.time()
        labels = [self.classifier.predict(imgs) for imgs in images]
        end = time.time()
        print('Classifications took', int((end - start) * 100) / 100, 's')
        return objects, labels

    def showDetection(self, image, labels):
        start = time.time()
        objects, classes = self.detect(image)
        end = time.time()
        print('Predition took', int((end - start) * 100) / 100, 's')

        imgEq = self.getPreprocessed(image)[0]
        fig, ax = plt.subplots(1)
        img = imgEq
        low, high = np.min(img), np.max(img)
        img = (img - low) / (high - low)
        ax.imshow(img.astype('float32'), cmap=plt.get_cmap('gray_r'))

        for i in range(0, len(objects)):
            pred = objects[i]
            x1, x2, y1, y2 = pred[1:]
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x, y),w, h,linewidth=1,edgecolor='r',facecolor='none')
            ax.text(x2, y, labels[classes[i]], bbox=dict(facecolor='red'))
            ax.add_patch(rect)
        plt.show()

    def prepareImages(self, objects, image, imgEq, imgStrecth, imgAdeq):
        images = []
        for obj in objects:
            x1, x2, y1, y2 = obj[1:]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            imgSimple = image[y1:y2, x1:x2]
            imgHOG = applyHOG(np.array([imgSimple]))[0]
            resized = resizeMany([imgSimple, imgStrecth[y1:y2, x1:x2], imgEq[y1:y2, x1:x2], imgAdeq[y1:y2, x1:x2]], (40, 40))
            resized[0] = applyNormalization(np.array([resized[0]]))[0]
            img = [*resized, imgHOG]
            images.append(img)
        return images
        
    def extendBoundingBoxes(self, objects, ratio):
        for obj in objects:
            x1, x2, y1, y2 = obj[1:]
            w, h = x2 - x1, y2 - y1
            dx, dy = w * ratio / 2, h * ratio / 2
            x1, x2, y1, y2 = x1 - dy, x2 + dx, y1 - dy, y2 + dy
            obj[1], obj[2], obj[3], obj[4] = x1, x2, y1, y2

    def getPreprocessed(self, image):
        return self.preprocessor(histogramEqualization)([image])[0],\
            self.preprocessor(histogramStretching)([image])[0],\
            self.preprocessor(adaptiveHistogramEqualization)([image])[0]

    def preprocessor(self, fn):
        def preprocess(imgs):
            imgs = np.array([ fn(img) for img in imgs])
            imgs = applyNormalization(imgs)
            return np.array(imgs)
        return preprocess

    def clone(self):
        new = TrafficSignDetector(False)
        new.classifier = self.classifier
        new.detector = self.detector

def preprocessor(fn):
    def preprocess(imgs):
        imgs = np.array([ fn(img) for img in imgs])
        imgs = applyNormalization(imgs)
        return np.array(imgs)
    return preprocess

def showPrediction(cnns, x, y):
    preds = np.array([ cnn.model.predict(np.array([x]))[0] for cnn in cnns ])
    truePreds = preds[preds[..., 0] > 0.5]
    if (np.sum(preds[..., 0]) > 0.4):
        pred = [ np.sum(preds[..., 0]) / len(preds), *np.sum(truePreds[..., 1:], axis=0) / len(truePreds) ]
    else:
        pred = [np.sum(preds[..., 0]) / 3, 0, 0, 0, 0]
    fig, ax = plt.subplots(1)
    ax.imshow(x)
    print('Predicted', int(pred[0] * 10000) / 100, "% chance of object")
    print('Expected:', y[0] * 100, '%')
    print(pred)
    w, h, x, y = pred[1:]
    w, h, x, y = 48 * w, 48 * h, 48 * x, 48 * y
    x1, y1 = x - w / 2, y - h / 2
    rect = patches.Rectangle((x1, y1),w, h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()

def runStatistics(cnns, x, y):
    preds = np.array([ cnn.model.predict(np.array(x)) for cnn in cnns ])
    preds = np.transpose(preds, axes=(1, 0, 2))
    preds = np.sum(preds[..., 0], axis = 1)
    truePreds = preds > 0.7
    yTrue = y[..., 0] > 0.5

    fp = np.sum(truePreds[~yTrue])
    fn = np.sum((1 - truePreds)[yTrue])
    tp = np.sum(truePreds[yTrue])
    tn = np.sum((1 - truePreds)[~yTrue])
    return fp / np.sum(yTrue), fn / np.sum(~yTrue), tp / np.sum(yTrue), tn / np.sum(~yTrue)

def getPreprocessed(image):
    return preprocessor(histogramEqualization)([image])[0],\
           preprocessor(histogramStretching)([image])[0],\
           preprocessor(adaptiveHistogramEqualization)([image])[0]
