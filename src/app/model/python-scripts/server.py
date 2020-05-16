import os, sys, json
from advanced_object_detection import TrafficSignDetector, loadImage
import zerorpc
import numpy as np
from pandas import datetime

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()

class ObjectDetector(object):
    def __init__(self):
        modelPath = 'src/assets/models'
        self.detector = TrafficSignDetector(modelPath)

    # def detect(self, imagePath):
    #     #imagePath = 'src/assets/testimage.png'
    #     img = loadImage(imagePath)
    #     bboxes, labels, certainties = fullPrediction(img)
    #     result = { 'bounding_boxes': bboxes, 'classifications': labels, 'certainties': certainties }
    #     return json.dumps(result, default=myconverter)

    def detect(self, imagePath):
        img = loadImage(imagePath)
        objects, labels = self.detector.detect(img)
        result = { 'objects': objects, 'classifications': labels }
        return json.dumps(result, default=myconverter)

class Server:
    def __init__(self):
        s = zerorpc.Server(ObjectDetector())
        s.bind("tcp://0.0.0.0:4242")
        print('Server started on port 4242')
        s.run()

server = Server()