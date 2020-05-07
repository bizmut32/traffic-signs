import os, sys, json
from full_object_detection import *
import zerorpc

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
    def detect(self, imagePath):
        #imagePath = 'src/assets/testimage.png'
        img = loadImage(imagePath)
        bboxes, labels, certainties = fullPrediction(img)
        result = { 'bounding_boxes': bboxes, 'classifications': labels, 'certainties': certainties }
        return json.dumps(result, default=myconverter)

class Server:
    def __init__(self):
        s = zerorpc.Server(ObjectDetector())
        s.bind("tcp://0.0.0.0:4242")
        print('Server started on port 4242')
        s.run()

server = Server()