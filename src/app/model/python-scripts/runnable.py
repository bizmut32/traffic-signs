import os, sys, json
from full_object_detection import *

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()

imagePath = sys.argv[1]
#imagePath = 'src/assets/testimage.png'
img = loadImage(imagePath)
bboxes, labels = fullPrediction(img)
result = { 'bounding_boxes': bboxes, 'classifications': labels }
print(json.dumps(result, default=myconverter))
sys.stdout.flush()