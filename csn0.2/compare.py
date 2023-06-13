import cv2, os, random, sys
import numpy as np
from keras.models import model_from_json
from image_manipulation import randomRotation

def load_pair(img1, img2, dir):
    return [cv2.imread(img1, cv2.IMREAD_GRAYSCALE), cv2.imread(img2, cv2.IMREAD_GRAYSCALE)]

DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
cluster_dirs = [os.path.join(DIR, "cluster/"+dir) for dir in range(0,6)] #use like cluster_dirs[0] for the 0 cluster

with open('model.json', 'r') as json_file:
    siamese_json = json_file.read()

siamese = model_from_json(siamese_json)
siamese.load_weights("model.h5")

results = siamese.predict(load_pair("V4-T8-31step10s.img","V4-T8-31step10s.img",cluster_dirs[1]))
print(f"same image (single void defect), prediction: ${results}")

results = siamese.predict(load_pair("V4-T8-31step10s.img",randomRotation("V4-T8-31step10s.img"),cluster_dirs[1]))
print(f"image with rotated version of the same image, prediction: ${results}")

