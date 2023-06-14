import cv2, os, random, sys
import numpy as np
from keras.models import model_from_json
from image_manipulation import randomRotation

def load_pair(img1, img2, dir):
    return [cv2.imread(dir+img1, cv2.IMREAD_GRAYSCALE), cv2.imread(dir+img2, cv2.IMREAD_GRAYSCALE)]

DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
cluster_dirs = [os.path.join(DIR, "cluster/"+dir) for dir in range(0,6)] #use like cluster_dirs[0] for the 0 cluster

with open('model.json', 'r') as json_file:
    siamese_json = json_file.read()

siamese = model_from_json(siamese_json)
siamese.load_weights("model.h5")

results = siamese.predict(load_pair("V4-T8-31step10s.png","V4-T8-31step10s.png",cluster_dirs[1]))
print(f"(single void defect) : (same image), prediction: ${results}")

results = siamese.predict(cv2.imread(dir+"V4-T8-31step10s.png", cv2.IMREAD_GRAYSCALE), randomRotation(cv2.imread(dir+"V4-T8-31step10s.png", cv2.IMREAD_GRAYSCALE)), cluster_dirs[1])
print(f"(single void defect) : (same image but rotated), prediction: ${results}")

results = siamese.predict(load_pair("V4-T8-31step10s.png","V4-T16-86step10s.png",cluster_dirs[1]))
print(f"(single void defect) : (double void defect), prediction: ${results}")

results = siamese.predict(load_pair("V4-T8-31step10s.png","V4-T7-79step10s.png",cluster_dirs[1]))
print(f"(single void defect) : (grain defect), prediction: ${results}")
