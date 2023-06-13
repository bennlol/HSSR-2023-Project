# take pairs of images from a cluseter true image predicted value and decision
import cv2, os, random, sys
import numpy as np
from keras.models import model_from_json

CLUSTER = 1

with open('model.json', 'r') as json_file:
    siamese_json = json_file.read()
    
siamese = model_from_json(siamese_json)
siamese.load_weights("model.h5")


def load_data(cluster):
    DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
    data_dir = os.path.join(DIR, "cluster/"+dir)
    filelist = []
    classlist = []
    for root, dirpath, names in os.walk(data_dir):
        for name in names:
            if root[-1] != "-1" and name[-3:]=='png':
                soft_path = os.path.join(root, name)
                filelist.append(soft_path)
                classlist.append(root[-1])
    # filelist = np.random.choice(filelist, size=3, replace=False)
    return filelist

paths = load_data(CLUSTER)

for i in range(len(paths)):
    path1 = paths[i]
    path2 = random.choice(paths)
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2= cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    results = siamese.predict([img1,img2])
    print(path1,path2,results)