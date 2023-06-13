# take pairs of images from a cluseter true image predicted value and decision
import cv2, os, random, sys
import numpy as np

CLUSTER = 1

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

for i in 