import cv2, os, random, sys
import numpy as np

ROTATIONS = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180 ]
DIR = os.path.dirname(os.path.realpath(__file__))

def randomRotation(img):
    return cv2.rotate(img, random.choice(ROTATIONS))

def randomFlip(img):
    return cv2.flip(img, random.randint(-1,1))

def resize(img, scale=0.5):
    return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation = cv2.INTER_AREA)

def mutateDirectory(fromDir, toDir):
    data_dir = os.path.join(DIR, fromDir)
    data_2_dir = os.path.join(DIR, toDir)
    
    for root, dirpath, names in os.walk(data_dir):
        for name in names:
            soft_path = os.path.join(root, name)
            if soft_path[-4:] == '.png':
                type = random.randint(0,2)
                image = cv2.imread(soft_path)
                image = randomFlip(image) * int(type!=0) + randomRotation(image) * (type+1)%2
                cv2.imwrite(data_2_dir+name, image)

def mutateImage(img):
    type = random.randint(0,2)
    return randomFlip(img) * int(type!=0) + randomRotation(img) * (type+1)%2  

def mutateArray(arr):   
    for i, img in enumerate(arr):
        type = random.randint(0,2)
        arr[i] = randomFlip(img) * int(type!=0) + randomRotation(img) * (type+1)%2
    return arr
        
if __name__ == 'main':
    mutateDirectory('data/test1/0','data/test1/1')
    print('Done')