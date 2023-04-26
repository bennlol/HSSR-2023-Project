import cv2, os, random
import numpy as np

rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
DIR = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(DIR, 'data')

images = []

for root, dirpath, names in os.walk(data_dir):
    for name in names:
        soft_path = os.path.join(root, name)
        if soft_path[-4:] == '.png':
            print(soft_path)
            
            image = cv2.imread(soft_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (int(image.shape[1]*0.2), int(image.shape[0]*0.2)),interpolation = cv2.INTER_AREA)
            
            if random.randint(0,2)==0:
                changedimage = cv2.rotate(image, random.choice(rotations))
            elif random.randint(0,1) ==0:
                changedimage = cv2.flip(image, random.randint(-1,1))
            else:
                changedimage = cv2.rotate(image, random.choice(rotations))
                changedimage = cv2.flip(image, random.randint(-1,1))
                
            images.append(np.hstack((image,changedimage)))


final_image = images[0]
for i in range(1,len(images)):
    final_image = np.vstack((final_image, images[i]))

print("/nOn the left is the original image and on the right is the image that has been fliped, rotated, or both")

cv2.imshow('images with rotations or translations', final_image)     
cv2.waitKey(0)
cv2.destroyAllWindows()