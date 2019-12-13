import sys, h5py
import cv2, glob
from multiprocessing import Pool
import numpy as np

hf = h5py.File('data.h5', "w")
train_paths = glob.glob('dataset/*/*.png')

def process_image(impath):
    im = cv2.imread(impath)
    im = cv2.resize(im, (55,55))
    im = im.transpose()
    return im

# Class dictionary
label_dict = {'fire': 0, 'nofire': 1}
def get_labels(impath):
    label = impath.split('\\')[1]
    return label_dict[label]

data = []
labels = []
#p = Pool(1) # set this to number of cores you have
for i in train_paths:
    data.append(process_image(i))
    labels.append(get_labels(i))
    print(len(data))

data = np.array(data)
labels = np.array(labels)
print(data)

hf.create_dataset('data', data=data)
hf.create_dataset('labels', data=labels)
hf.close()