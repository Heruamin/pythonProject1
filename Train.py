from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
import time

from tensorflow.keras.layers import Activation, Dense, LSTM, Attention, Flatten
from tensorflow.keras.layers import Conv2D, Embedding, Input, Dropout, TimeDistributed
from tensorflow.keras import activations
from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import skimage
import skimage.transform as skimage_t

from glob import glob        # tool to match file patterns
import numpy as np

# Caricamento dei dati, sto seguendo quello che ci ha mostrato vincenzo in :
# - https://colab.research.google.com/drive/1d7rSfAKhDoYBqX_2Rzj9LuznYEM9kiET?usp=sharing#scrollTo=Itd-h9N_ndB-

n_classes = 2

path_drive = '/content/drive/MyDrive/Colab Notebooks/ColabProgetto/Progetto/Split/HMDB51/train'

NoViol_images = glob(path_drive + '/NoViolence/*.jpg')
Viol_images = glob(path_drive + '/Violence/*.jpg')

# build labels array, starting as a list
# you could probably do this more efficiently with list comprehensions
# but I find this more readable
labels = []

for i in NoViol_images:
    labels.append(0)           # D=0 for NoViol, undefaced

for i in Viol_images:
    labels.append(1)           # D=1 for Viol

labels = np.uint8(labels)
image_filenames = NoViol_images + Viol_images

n_images = len(labels)
n_channels = 3               # colour channels

print('Created list with labels. There are', n_images, 'of them.')

image_data = np.zeros((n_images, 150, 150, n_channels), dtype=np.float32)

for i, img_filename in enumerate(image_filenames):
    img = plt.imread(img_filename)
    print(img.shape, end = ' - ')
    img = skimage.img_as_float(img)
    img = skimage_t.resize(img, output_shape = (150,150,3))
    print(img.shape)
    img = img / np.max(img)

    image_data[i, :, :, :] = img

# print('First image (NoViolence):')
# plt.imshow(image_data[0, :, :, :])
# plt.axis('off')
# plt.show()
#
# print('Last image (Violence):')
# plt.imshow(image_data[-1, :, :, :])
# plt.axis('off')
# plt.show()