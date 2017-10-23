from __future__ import division
import math
import os
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

def save_images(images, size, image_path):
    
    if isinstance(images, (list, tuple)):
        for i, imgs in enumerate(images):
            images[i] = np.expand_dims(imgs, axis = 1)
        images = np.concatenate(images, axis = 1)
        _, _, h, w, c = images.shape
        images = np.reshape(images, (-1, h, w, c))
    num_im = size[0] * size[1]
    return imsave(images[:num_im], size, image_path)

def inverse_transform(images):
    return (images+1.)/2.

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img
