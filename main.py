import torch
import numpy as np

from image_loader import image_loader
from vgg import model, feature_extracter
from preprocess import preprocess_image


sketch_model = model( feature_extracter, 50 )

path_sketch = '/home/iacv/project/sketch/dataset/sketches/'
path_images = '/home/iacv/project/sketch/dataset/images/'
path_list_classname = '/home/iacv/project/sketch/train_class_list.txt'

list_classname = np.loadtxt(path_list_classname,dtype= 'str')

def model_train(model):

    loader = image_loader(path_sketch, list_classname)
    


