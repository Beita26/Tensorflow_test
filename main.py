import tensorflow as tf
import numpy as np
from scipy import misc
from PIL import Image
import glob
import matplotlib.pyplot as plt
import scipy.misc
from matplotlib.pyplot import imshow
from IPython.display import SVG
import cv2
import seaborn as sn
import pandas as pd
import pickle
from keras import layers
from keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import layer_utils, np_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras import losses
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

#Importamos la bbdd de ejemplo
from keras.datasets import cifar100

# x son las imagenes e y etiquetas
#label_mode es el tipo de etiqueta asociada a la imagen
(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data(label_mode='fine')
#Convertimos los arrays de etiquetas en su versión one-hot-encoding
y_train = np_utils.to_categorical(y_train_original,100)
y_test = np_utils.to_categorical(y_test_original,100)

#Representamos la imagen que esconden estas matrices, el 3 es numero de imagen de la matriz
#imgplot = plt.imshow(x_train_original[3])
#plt.show()

#Obtenemos que el array comprenderá valores de entre 0 y 1
x_train = x_train_original/255
x_test = x_test_original/255

#Definimos los canales
K.set_image_data_format('channels_last')
#Definimos la fase del experimento: Fase 1 = Entrenamiento
K.set_learning_phase(1)

#Entrenemos una red neuronal sencilla
def create_simple_nn():
    model = Sequential() #Modelo usado para redes neuronales sencillas
    # Flatten convierte los elementos de la matriz de imagenes de entrada en un array plano
    model.add(Flatten(input_shape=(32,32,3),name="Input_layer"))  #Tamaño de la imagen 32x32 3 canales(RGB)
    # Dense, añadimos una capa oculta (hidden layer) de la red neuronal
    model.add(Dense(1000, activation='relu',name="Hidden_layer_1")) #Nº de nodos 1000, función de activación ReLu
    model.add(Dense(500, activation='relu', name="Hidden_layer_2"))
    model.add(Dense(100, activation='sofmax', name="Output_layer"))

    return model



