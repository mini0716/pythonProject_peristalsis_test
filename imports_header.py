import os
import cv2
import h5py
import glob
import librosa
import numpy as np
import tensorflow as tf
import scipy.signal as sps
import matplotlib.pyplot as plt

import librosa
import librosa.display

from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, BatchNormalization, ReLU, Add, Layer, Conv2D, GlobalAveragePooling3D, Conv1D, \
    GlobalAveragePooling2D, Dense, MaxPooling2D, Conv3D, MaxPooling3D, Flatten, LSTM, concatenate, GlobalAveragePooling1D, MaxPooling1D, add, Dropout, AveragePooling1D, AveragePooling2D, AveragePooling3D
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, GlobalAveragePooling3D, Dense, Flatten, Multiply, Lambda, Concatenate, Conv2DTranspose, Conv3DTranspose, UpSampling3D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import pandas as pd
import csv







