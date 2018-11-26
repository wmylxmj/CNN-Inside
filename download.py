# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 02:39:07 2018

@author: wmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import scipy
import math
import os
import time
import random
from keras import models
from keras import layers
from keras.applications.inception_v3 import InceptionV3

conv_base = InceptionV3(include_top=False, 
                        weights='imagenet',   
                        input_shape=(256, 256, 3))

conv_base.summary()