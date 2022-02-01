import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from CPMLAgent import CPMLGameEnv
from importlib_metadata import version

modelFileName = "./cpmlModel"

print(keras.__version__)

model = keras.models.load_model(modelFileName)


