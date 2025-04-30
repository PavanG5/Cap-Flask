import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_transformer_model(num_features, num_classes):
    # Updated to use num_features instead of input_shape
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(num_features,)))  # Use num_features directly
    model.add(Dense(num_classes, activation='softmax'))
    return model

