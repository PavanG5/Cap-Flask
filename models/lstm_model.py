import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    
    # Define the input layer explicitly
    model.add(Input(shape=input_shape))
    
    # LSTM layer
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model