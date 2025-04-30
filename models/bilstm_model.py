from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

def build_bilstm_model(input_shape, num_classes):
    model = Sequential()
    
    # Define the input layer explicitly
    model.add(Input(shape=input_shape))
    
    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(64, kernel_regularizer=l2(0.01))))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model