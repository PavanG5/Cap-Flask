from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

def build_gru_model(input_shape, num_classes):
    model = Sequential()
    
    # Define the input layer explicitly
    model.add(Input(shape=input_shape))
    
    # First GRU layer
    model.add(GRU(256, recurrent_dropout=0.3, kernel_regularizer=l2(0.01), return_sequences=True))
    
    # Second GRU layer
    model.add(GRU(128, recurrent_dropout=0.3, kernel_regularizer=l2(0.01)))
    
    # Dropout and Dense layers
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model