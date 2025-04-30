from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

def build_shallow_network(input_shape, num_classes):
    model = Sequential()
    
    # Explicitly define the input layer with the given input shape
    model.add(Input(shape=input_shape))
    
    # First layer (Shallow Network with regularization)
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.05)))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model