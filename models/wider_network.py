from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential
def build_wider_network(input_shape, num_classes):
    model = Sequential()
    
    # Explicitly define the input layer with the given input shape
    model.add(Input(shape=input_shape))

    # First layer (Wider Network with more neurons and regularization)
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))

    # Additional dense layer
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    return model