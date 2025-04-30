from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

def build_rnn_model(input_shape, num_classes):
    model = Sequential()
    
    # Define the input layer explicitly
    model.add(Input(shape=input_shape))
    
    # First RNN layer
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(Dropout(0.5))  # Add dropout for regularization
    
    # Second RNN layer
    model.add(SimpleRNN(64))
    model.add(Dropout(0.5))  # Additional dropout layer
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model