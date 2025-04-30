from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_basic_dense_network(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),  # Ensure input_dim is wrapped in a tuple
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
