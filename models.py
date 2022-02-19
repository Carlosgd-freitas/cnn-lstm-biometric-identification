from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, LSTM
from tensorflow.keras import Input

def scheduler(current_epoch, learning_rate):
    """
    Lowers the learning rate hyperparameter relative to the number of epochs.
    """
    if current_epoch < 2:
        learning_rate = 0.01
    elif current_epoch < 37:
        learning_rate = 0.001
    else:
        learning_rate = 0.0001
    return learning_rate

def create_model_cnn(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the base CNN model.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """
    model = None

    if(remove_last_layer == False):
        model = Sequential(name='Biometric_for_Identification')
    else:
        model = Sequential(name='Biometric_for_Verification')

    model.add(Conv1D(96, (11), input_shape=(window_size, num_channels), activation='relu', name='Conv1'))
    model.add(BatchNormalization(name='Norm1'))
    model.add(MaxPooling1D(strides=4, name='Pool1'))
    
    model.add(Conv1D(128, (9), activation='relu', name='Conv2'))
    model.add(BatchNormalization(name='Norm2'))
    model.add(MaxPooling1D(strides=2, name='Pool2'))

    model.add(Conv1D(256, (9), activation='relu', name='Conv3')) 
    model.add(BatchNormalization(name='Norm3'))
    model.add(MaxPooling1D(strides=2, name='Pool3'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='FC1'))
    model.add(Dense(4096, activation='relu', name='FC2'))
    model.add(Dense(256, name='FC3'))
    model.add(BatchNormalization(name='Norm4'))

    if(remove_last_layer == False):
        model.add(Dropout(0.1, name='Drop'))
        model.add(Dense(num_classes, activation='softmax', name='FC4'))

    return model

def create_model_lstm_cnn(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the LSTM + CNN model used in the article.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """
    model = None

    if(remove_last_layer == False):
        model = Sequential(name='Biometric_for_Identification')
    else:
        model = Sequential(name='Biometric_for_Verification')

    model.add(Input(shape=(window_size, num_channels)))

    model.add(LSTM(128, return_sequences=True, name='LSTM1'))
    model.add(LSTM(128, return_sequences=True, name='LSTM2'))
    model.add(LSTM(128, return_sequences=True, name='LSTM3'))
    model.add(LSTM(128, return_sequences=True, name='LSTM4'))
    model.add(LSTM(128, return_sequences=True, name='LSTM5'))

    model.add(Conv1D(96, (11), input_shape=(window_size, num_channels), activation='relu', name='Conv1'))
    model.add(BatchNormalization(name='Norm1'))
    model.add(MaxPooling1D(strides=4, name='Pool1'))
    
    model.add(Conv1D(128, (9), activation='relu', name='Conv2'))
    model.add(BatchNormalization(name='Norm2'))
    model.add(MaxPooling1D(strides=2, name='Pool2'))
    
    model.add(Conv1D(256, (9), activation='relu', name='Conv3')) 
    model.add(BatchNormalization(name='Norm3'))
    model.add(MaxPooling1D(strides=2, name='Pool3'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='FC1'))
    model.add(Dense(4096, activation='relu', name='FC2'))
    model.add(Dense(256, name='FC3'))
    model.add(BatchNormalization(name='Norm4'))

    if(remove_last_layer == False):
        model.add(Dropout(0.1, name='Drop'))
        model.add(Dense(num_classes, activation='softmax', name='FC4'))

    return model
