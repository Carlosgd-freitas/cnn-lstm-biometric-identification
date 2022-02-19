import models
import preprocessing
import utils
import data_manipulation
import loader

import argparse
import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from numpy import savetxt, loadtxt

# Seeds
random.seed(1051)
np.random.seed(1051)
tf.random.set_seed(1051)

# Hyperparameters
batch_size = 100              # Batch Size
training_epochs = 40          # Total number of training epochs
initial_learning_rate = 0.01  # Initial learning rate

# Database folder paths
database_path = ''
processed_data_path = '/media/work/carlosfreitas/IniciacaoCientifica/RedeNeural/'

# Parameters used in pre-processing the data
filter_cuts = [30, 50]        # Low and High cut of the used filter: 30~50Hz
sample_frequency = 160        # Frequency of the sampling
filter_order = 12             # Order of the filter
filter_type = 'filtfilt'      # Type of the used filter: 'sosfilt' or 'filtfilt'
normalize_type = 'sun'        # Type of the normalization that will be applied: 'sun' (The one used in Sun et al.
                              # article), 'each_channel' or 'all_channels'

# Parameters used in data augmentation
window_size = 1920            # Sliding window size, used when composing the dataset
offset = 35                   # Sliding window offset (deslocation), used when composing the dataset
split_ratio = 0.9             # 90% for training | 10% for validation

# Other parameters
num_channels = 64             # Number of channels in an EEG signal
num_classes = 109             # Total number of classes (individuals)

# 9 channels present in Yang et al. article
frontal_lobe_yang = ['Af3.', 'Afz.', 'Af4.']
motor_cortex_yang = ['C1..', 'Cz..', 'C2..']
occipital_lobe_yang = ['O1..', 'Oz..', 'O2..']
all_channels_yang = ['C1..', 'Cz..', 'C2..', 'Af3.', 'Afz.', 'Af4.', 'O1..', 'Oz..', 'O2..']

# Tasks:
# Task 1 - EO
# Task 2 - EC
# Task 3 - T1R1
# Task 4 - T2R1
# Task 5 - T3R1
# Task 6 - T4R1
# Task 7 - T1R2
# Task 8 - T2R2
# Task 9 - T3R2
# Task 10 - T4R2
# Task 11 - T1R3
# Task 12 - T2R3
# Task 13 - T3R3
# Task 14 - T4R3

# Logger
sys.stdout = utils.Logger(os.path.join(processed_data_path, 'results', 'log_script.txt'))
sys.stderr = sys.stdout

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datagen', action='store_true',
                    help='the model will use Data Generators to crop data on the fly')
parser.add_argument('--nofit', action='store_true',
                    help='model.fit will not be executed. The weights will be gathered from the file'+
                    ' \'model_weights.h5\', that is generated if you have ran the model in Identification mode'
                    ' at least once')
parser.add_argument('--noimode', action='store_true',
                    help='the model won\'t run in Identification Mode')
parser.add_argument('--novmode', action='store_true',
                    help='the model won\'t run in Verification Mode')

parser.add_argument('-train', nargs="+", type=int, required=True, 
                    help='list of tasks used for training and validation. All specified tasks need to be higher than\n'+
                    ' 0 and lower than 15. This is a REQUIRED flag')
parser.add_argument('-test', nargs="+", type=int, required=True, 
                    help='list of tasks used for testing. All specified tasks need to be higher than 0 and lower than\n'+
                    ' 15. This is a REQUIRED flag')
args = parser.parse_args()

train_tasks = args.train
test_tasks = args.test

for task in train_tasks:
    if(task <= 0 or task >= 15):
        print('ERROR: All training/validation and testing tasks need to be higher than 0 and lower than 15.\n')
        sys.exit()

for task in test_tasks:
    if(task <= 0 or task >= 15):
        print('ERROR: All training/validation and testing tasks need to be higher than 0 and lower than 15.\n')
        sys.exit()

# Defining the optimizer and the learning rate scheduler
opt = SGD(learning_rate = initial_learning_rate, momentum = 0.9)
lr_scheduler = LearningRateScheduler(models.scheduler, verbose = 0)
model = None

# Not using Data Generators
if(not args.datagen):
    # Loading the raw data
    train_content, test_content = loader.load_data(database_path, train_tasks, test_tasks, 'csv', num_classes, 1)   

    # Filtering the raw data
    train_content = preprocessing.filter_data(train_content, filter_cuts, sample_frequency, filter_order, filter_type, 1)
    test_content = preprocessing.filter_data(test_content, filter_cuts, sample_frequency, filter_order, filter_type, 1)

    # Normalize the filtered data
    train_content = preprocessing.normalize_data(train_content, normalize_type, 1)
    test_content = preprocessing.normalize_data(test_content, normalize_type, 1)

    # Getting the training, validation and testing data
    x_train, y_train, x_val, y_val = data_manipulation.crop_data(train_content, train_tasks, num_classes,
                                                        window_size, offset, split_ratio)
    x_test, y_test = data_manipulation.crop_data(test_content, test_tasks, num_classes, window_size,
                                        window_size)

    # Training the model
    if(not args.nofit):

        # Creating the model
        model = models.create_model_lstm_cnn(window_size, num_channels, num_classes)
        model.summary()

        # Compiling, defining the LearningRateScheduler and training the model
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

        fit_begin = time.time()

        results = model.fit(x_train,
                            y_train,
                            batch_size = batch_size,
                            epochs = training_epochs,
                            callbacks = [lr_scheduler],
                            validation_data = (x_val, y_val)
                            )

        fit_end = time.time()
        print(f'Training time in seconds: {fit_end - fit_begin}')
        print(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
        print(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

        # Summarize history for accuracy
        plt.subplot(211)
        plt.plot(results.history['accuracy'])
        plt.plot(results.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])

        # Summarize history for loss
        plt.subplot(212)
        plt.plot(results.history['loss'])
        plt.plot(results.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        plt.tight_layout()
        plt.savefig(r'accuracy-loss.png', format='png')
        plt.show()

        max_loss = np.max(results.history['loss'])
        min_loss = np.min(results.history['loss'])
        print("Maximum Loss : {:.4f}".format(max_loss))
        print("Minimum Loss : {:.4f}".format(min_loss))
        print("Loss difference : {:.4f}\n".format((max_loss - min_loss)))

        # Saving model weights
        model.save('model_weights.h5')
        print('model was saved to model_weights.h5.\n')

    # Running the model in Identification Mode
    if(not args.noimode):

        # Evaluate the model to see the accuracy
        if(model is None):
            model = models.create_model_lstm_cnn(window_size, num_channels, num_classes)
            model.summary()
            model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.load_weights('model_weights.h5', by_name=True)

        print('\nEvaluating on training set...')
        (loss, accuracy) = model.evaluate(x_train, y_train, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on validation set...')
        (loss, accuracy) = model.evaluate(x_val, y_val, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on testing set...')
        test_begin = time.time()

        (loss, accuracy) = model.evaluate(x_test, y_test, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        test_end = time.time()
        print(f'Evaluating on testing set time in miliseconds: {(test_end - test_begin) * 1000.0}')
        print(f'Evaluating on testing set time in seconds: {test_end - test_begin}')
        print(f'Evaluating on testing set time in minutes: {(test_end - test_begin)/60.0}\n')

    # Running the model in Verification Mode
    if(not args.novmode):

        # Removing the last layers of the model and getting the features array
        model_for_verification = models.create_model_lstm_cnn(window_size, num_channels, num_classes, True)
        model_for_verification.summary()
        model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model_for_verification.load_weights('model_weights.h5', by_name=True)

        x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

        # Calculating EER and Decidability
        y_test_classes = utils.one_hot_encoding_to_classes(y_test)
        d, eer, thresholds = utils.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
        print(f'EER: {eer*100.0} %')
        print(f'Decidability: {d}')

# Using Data Generators
else:

    # Loading the raw data
    train_content, test_content = loader.load_data(database_path, [], test_tasks, 'csv', num_classes)   

    # Filtering the raw data
    test_content = preprocessing.filter_data(test_content, filter_cuts, sample_frequency, filter_order, filter_type)

    # Normalize the filtered data
    test_content = preprocessing.normalize_data(test_content, normalize_type)

    # Getting the testing data
    x_test, y_test = data_manipulation.crop_data(test_content, test_tasks, num_classes, window_size, window_size)

    # Processing train/validation data
    for task in train_tasks:

        if(not os.path.exists(processed_data_path + 'processed_data/task'+str(task))):
            folder = Path(processed_data_path + 'processed_data/task'+str(task))
            folder.mkdir(parents=True)

            # Loading the raw data
            train_content, test_content = loader.load_data(database_path, [task], [], 'csv', num_classes)

            # Filtering the raw data
            train_content = preprocessing.filter_data(train_content, filter_cuts, sample_frequency, filter_order, filter_type)

            # Normalize the filtered data
            train_content = preprocessing.normalize_data(train_content, normalize_type)

            list = []
            for index in range(0, len(train_content)):
                data = train_content[index]
                string = 'x_subject_' + str(index+1)
                savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv', data, fmt='%f', delimiter=';')
                print(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv was saved.')
                list.append(string+'.csv')
                
            savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + 'x_list.csv', [list], delimiter=',', fmt='%s')
            print(f'file names were saved to processed_data/task{task}/x_list.csv')

    # Getting the file names that contains the preprocessed data
    x_train_list = []

    for task in train_tasks:
        x_train_list.append(loadtxt(processed_data_path + 'processed_data/task'+str(task)+'/x_list.csv', delimiter=',', dtype='str'))

    x_train_list = [item for sublist in x_train_list for item in sublist]

    # Defining the data generators
    training_generator = data_manipulation.DataGenerator(x_train_list, batch_size, window_size, offset,
        num_channels, num_classes, train_tasks, 'train', split_ratio, processed_data_path, True)
    validation_generator = data_manipulation.DataGenerator(x_train_list, batch_size, window_size, offset,
        num_channels, num_classes, train_tasks, 'validation', split_ratio, processed_data_path, True)

    # Training the model
    if(not args.nofit):
        # Creating the model
        model = models.create_model_lstm_cnn(window_size, num_channels, num_classes)
        model.summary()

        # model.load_weights('model_weights.h5', by_name=True) ###### When the connection breaks ######

        # Compiling, defining the LearningRateScheduler and training the model
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

        fit_begin = time.time()

        results = model.fit(training_generator,
                            validation_data = validation_generator,
                            epochs = training_epochs,
                            callbacks = [lr_scheduler]
                            )

        fit_end = time.time()
        print(f'Training time in seconds: {fit_end - fit_begin}')
        print(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
        print(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

        # Summarize history for accuracy
        plt.subplot(211)
        plt.plot(results.history['accuracy'])
        plt.plot(results.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])

        # Summarize history for loss
        plt.subplot(212)
        plt.plot(results.history['loss'])
        plt.plot(results.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        plt.tight_layout()
        plt.savefig(r'accuracy-loss.png', format='png')
        plt.show()

        max_loss = np.max(results.history['loss'])
        min_loss = np.min(results.history['loss'])
        print("Maximum Loss : {:.4f}".format(max_loss))
        print("Minimum Loss : {:.4f}".format(min_loss))
        print("Loss difference : {:.4f}\n".format((max_loss - min_loss)))
        
        # Saving model weights
        model.save('model_weights.h5')
        print('model was saved to model_weights.h5.\n')

    # Running the model in Identification Mode
    if(not args.noimode):

        # Evaluate the model to see the accuracy
        if(model is None):
            model = models.create_model_lstm_cnn(window_size, num_channels, num_classes)
            model.summary()
            model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.load_weights('model_weights.h5', by_name=True)

        print('\nEvaluating on training set...')
        (loss, accuracy) = model.evaluate(training_generator, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on validation set...')
        (loss, accuracy) = model.evaluate(validation_generator, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on testing set...')
        test_begin = time.time()

        (loss, accuracy) = model.evaluate(x_test, y_test, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        test_end = time.time()
        print(f'Evaluating on testing set time in miliseconds: {(test_end - test_begin) * 1000.0}')
        print(f'Evaluating on testing set time in seconds: {test_end - test_begin}')
        print(f'Evaluating on testing set time in minutes: {(test_end - test_begin)/60.0}\n')
    
    # Running the model in Verification Mode
    if(not args.novmode):

        # Removing the last layers of the model and getting the features array
        model_for_verification = models.create_model_lstm_cnn(window_size, num_channels, num_classes, True)
        model_for_verification.summary()
        model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model_for_verification.load_weights('model_weights.h5', by_name=True)

        x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

        # Calculating EER and Decidability
        y_test_classes = utils.one_hot_encoding_to_classes(y_test)
        d, eer, thresholds = utils.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
        print(f'EER: {eer * 100.0} %')
        print(f'Decidability: {d}')
