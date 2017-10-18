from __future__ import print_function
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Dropout, SimpleRNN
from keras.models import load_model
from keras.callbacks import *
import numpy as np
import csv
import random as r
from util import elapsed
import logging
import json

import os

# Ref: https://stackoverflow.com/a/40164869
from datetime import datetime
now = datetime.now()

# If Windows
if os.name == 'nt':
    logging.basicConfig(filename='/Users/Mathias/Documents/GitHub/MotionClassifier/logs/classifier_18.log',
                        level=logging.DEBUG)
else:
    logging.basicConfig(filename='/home/mathias/PycharmProjects/MotionClassifier/logs/gru/01_%s.log' % now.strftime("%Y%m%d-%H%M%S"), level=logging.DEBUG)

run_path = 'runs/gru_01'

# tb = TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True,
#                  write_images=False, embeddings_freq=0, embeddings_layer_names=None,
#                  embeddings_metadata=None)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

kind = 'gru'
csv_logger = CSVLogger('logs/%s/training01.log' % kind)


def get_the_fing_data(filepath):
    log_msg = "[GET DATA] "
    data = []
    target = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        tmp_data = []
        tmp_target = []
        #     reader = csv.DictReader(csvfile, fieldnames=['D%s'% x for x in range(8)]+['Class'])
        #     train_data = [tr for tr in reader if tr[:-1] != ['0.0','0.0','0.0','0.0','0.0','0.0','0.0','0.0']]

        for tr in reader:

            # tr[:-n] where n depends on how the collated raw data is saved;
            # .. new data format ends with <filename>, <frame number>, <class> so n = 3
            # .. otherwise, n = 1 (no filename or frame number)
            if tr[:-3] == ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']:
                # Has to have the correct shape (10,8)
                if np.array(tmp_data).shape == (10, 8):
                    data.append(list(tmp_data))
                    target.append(list(tmp_target))
                tmp_data = []
                tmp_target = []

            else:
                tmp_data.append([float(i) for i in tr[:-3]])
                #tmp_target.append([float(tr[-1])])

                print(tr)
                # One-hot version
                if tr[-1] == "0.0":
                    tx = [1.0, 0.0]
                elif tr[-1] == "1.0":
                    tx = [0.0, 1.0]
                tmp_target.append(tx)
                tmp_target.append(tr[-3:-2]) # append filename and frame number for analysis


    fdata = np.array(data)
    ftarget = np.array(target)
    msg = "file: {0}, data: {1} - target: {2}".format(filepath, str(fdata.shape), str(ftarget.shape))
    logging.info(log_msg + msg)

    return fdata, ftarget


def shuffle(data, targets):
    n = len(data)
    d_data = data
    t_targets = targets
    shuffled_data = []
    shuffled_targets = []
    while len(d_data) > 1 and len(t_targets) > 1:
        pick = r.choice(range(len(d_data)))
        mask = np.ones(len(d_data), dtype=bool)
        mask[[pick]] = False
        shuffled_data.append(d_data[pick])
        shuffled_targets.append(t_targets[pick])
        d_data = d_data[mask,...]
        t_targets = t_targets[mask,...]
    shuffled_data.append(d_data[0])
    shuffled_targets.append(t_targets[0])
    return np.array(shuffled_data), np.array(shuffled_targets)


@elapsed
def retrain_model(i, x_train, y_train, x_val, y_val, epochs, model=None, model_source=None, batch_size=50):
    """
    Retrain a saved model
    :param x_train: training data
    :param y_train: training targets
    :param x_val: validation data
    :param y_val: validation targets
    :param epochs: number of runs
    :param model: a model to retrain
    :param model_source: path to saved model e.g. 'rnn_results/motion_derivative_dataset_model0.h5'
    :param batch_size: number of data points to process in one epoch
    :return model0: the retrained model
    """
    log_msg = "[RETRAIN] "
    if not model and not model_source:
        logging.error(log_msg + "You gotta give either a model or a path to a saved model!")

    model0 = model
    if model_source:
        model0 = load_model(model_source)

    x_retrain, y_retrain = shuffle(x_train, y_train)
    x_reval, y_reval = shuffle(x_val, y_val)

    train_size = len(x_train) - (len(x_train) % batch_size)
    x_retrain = np.array(x_retrain)[:train_size]
    y_retrain = np.array(y_retrain)[:train_size]

    val_size = len(x_val) - (len(x_val) % batch_size)
    x_reval = np.array(x_reval)[:val_size]
    y_reval = np.array(y_reval)[:val_size]

    tbx = TensorBoard(log_dir='logs/%s/retrain_lstm_%s/' % (run_path, i), histogram_freq=0, write_graph=True,
                     write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    model0.fit(x_retrain, y_retrain,
               batch_size=batch_size, epochs=epochs, shuffle=True,
               validation_data=(x_reval, y_reval), callbacks=[tbx, reduce_lr, csv_logger])

    logging.info(model0.summary())
    return model0


def do_the_thing(train_data, train_target, validation_data, validation_target, data_dim, timesteps,
                 num_classes, batch_size, hidden_size, epochs, hidden_layers, kind='LSTM'):
    log_msg = "[DO THE THING] "
    inputs, targets = shuffle(train_data, train_target)
    validate_data, validate_target = shuffle(validation_data, validation_target)

    # learning_rate = 5e-1
    # seq_length = 10
    # data_dim = 8
    # timesteps = seq_length
    # num_classes = 1
    # batch_size = 50
    # hidden_size = 512
    # epochs = 20

    # Expected input batch shape: (batch_size, timesteps, data_dim)
    # Note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.

    modelx = Sequential()
    if kind == 'LSTM':
        modelx.add(LSTM(hidden_size, return_sequences=True,  # stateful=True,
                        batch_input_shape=(batch_size, timesteps, data_dim)))
        for i in range(hidden_layers):
            modelx.add(LSTM(hidden_size, return_sequences=True, stateful=True))
            # modelx.add(LSTM(hidden_size, return_sequences=True, stateful=True))
    elif kind == 'GRU':
        modelx.add(GRU(hidden_size, return_sequences=True,  # stateful=True,
                        batch_input_shape=(batch_size, timesteps, data_dim)))
        for i in range(hidden_layers):
            modelx.add(GRU(hidden_size, return_sequences=True, stateful=True))
            # modelx.add(GRU(hidden_size, return_sequences=True, stateful=True, ))
    modelx.add(Dropout(0.2))
    # model3.add(GRU(hidden_size, return_sequences=True, stateful=True))
    # model3.add(GRU(hidden_size, return_sequences=True, stateful=True))
    # model3.add(LSTM(v_size, return_sequences=True, stateful=True, activation='softmax'))

    # Pair sigmoid with mse
    # modelx.add(Dense(num_classes, activation='sigmoid'))
    # modelx.compile(loss='mean_squared_error',
    #                optimizer='rmsprop', metrics=['accuracy'])

    # Use for num_classes > 1
    modelx.add(Dense(num_classes, activation='softmax'))

    modelx.compile(loss='categorical_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

    # Generate dummy training data
    # shape: (#batch*x, sequence_length, input_vector_size)
    train_size = len(inputs) - (len(inputs) % batch_size)
    x_train = np.array(inputs)[:train_size]
    y_train = np.array(targets)[:train_size]

    val_size = len(validate_data) - (len(validate_data) % batch_size)
    x_val = np.array(validate_data)[:val_size]
    y_val = np.array(validate_target)[:val_size]

    # # Generate dummy validation data
    # x_val = np.random.random((batch_size * 3, timesteps, data_dim))
    # y_val = np.random.random((batch_size * 3, num_classes))
    logging.info(log_msg + str(x_train.shape))
    logging.info(log_msg + str(y_train.shape))

    tb = TensorBoard(log_dir='logs/%s/' % run_path, histogram_freq=0, write_graph=True,
                     write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    modelx.fit(x_train, y_train,
               batch_size=batch_size, epochs=epochs, shuffle=True,
               validation_data=(x_val, y_val), callbacks=[tb, reduce_lr, csv_logger])

    modelx.summary()
    logging.info(json.dumps(modelx.to_json()))
    return modelx


def save_model(model, path):
    log_msg = "[SAVE_MODEL] "
    logging.info(log_msg + "Saving model to %s ..." % path)
    try:
        model.save(path)
        logging.info(log_msg + "Model saved.")
    except Exception as e:
        logging.error(log_msg + "Something went wrong: %s" % e.message)


@elapsed
def test_the_thing(model, test_source=None, batch_size=50, save=True):
    """
    Test a trained model
    :param model:
    :param test_source:
    :param batch_size: batch of data points (not sequence length)
    :return:
    """
    log_msg = "[TEST] "
    model = model
    testpath = '/home/mathias/Projects/motion_data/testx.csv'
    if test_source:
        testpath = test_source
    test_data, test_target = get_the_fing_data(testpath)

    test_size = len(test_data) - (len(test_data) % batch_size)
    x_test = np.array(test_data)[:batch_size]
    y_test = np.array(test_target)[:batch_size]

    predictions = model.predict_on_batch(x_test)

    # print predictions
    # print y_test
    correct = 0
    wrong = 0

    for pred, yak in zip(predictions, y_test):
        # p = np.mean(pred)
        # y = np.mean(yak)
        # d = abs(p-y)
        d, p, y = prediction_diff(yak, pred)
        show = "p vs y: %s - %s --> (abs) %s" % (p, y, d)
        if d < 0.3 :
            correct += 1
            show += '*'
        else:
            wrong += 1
        print(show)
        logging.info(log_msg + show)

    acc = correct*1.0/(correct+wrong)
    logging.info(log_msg + "Correct: %s vs. Wrong: %s" % (correct, wrong))
    logging.info(log_msg + "Accuracy: {:.04f} ({:.02f}%)".format(acc, acc*100))


def prediction_diff(target, prediction):
    p = np.mean(prediction, axis=0)
    y = np.mean(target, axis=0)
    d = np.mean(np.abs(np.diff([y, p], axis=0)))
    return d, p, y
