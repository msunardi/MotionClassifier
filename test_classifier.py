from __future__ import print_function
from classifier import *
from util import elapsed
import os

@elapsed
def run():
    """

    :return:
    """
    base_path = '/home/mathias/PycharmProjects/MotionClassifier/'
    if os.name == 'nt':
        base_path = '/Users/Mathias/Documents/GitHub/MotionClassifier/'

    suffix = 'x7'
    # trainpath = base_path + 'dataset/train%s.csv' % suffix
    # testpath = base_path + 'dataset/test%s.csv' % suffix
    # validationpath = base_path + 'dataset/validation%s.csv' % suffix

    testpath = base_path + 'dataset/kinect_test2.csv'
    rnnkind = 'GRU'

    # rnnkind = 'LSTM'
    # gru_id = '015'
    # lstm_id = '09'
    model_path = None
    model_name = None
    model = rnnkind

    if rnnkind == 'GRU':
        model_path = base_path + '/models/gru'
        model_name = 'model_gru_' + gru_id  # Update this with every run_gru_15
    elif rnnkind == 'LSTM':
        model_path = base_path + '/models/lstm'
        model_name = 'model_lstm_' + lstm_id  # Update this with every run_gru_15

    # Parameters
    learning_rate = 5e-1
    seq_length = 24
    data_dim = 16
    timesteps = seq_length
    num_classes = 2
    batch_size = 60
    hidden_size = 128
    epochs = 150
    hidden_layers = 3
    retrain_it = 0 # times

    # Load the model

    model0 = load_model('./models/gru/model_gru_023_0.h5')

    test_the_thing(model0, testpath, batch_size, timesteps=timesteps, features=data_dim)

    print("Done!")


if __name__ == "__main__":
    run()
