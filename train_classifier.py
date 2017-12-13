from __future__ import print_function
from classifier import *
from util import elapsed
import os

# gru_id = '015'
# lstm_id = '09'
#
# rnnkind = 'LSTM'

@elapsed
def run():
    """

    :return:
    """
    base_path = '/home/mathias/PycharmProjects/MotionClassifier/'
    if os.name == 'nt':
        base_path = '/Users/Mathias/Documents/GitHub/MotionClassifier/'

    suffix = 'x7'
    # trainpath = '/home/mathias/Projects/motion_data/trainx.csv'
    # testpath = '/home/mathias/Projects/motion_data/testx.csv'
    # validationpath = '/home/mathias/Projects/motion_data/validationx.csv'
    trainpath = base_path + 'dataset/train%s.csv' % suffix
    testpath = base_path + 'dataset/test%s.csv' % suffix
    validationpath = base_path + 'dataset/validation%s.csv' % suffix

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

    inputs, targets = get_the_fing_data(trainpath, timesteps=timesteps, features=data_dim)
    validate_data, validate_target = get_the_fing_data(validationpath, timesteps=timesteps, features=data_dim)

    model = do_the_thing(inputs, targets, validate_data, validate_target,
                         data_dim, timesteps, num_classes, batch_size,
                         hidden_size, epochs, hidden_layers=hidden_layers, kind=model)
    test_the_thing(model, testpath, batch_size, timesteps=timesteps, features=data_dim)
    save_model(model, '{0}/{1}_{2}.h5'.format(model_path, model_name, 0))

    # Repeat and retrain
    # for i in range(1, retrain_it+1):
    #     model = retrain_model(i, model=model, x_train=inputs, y_train=targets,
    #                           x_val=validate_data, y_val=validate_target,
    #                           epochs=epochs, batch_size=batch_size)
    #     test_the_thing(model, testpath, batch_size)
    #     save_model(model, '{0}/{1}_{2}.h5'.format(model_path, model_name, i))
    print("Done!")


if __name__ == "__main__":
    run()
