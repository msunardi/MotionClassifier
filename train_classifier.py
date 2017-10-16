from __future__ import print_function
from dataset import *
from classifier import *
from util import elapsed
import keras


@elapsed
def run():
    """

    :return:
    """
    # base_path = '/home/mathias/PycharmProjects/MotionClassifier/'
    base_path = '/Users/Mathias/Documents/GitHub/MotionClassifier/'
    suffix = 'x3'
    # trainpath = '/home/mathias/Projects/motion_data/trainx.csv'
    # testpath = '/home/mathias/Projects/motion_data/testx.csv'
    # validationpath = '/home/mathias/Projects/motion_data/validationx.csv'
    trainpath = base_path + 'dataset/train%s.csv' % suffix
    testpath = base_path + 'dataset/test%s.csv' % suffix
    validationpath = base_path + 'dataset/validation%s.csv' % suffix
    model_path = base_path + '/models'

    learning_rate = 5e-1
    seq_length = 10
    data_dim = 8
    timesteps = seq_length
    num_classes = 2
    batch_size = 60
    hidden_size = 1024
    epochs = 20
    hidden_layers = 2

    retrain_it = 1  # times

    model_name = 'model_gru_15'  # Update this with every run

    inputs, targets = get_the_fing_data(trainpath)
    validate_data, validate_target = get_the_fing_data(validationpath)

    model = do_the_thing(inputs, targets, validate_data, validate_target,
                         data_dim, timesteps, num_classes, batch_size,
                         hidden_size, epochs, hidden_layers=hidden_layers, kind='GRU')
    test_the_thing(model, testpath, batch_size)
    # save_model(model, '{0}/{1}_{2}.h5'.format(model_path, model_name, 0))

    # Repeat and retrain
    for i in range(1, retrain_it):
        model = retrain_model(model=model, x_train=inputs, y_train=targets, x_val=validate_data, y_val=validate_target,
                              epochs=epochs, batch_size=batch_size)
        test_the_thing(model, testpath, batch_size)
        #save_model(model, '{0}/{1}_{2}.h5'.format(model_path, model_name, i))
    print("Done!")


if __name__ == "__main__":
    run()
