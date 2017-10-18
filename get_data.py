"""
Separating raw dataset into training, test, and validation sets
"""
from __future__ import print_function

import logging
import numpy as np
from math import floor

from util import elapsed
from dataset import *
# from classifier import get_the_fing_data
logging.basicConfig(filename='/home/mathias/PycharmProjects/MotionClassifier/logs/dataset.log', level=logging.DEBUG)


@elapsed
def run():
    # get_newdata()
    # get_derivative('synthetic.csv', save=True)
    # get_the_fing_data('/home/mathias/PycharmProjects/MotionClassifier/dataset/synthetic_derivative.csv')
    new_dataset()

@elapsed
def get_newdata():
    logging.info("Collecting raw data into datasets...")
    # get_data(new_filename='synthetic.csv', path='/home/mathias/catkin_ws/src/motion_generator/src/csv/*.csv')
    get_data(new_filename='mocap.csv', path='/home/mathias/Projects/Blender/dataset2/*.csv')

@elapsed
def new_dataset():
    logging.info("Getting new datasets ...")
    # Pick the source/raw datasets
    # First derivatives
    # generated = np.genfromtxt('/home/mathias/catkin_ws/src/motion_generator/src/csv/combined_derivative.csv',
    #                           delimiter=',',
    #                           skip_header=0,
    #                           skip_footer=0)
    # mocap = np.genfromtxt('/home/mathias/Projects/Blender/combined/combined_10_derivative.csv', delimiter=',',
    #                       skip_header=0,
    #                       skip_footer=0)

    generated = np.genfromtxt('/home/mathias/PycharmProjects/MotionClassifier/dataset/synthetic_derivative.csv',
                              delimiter=',',
                              skip_header=0,
                              skip_footer=0, dtype=None)
    mocap = np.genfromtxt('/home/mathias/PycharmProjects/MotionClassifier/dataset/mocap_derivative.csv',
                          delimiter=',',
                          skip_header=0,
                          skip_footer=0, dtype=None)

    # Second derivatives
    # generated = np.genfromtxt('/home/mathias/catkin_ws/src/motion_generator/src/csv/combined_derivative_derivative.csv',
    #                           delimiter=',', skip_header=0, skip_footer=0)
    # mocap = np.genfromtxt('/home/mathias/Projects/Blender/dataset2/combined_10_derivative_derivative.csv',
    #                       delimiter=',', skip_header=0, skip_footer=0)
    logging.info(str(generated.shape))
    logging.info(str(mocap.shape))
    gen = [list(g) for g in generated]
    moc = [list(m) for m in mocap]
    # generated_indices = generated.shape[0]
    # mocap_indices = mocap.shape[0]
    sequence_size = 10
    test_validation_ratio = 0.07
    total_points = min(len(gen), len(moc))
    data_dimensions = 8

    # total_points is used to balance the +/- samples
    # the maximum number of points is 2xtotal_points; use only part of that
    tp = 1.9 * total_points

    # Determine data size based on batch_size
    # e.g. for 2000 total points, batch_size=20, there will be (2000 - 2000%20)/20 = 100 data points; each a sequence of 20 points
    # for 2034 total points, batch_size=20, there will be (2034 - 2034%20)/20 = 101 data points

    test_size = int(floor(test_validation_ratio * tp))
    validation_size = int(floor(test_validation_ratio * tp))
    train_size = int((tp - test_size - validation_size))

    print("Test points: %s" % test_size)
    print("Validation points: %s" % validation_size)
    print("Train points: %s" % train_size)

    logging.info("Test points: %s" % test_size)
    logging.info("Validation points: %s" % validation_size)
    logging.info("Train points: %s" % train_size)

    # Run it!
    logging.info("Before: Generated: %s, Motion capture: %s" % (len(gen), len(moc)))

    train_data, gen, moc = collect_data(gen, moc, train_size, sequence_size, data_dimensions)
    test_data, gen, moc = collect_data(gen, moc, test_size, sequence_size, data_dimensions)
    validation_data, gen, moc = collect_data(gen, moc, validation_size, sequence_size, data_dimensions)

    print("Train data size: %s" % (len(train_data)))
    print("Test data size: %s" % (len(test_data)))
    print("Validation data size: %s" % (len(validation_data)))
    print("After: Generated: %s, Motion capture: %s" % (len(gen), len(moc)))

    logging.info("Train data size: %s" % (len(train_data)))
    logging.info("Test data size: %s" % (len(test_data)))
    logging.info("Validation data size: %s" % (len(validation_data)))
    logging.info("After: Generated: %s, Motion capture: %s" % (len(gen), len(moc)))

    save_path = '/home/mathias/PycharmProjects/MotionClassifier/dataset/'
    suffix = 'x4'
    save_data(train_data, 'train%s.csv' % suffix, path=save_path)
    save_data(test_data, 'test%s.csv' % suffix, path=save_path)
    save_data(validation_data, 'validation%s.csv' % suffix, path=save_path)
    # save_data(train_data, 'train_derivative.csv')
    # save_data(test_data, 'test_derivative.csv')
    # save_data(validation_data, 'validation_derivative.csv')
    logging.info("Done!")


if __name__ == "__main__":
    run()
