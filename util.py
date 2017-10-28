from __future__ import print_function
import time
from datetime import timedelta, datetime
import logging
import os

now = datetime.now()

# Default logdir (Linux)
logdir = '/home/mathias/PycharmProjects/MotionClassifier/logs/'

lstm_id = '011'
gru_id = '020'
kind = 'GRU'

if os.name == 'nt':
    logdir = '/Users/Mathias/Documents/GitHub/MotionClassifier/logs/'
    # logging.basicConfig(filename='/Users/Mathias/Documents/GitHub/MotionClassifier/logs/classifier_18.log',
    #                     level=logging.DEBUG)

if kind == 'GRU':
    logging.basicConfig(
        filename=logdir + 'gru/{0}_{1}.log'.format(gru_id, now.strftime("%Y%m%d-%H%M%S")),
        level=logging.DEBUG)
elif kind == 'LSTM':
    logging.basicConfig(
        filename=logdir + 'lstm/{0}_{1}.log'.format(lstm_id, now.strftime("%Y%m%d-%H%M%S")),
        level=logging.DEBUG)


def elapsed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        lapse = end-start
        logging.info('[%s] - Elapsed: %s' % (str(func.__name__).upper(), str(timedelta(seconds=lapse))))
        return result
    return wrapper
