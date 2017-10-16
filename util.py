from __future__ import print_function
import time
from datetime import timedelta
import logging

# logging.basicConfig(filename='/home/mathias/Projects/jupyter_notebooks/motion_data/models/classifier_1.log', level=logging.DEBUG)
logging.basicConfig(filename='/home/mathias/PycharmProjects/MotionClassifier/logs/classifier_16.log', level=logging.DEBUG)


def elapsed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        lapse = end-start
        logging.info('[%s] - Elapsed: %s' % (str(func.__name__).upper(), str(timedelta(seconds=lapse))))
        return result
    return wrapper
