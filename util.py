from __future__ import print_function
import time
from datetime import timedelta, datetime
import logging
import os

now = datetime.now()

if os.name == 'nt':
    logging.basicConfig(filename='/Users/Mathias/Documents/GitHub/MotionClassifier/logs/classifier_18.log',
                        level=logging.DEBUG)
else:
    logging.basicConfig(
        filename='/home/mathias/PycharmProjects/MotionClassifier/logs/gru/01_%s.log' % now.strftime("%Y%m%d-%H%M%S"),
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
