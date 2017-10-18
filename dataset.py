from __future__ import print_function
import csv
from glob import glob
import numpy as np
import random as rn
from util import elapsed
import logging

logging.basicConfig(filename='/home/mathias/PycharmProjects/MotionClassifier/logs/dataset.log', level=logging.DEBUG)

motion_capture_path = '/home/mathias/Projects/Blender/dataset2/*.csv'
generated_motion_path = '/home/mathias/catkin_ws/src/motion_generator/src/csv/*.csv'


@elapsed
def get_data(new_filename=None, batch_size=20, path='/home/mathias/Projects/Blender/dataset2/*.csv', target_path=None, keep_header=False):
    """
    get_data

    Collects clips of motion data and combine them into one file
    """
    files = glob(path)

    base_path = path.split('*.csv')[0]

    if not new_filename:
        new_filename = 'combined.csv'
    index = 0
    while base_path + new_filename in files:
        index += 1
        new_filename = 'combined_%s.csv' % index

    if not target_path:
        target_path = '/home/mathias/PycharmProjects/MotionClassifier/dataset/'
    new_filename = target_path + new_filename

    new_data = []

    for f in files:
        with open(f, 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            # counter for batch for each file
            # Only add to combined file in multiples of batch_size
            # s = 0
            # batch = []
            #
            # for row in csvreader:
            #     try:
            #         np.array(row, dtype=np.float)
            #         if s % batch_size == 0 and len(batch) > 0:
            #             for b in batch:
            #                 new_data.append(b)
            #             batch = []
            #         else:
            #             batch.append(row)
            #     except ValueError as v:
            #         continue
            #
            #     s += 1
            s = 1
            for row in csvreader:
                line = []
                try:
                    np.array(row, dtype=np.float)
                    fname = f.split('/')[-1]

                    line += row
                    line.append(fname)
                    line.append(s)
                    new_data.append(line)
                except ValueError as v:
                    print("Error: %s" % v)
                    continue
                s += 1

    with open(new_filename, 'w') as newcsv:
        newwriter = csv.writer(newcsv, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_NONE)
        logging.info('Writing new data:')
        logging.info('Filename: %s' % new_filename)
        logging.info('Data: %s lines' % len(new_data))
        try:
            newwriter.writerows(new_data)
            logging.info('Success.')
        except IOError as io:
            logging.error('Something went wrong ...')


@elapsed
def get_derivative(filename, path=None, save=True):
    """
    Calculate one derivative from a given dataset
    Apply multiple times to get higher order derivatives
    """

    if not path:
        # path = '/home/mathias/Projects/Blender/dataset2/'
        path = '/home/mathias/PycharmProjects/MotionClassifier/dataset/'

    if '.csv' not in filename:
        raise ValueError('Must be a .csv file')
        return None

    fullpath = path + filename
    with open(fullpath, 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Get the first line to calculate the receptacle (ndarray)
        yo = reader.next()

        dofs = len(yo) - 2
        blah = np.array(yo[:-2], dtype=np.float).reshape(dofs, 1)
        diff_array = np.ndarray((dofs, 0))
        diff_array = np.append(diff_array, blah, 1)

        count = 0
        labels = []
        labels.append(yo[-2:])

        for row in reader:
            # print("Row: {0}".format(row))
            #             logging.info dofs
            r = np.array(row[:-2], dtype=np.float)
            #             logging.info r
            try:
                r.resize(len(row)-2, 1)
                diff_array = np.append(diff_array, r, axis=1)
                labels.append(row[-2:])
            except ValueError as e:
                logging.info(row)
                logging.error(e)
                rowx = [v for v in row[:-2]]
                x = np.array(rowx, dtype=np.float).reshape(len(rowx), 1)
                r = np.resize(x, (diff_array.shape[0], 1))
                logging.info(r.shape)
                diff_array = np.append(diff_array, r, axis=1)
                count += 1
                continue

    xdiff = np.diff(diff_array).T

    # Attach labels back to data
    diff = []
    for x, y in zip(xdiff, labels):
        z = [v for v in x] + y
        diff.append(z)

    logging.info("DIFF:\n{0}".format(diff[:10]))

    if save:
        try:
            diff_file = path + filename.split('.csv')[0] + '_derivative.csv'
            with open(diff_file, 'w') as diffcsv:
                diffwriter = csv.writer(diffcsv, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_NONE)
                diffwriter.writerows(diff)
            logging.info("Successfully written to: %s" % diff_file)
        except IOError as e:
            logging.error(e)

    logging.info("Diff shape: (%s x %s)" % (len(diff), len(diff[0])))
    if count > 0:
        logging.info('Errors encountered: %s' % count)

    return diff

@elapsed
def collect_data(generated_source, mocap_source, N, batch_size, data_dimensions):
    """
    From raw dataset, collect into sets
    Since separating into train, test, and validation sets must be unique,
    during collection, data points that are selected are removed from the source datasets
    :param generated_source: raw data of generated motion type
    :param mocap_source: raw data of motion capture type
    :param N: size of dataset
    :param batch_size: size of batch (sequence)
    :return out: collected data, gen: reduced generated motion data, moc: reduced motion capture data
    """
    print("Collecting data:\ngen_source: {0}, moc_source: {1}, N: {2}, batch: {3}, dim: {4}".format(
        len(generated_source), len(mocap_source), N, batch_size, data_dimensions
    ))
    is_generated = True  # start with generated data
    gen = generated_source
    moc = mocap_source
    out = []
    while len(out) <= N:
        sequence = []
        # Pick a random point in the dataset,
        # but make sure there are <sequence_size> datapoints from that point
        if is_generated:
            source = gen
            # have_picked = seen['generated']
            #             klass = [1.0, 0.0]   # Or just [1.0] vs [0.0] for binary classifications
            klass = [1.0]
        else:
            source = moc
            # have_picked = seen['mocap']
            #             klass = [0.0, 1.0]
            klass = [0.0]

        is_generated = not is_generated
        #         source = [s for s in source]
        l = len(source)

        # Get size of each row
        #         source_size = len(source[0])

        p = rn.choice(range(l))
        if p < l - batch_size:  # or p in have_picked:

            # Record points that have been used/picked
            # have_picked.append(p)

            # Collect sequence from point p of <sequence_size>
            # sequence = source[p:p+batch_size]
            # Remove picked points

            try:
                # If there are more dimensions than data_dimensions, try randomly pick sub-dimension
                data_width = len(source[0])
                if data_width > data_dimensions + 3:  # +3 to account for class value and labels
                    j = rn.choice(range(len(source[0])-(batch_size+1)))
                else:
                    j = 0

                for i in range(batch_size):
                    #                     logging.info("source [%s]: %s" % (p+i, source[p+i]))
                    popp = source.pop(p + i)
                    out.append(popp[j:j+data_dimensions] + popp[-2:] + klass)
            except IndexError as e:
                logging.error("Reached end of source.")
                continue
            # out.append(sequence)
            delimiter = [0.0] * (data_dimensions+2) + klass  # +2 to include labels
            out.append(delimiter)
            #         logging.info(len(out))
    return out, gen, moc


@elapsed
def save_data(data, filename, path='/home/mathias/Projects/motion_data/'):
    """
    Saves data to a .csv file
    :param data:
    :param filename:
    :param path:
    :return None:
    """
    logging.info("Saving: %s" % filename)
    logging.info("Data size: %s x %s" % (len(data), len(data[0])))
    try:
        save_file = path + filename.split('.csv')[0] + '.csv'
        with open(save_file, 'w') as savecsv:
            writer = csv.writer(savecsv, delimiter=',',  # escapechar='\\',
                                quotechar='|', quoting=csv.QUOTE_NONE)
            writer.writerows(data)
        logging.info( "Successfully written %s lines to: %s" % (len(data), save_file))
    except IOError as e:
        logging.error(e)

# # Pick the source/raw datasets
# # First derivatives
# generated = np.genfromtxt('/home/mathias/catkin_ws/src/motion_generator/src/csv/combined_derivative.csv',
#                           delimiter=',',
#                           skip_header=1,
#                           skip_footer=0)
# mocap = np.genfromtxt('/home/mathias/Projects/Blender/dataset2/combined_10_derivative.csv', delimiter=',',
#                       skip_header=1,
#                       skip_footer=0)
# # Second derivatives
# # generated = np.genfromtxt('/home/mathias/catkin_ws/src/motion_generator/src/csv/combined_derivative_derivative.csv', delimiter=',', skip_header=1,
# #                          skip_footer=0)
# # mocap = np.genfromtxt('/home/mathias/Projects/Blender/dataset2/combined_10_derivative_derivative.csv', delimiter=',', skip_header=1,
# #                      skip_footer=0)
# logging.info generated.shape
# logging.info mocap.shape
# gen = [list(g) for g in generated]
# moc = [list(m) for m in mocap]
# generated_indices = generated.shape[0]
# mocap_indices = mocap.shape[0]
# sequence_size = 10
# test_validation_ratio = 0.07
# total_points = min(len(gen), len(moc))
# data_dimensions = 8
#
# # total_points is used to balance the +/- samples
# # the maximum number of points is 2xtotal_points; use only part of that
# tp = 1.8 * total_points
#
# # Determine data size based on batch_size
# # e.g. for 2000 total points, batch_size=20, there will be (2000 - 2000%20)/20 = 100 data points; each a sequence of 20 points
# # for 2034 total points, batch_size=20, there will be (2034 - 2034%20)/20 = 101 data points
#
# test_size = int(floor(test_validation_ratio * tp))
# validation_size = int(floor(test_validation_ratio * tp))
# train_size = int((tp - test_size - validation_size))
#
# logging.info "Test points: %s" % test_size
# logging.info "Validation points: %s" % validation_size
# logging.info "Train points: %s" % train_size
#
# train_data = []
# test_data = []
# validation_data = []
#
#
# # Run it!
# logging.info "Before: Generated: %s, Motion capture: %s" % (len(gen), len(moc))
#
# train_data = collect_data(gen, moc, train_size, sequence_size)
# test_data = collect_data(gen, moc, test_size, sequence_size)
# validation_data = collect_data(gen, moc, validation_size, sequence_size)
#
# logging.info "Train data size: %s" % (len(train_data))
# logging.info "Test data size: %s" % (len(test_data))
# logging.info "Validation data size: %s" % (len(validation_data))
# logging.info "After: Generated: %s, Motion capture: %s" % (len(gen), len(moc))
#
# save_data(train_data, 'trainx.csv')
# save_data(test_data, 'testx.csv')
# save_data(validation_data, 'validationx.csv')
# # save_data(train_data, 'train_derivative.csv')
# # save_data(test_data, 'test_derivative.csv')
# # save_data(validation_data, 'validation_derivative.csv')
# logging.info "Done!"