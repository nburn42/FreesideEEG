import os

import numpy as np
# import matplotlib.pyplot as plt

CHANNELS = 4 # The number of eeg channels in the recordings
FREQ_BINS = 120 # The number of freq bins per channel
MAX_STD = 3 # The max stnadard deviation to crop outliers to
SLICE_SIZE = 4 # The number of samples the network will look at at once to make a prediction


def parse_file_list(filename_list):
    """
    Takes an eeg data filename list and normalizes each row
    It does not format it for being read by the NN

    :param filename_list: list of filenames of eeg data
    :type filename_list: list of str
    :return: tuple (data, label)
        WHERE
        list list float data is a list of (lists of data) in a recording
        list int       label is a list of (labels) for each recording
    """
    data = []
    labels = []
    for filename in filename_list:
        d, l = parse_file(filename)
        data.extend(d)
        labels.extend(l)
    return data, labels


def parse_file(filename):
    """
    Takes an eeg data filename and normalizes each row
    It does not format it for being read by the NN

    :param filename: filename of eeg data
    :type filename: str
    :return: tuple (data, label)
        WHERE
        list list float data is a list of (lists of data) in a recording
        list int       label is a list of (labels) for each recording
    """
    labels = []
    with open(filename, "r") as f:
        columns = [[] for x in range(CHANNELS * FREQ_BINS)]

        for ind, line in enumerate(f):
            # # remove first 50 time steps twhere filter is filling up
            # if ind < START_CUT:
            #     continue
            #remove brakets
            line = line.strip()[1:-1]
            #cut up and format
            data = list(map(float, line.split(", ")))
            # pull out label off of end
            label = data.pop(-1) + 1 # move label from <-1:1> to <0:2>
            labels.append(label)
            # Add into into columns for column based normalization
            for di, d in enumerate(data):
                columns[di].append(d)
            # Debug data
            # print(ind/20.0, "label", label, "data", data)


        # mins = []
        # maxs = []
        means = []
        stds = []
        # column based
        actual_data = []
        for c in columns:
            # mins.append(min(c))
            # maxs.append(max(c))
            mean = np.mean(c)
            means.append(mean)
            std = np.std(c)
            stds.append(std)
            actual_data.append([min((cc - mean) / std, MAX_STD) for cc in c])


        # row based
        actual_data_rows = [[] for x in range(len(actual_data[0]))]
        for col in actual_data:
            for ind, c in enumerate(col):
                actual_data_rows[ind].append(c)

        # Debug data

        # print("min", mins)
        # print("max", maxs)
        # print("mean", means)
        # print("std", stds)

        # plt.scatter(range(len(columns)), maxs)
        # plt.scatter(range(len(columns))[:120], stds[:120])

        # # plot x as time samples
        # for i in range(0, 120, 10):
        #     for j in range(4):
        #         plt.scatter(range(len(actual_data[0])), actual_data[i + (j * 120)])

        # # plot x as freq bins
        # for i in range(0, len(actual_data_rows), 20):
        #     plt.scatter(range(len(actual_data_rows[0])), actual_data_rows[i])
        #
        # plt.show()
        return actual_data_rows, labels


def slice_folder(folder_path):
    """
    Takes an eeg data filename list and normalizes each row
    It combines each SLICE_SIZE number of recordings into a single network input
    Data and Labels are ready to be ingested by the network
    return looks like this
    [([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 2.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 2.0)]

    :param folder: path of folder of eeg data
    :type folder: str
    :return: tuple (data, label)
        WHERE
        list of float data is a list of (chunks of recordings)
        list of int   label is a list of (labels) for each chunk of recordings
    """
    filename_list = [folder_path + "/" + x for x in os.listdir(folder_path)]
    print("Parsing files:", filename_list)
    return slice_file_list(filename_list)


def slice_file_list(filename_list):
    """
    Takes an eeg data filename list and normalizes each row
    It combines each SLICE_SIZE number of recordings into a single network input
    Data and Labels are ready to be ingested by the network
    return looks like this
    [([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 2.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 2.0)]

    :param filename_list: list of filenames of eeg data
    :type filename_list: list of str
    :return: list of tuple (data, label)
        WHERE
        list of float data is a list of (chunks of recordings)
        list of int   label is a list of (labels) for each chunk of recordings
    """
    unsliced_data, labels = parse_file_list(filename_list)
    return slice_data(unsliced_data, labels)


def slice_file(filename):
    """
        Takes an eeg data filename and normalizes each row
        It combines each SLICE_SIZE number of recordings into a single network input
        Data and Labels are ready to be ingested by the network
        return looks like this
        [([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 2.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 2.0)]

        :param filename_list: list of filenames of eeg data
        :type filename: str
        :return: list of tuple (data, label)
            WHERE
            list float data is a list of (chunks of recordings)
            list int   label is a list of (labels) for each chunk of recordings
        """
    unsliced_data, labels = parse_file(filename)
    return slice_data(unsliced_data, labels)


def slice_data(data, labels):
    """
        Takes normalized eeg data list and formats it to be ingested by the network
        It combines each SLICE_SIZE number of recordings into a single network input
        It takes the last label from each chunk
        Data and Labels are ready to be ingested by the network
        return looks like this
        [([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 2.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 2.0)]

        :param data: list of eeg recordings
        :type data: list of list of float
        :param labels: list of labels for each recording
        :type labels: list of int
        :return: list of tuple (data, label)
            WHERE
            list float data is a list of (chunks of recordings)
            list int   label is a list of (labels) for each chunk of recordings
        """
    assert len(labels) == len(data)
    sliced_data = [flatten_list(data[i - SLICE_SIZE:i]) for i in range(SLICE_SIZE, len(data), SLICE_SIZE)]
    sliced_labels = [labels[i - 1] for i in range(SLICE_SIZE, len(data), SLICE_SIZE)]
    return zip(sliced_data, sliced_labels)


def flatten_list(list_of_lists):
    """
    Takes a list of lists
    flattens them into a single list
    :param list_of_lists: a list of lists
    :type list_of_lists: list of list of float
    :return flattened_list: a flattened list
    """
    return [element_in_list_in_list for list_in_list in list_of_lists for element_in_list_in_list in list_in_list]
