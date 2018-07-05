import os

import numpy as np
import matplotlib.pyplot as plt

CHANNELS = 4 # The number of eeg channels in the recordings
FREQ_BINS = 120 # The number of freq bins per channel
MAX_STD = 3 # The max stnadard deviation to crop outliers to
SLICE_SIZE = 4 # The number of samples the network will look at at once to make a prediction


def analyze_dataset(data, plot=False):
    columns = [[] for x in data[0]]

    # Add into columns for column based analysis
    for datapoint in data:
        for di, d in enumerate(datapoint):
            columns[di].append(d)

    # mins = []
    # maxs = []
    means = []
    stds = []
    for c in columns:
        # remove outliers
        for _ in range(len(c)//100):
            c.remove(max(c))
            c.remove(min(c))

        # mins.append(min(c))
        # maxs.append(max(c))
        mean = np.mean(c)
        means.append(mean)
        std = np.std(c)
        stds.append(std)


    # # Debug data
    # print("min", mins)
    # print("max", maxs)
    # print("mean", means)
    # print("std", stds)

    if plot:
        plt.legend(loc='upper left')

        plt.title("Mean Values")
        plt.xlabel("Freq Bin")
        plt.ylabel("Mean")
        for i in range(CHANNELS):
            plt.scatter(range(FREQ_BINS), means[i*FREQ_BINS:(i+1)*FREQ_BINS], label="Channel #{}".format(i))
        plt.show()

        plt.title("Standard Deviation")
        plt.xlabel("Freq Bin")
        plt.ylabel("STD")
        for i in range(CHANNELS):
            plt.scatter(range(FREQ_BINS), stds[i * FREQ_BINS:(i + 1) * FREQ_BINS], label="Channel #{}".format(i))
        plt.show()

    return means, stds


def normalize_dataset(dataset, mean, std):
    return [
        [max(min((value - mean[ind]) / std[ind], MAX_STD), -MAX_STD) for ind, value in enumerate(datapoint)]
        for datapoint in dataset]


def parse_file_list(filename_list):
    """
    Takes an eeg data filename list and parses it
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
    Takes an eeg data filename and parses it
    It does not format it for being read by the NN

    :param filename: filename of eeg data
    :type filename: str
    :return: tuple (data, label)
        WHERE
        list list float data is a list of (lists of data) in a recording
        list int       label is a list of (labels) for each recording
    """
    data = []
    labels = []
    with open(filename, "r") as f:

        for ind, line in enumerate(f):
            # remove brackets
            line = line.strip()[1:-1]

            # cut up and format
            data_point = list(map(float, line.split(", ")))

            # pull out label off of end
            # move label from <-1:1> to <0:2>
            label = int(data_point.pop(-1) + 1)
            labels.append(label)

            data.append(np.log(data_point))

            # # Debug data
            # print(ind/20.0, "label", label, "data", data)

    return data, labels


def slice_folder(folder_path, mean=None, std=None, plot_tag="", plot=False):
    """
    Takes an eeg data filename list and normalizes each row
    It combines each SLICE_SIZE number of recordings into a single network input
    Data and Labels are ready to be ingested by the network
    return looks like this
    [([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 2.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 2.0)]

    :param folder_path: path of folder of eeg data
    :type folder_path: str
    :param mean: list of means, one for each freq bin for each channel
    :type mean: list of float
    :param std: list of stds, one for each freq bin for each channel
    :type std: list of float
    :return: tuple(list of tuple (data, label), mean, std)
        WHERE
        list of float data is a list of (chunks of recordings)
        list of int  label is a list of (labels) for each chunk of recordings
        list of float mean is a list of means, one for each freq bin for each column
        list of float std is a list of stds, one for each freq bin for each column
    """
    filename_list = [folder_path + "/" + x for x in os.listdir(folder_path)]
    print("Parsing files:", filename_list)
    return slice_file_list(filename_list, mean, std, plot_tag, plot)


def slice_file_list(filename_list, mean=None, std=None, plot_tag="", plot=False):
    """
    Takes an eeg data filename list and normalizes each row
    It combines each SLICE_SIZE number of recordings into a single network input
    Data and Labels are ready to be ingested by the network
    return looks like this
    [([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 0.0), ([0.2, 0.6, 0.1], 2.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 1.0), ([0.2, 0.6, 0.1], 2.0)]

    :param filename_list: list of filenames of eeg data
    :type filename_list: list of str
    :param mean: list of means, one for each freq bin for each channel
    :type mean: list of float
    :param std: list of stds, one for each freq bin for each channel
    :type std: list of float
    :return: tuple(list of tuple (data, label), mean, std)
        WHERE
        list of float data is a list of (chunks of recordings)
        list of int  label is a list of (labels) for each chunk of recordings
        list of float mean is a list of means, one for each freq bin for each column
        list of float std is a list of stds, one for each freq bin for each column
    """
    data, labels = parse_file_list(filename_list)

    if mean is None or std is None:
        mean, std = analyze_dataset(data, plot)

    if plot:
        # plot x as time
        plt.legend(loc=None)

        plt.title(plot_tag + "Unnormalized vs Freq bin")
        plt.xlabel("Value")
        plt.ylabel("Freq bin & Channel")
        for i in range(0, len(data)):
            plt.scatter(range(len(data[i])), data[i], marker=".")
        plt.show()

        plt.title(plot_tag + "Unnormalized vs Time")
        plt.xlabel("Value")
        plt.ylabel("Time")
        for i in range(0, len(data[0])):
            plt.scatter(range(len(data)), [x[i] for x in data], marker=".")
        plt.show()

    data = normalize_dataset(data, mean, std)

    if plot:
        # plot x as time
        plt.legend(loc=None)

        plt.title(plot_tag + "Normalized vs Freq bin")
        plt.xlabel("Value")
        plt.ylabel("Freq bin & Channel")
        for i in range(0, len(data)):
            plt.scatter(range(len(data[i])), data[i], marker=".")
        plt.show()

        plt.title(plot_tag + "Normalized vs Time")
        plt.xlabel("Value")
        plt.ylabel("Time")
        for i in range(0, len(data[0])):
            plt.scatter(range(len(data)), [x[i] for x in data], marker=".")
        plt.show()

    return list(slice_data(data, labels)), mean, std


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
