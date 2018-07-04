import tensorflow as tf
import random
import shutil, errno

def get_accuracy_op(logits, labels):
    correct_prediction = tf.equal(tf.cast(tf.round(logits), tf.int64), tf.cast(labels, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy_op


def get_batch(dataset, batch_size):
    batch_data, batch_labels = zip(*random.sample(dataset, batch_size))
    return batch_data, batch_labels


def copyRecursive(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def get_shape_list(tensor):
    return [dim.value for dim in tensor.get_shape()]