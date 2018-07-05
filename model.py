import numpy as np
import tensorflow as tf

import tf_util


def basic_model(channel_count, freq_bin_count, slice_size, layer_count, layer_neuron_count, class_count):
    print("*"*40)
    print("Building basic model")
    parameter_count = 0
    input_placeholder = tf.placeholder(tf.float32,
                                        shape=(None, channel_count * freq_bin_count * slice_size),
                                        name="input_Placeholder")
    label_placeholder = tf.placeholder(tf.int32,
                                        shape=(None),
                                        name="label_placeholder")
    is_training_placeholder = tf.placeholder_with_default(False,
                                        shape=(),
                                        name="training_placeholder")

    current_layer = input_placeholder
    current_layer = tf.layers.batch_normalization(current_layer, training=is_training_placeholder)

    print("input layer:", current_layer)

    # make [layer_count] layers
    for layer in range(layer_count):
        # fully connected -> BN -> Relu -> Dropout
        layer_parameter_count = tf_util.get_shape_list(current_layer)[-1] * layer_neuron_count
        parameter_count += layer_parameter_count
        current_layer = tf.layers.dense(current_layer,
                                        units=layer_neuron_count,
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.005),
                                        kernel_constraint=tf.keras.constraints.MaxNorm())
        current_layer = tf.layers.batch_normalization(current_layer, training=is_training_placeholder)
        current_layer = tf.nn.leaky_relu(current_layer)
        current_layer = tf.layers.dropout(current_layer, 0.3, training=is_training_placeholder)
        print("hidden layer ({} parameters): {}".format(layer_parameter_count, current_layer))

    # Logits Layer
    layer_parameter_count = tf_util.get_shape_list(current_layer)[-1] * class_count
    parameter_count += layer_parameter_count
    logits = tf.layers.dense(inputs=current_layer, units=class_count)

    # labels_reshaped = tf.reshape(label_placeholder, [-1, 1])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.one_hot(label_placeholder, class_count),
        logits=logits))

    prediction = tf.nn.softmax(logits, name="prediction")

    print("output layer ({} parameters): {}".format(layer_parameter_count, prediction))

    optimizer = tf.train.AdamOptimizer()
    # upgrade when moving to next tf version
    # optimizer = tf.contrib.training.AdamWOptimizer()

    # always do this to make batch norm work
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

    accuracy = tf_util.get_accuracy_op(prediction, label_placeholder)

    # tensorboard summaries
    tf.summary.histogram("input", input_placeholder)
    tf.summary.histogram("raw_logits", logits)
    tf.summary.histogram("logits", prediction)
    tf.summary.histogram("labels", tf.one_hot(label_placeholder, class_count))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)

    print("*"*20)
    print("Model has a total of {} parameters. Ideally the training side will be larger than the parameters".format(parameter_count))
    print("*"*40)

    return input_placeholder, label_placeholder, is_training_placeholder, prediction, accuracy, loss, train_op


