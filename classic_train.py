import tensorflow as tf
import random

import os
import json

import eeg_data_util
import model
import tf_util
from tf_util import copyRecursive, get_batch

data_train_path = "TrainingData"
data_test_path = "TestData"
TENSORBOARD_LOGDIR = "FreesideEegModelDir"
MAX_TRAIN_STEP = 3000
BATCH_SIZE = 80

# delete the model directory to restart training
# otherwise it will continue training
continue_training = os.path.isdir(TENSORBOARD_LOGDIR)

# get dataset
train_data = list(eeg_data_util.slice_folder(data_train_path))
test_data = list(eeg_data_util.slice_folder(data_test_path))

print("Training Data size:", len(train_data))
print("Test Data size:    ", len(test_data))
training_sample = random.choice(train_data)
test_sample = random.choice(test_data)
print("Training Sample", "label:", training_sample[1], "record len:", len(training_sample[0]), "data:", training_sample[0])
print("Test Sample    ",  "label:", test_sample[1], "record len:", len(test_sample[0]), "data:", test_sample[0])

# create model
(input_placeholder, label_placeholder, training_placeholder,
 logits_tensor, accuracy_tensor, loss_tensor, train_op
 ) = model.basic_model(
    eeg_data_util.CHANNELS,
    eeg_data_util.FREQ_BINS,
    eeg_data_util.SLICE_SIZE,
    layer_count=4,
    layer_neuron_count=50,
    class_count=3)

# summaries
summary_tensor = tf.summary.merge_all()

saver = tf.train.Saver()
step_count = 0

## Make tensorflow session
with tf.Session() as sess:
    # one set of charts for training, one for test
    training_summary_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR + "/training", sess.graph)
    test_summary_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR + "/test", sess.graph)

    if continue_training:
        # restore last saved model
        saver.restore(sess, TENSORBOARD_LOGDIR + "/model.ckpt")
        with open(TENSORBOARD_LOGDIR + "/step.json", "r") as f:
            step_count = json.load(f)
        print("Continuing training at step {}".format(step_count ))
    else:
        ## Initialize variables
        sess.run(tf.global_variables_initializer())
        print("Starting new training ")

    while step_count < MAX_TRAIN_STEP:
        step_count += 1

        with open(TENSORBOARD_LOGDIR + "/step.json", "w") as f:
            json.dump(step_count, f)

        if step_count % 10 == 0:
            print("step ", step_count)

        batch_training_data, batch_training_labels = get_batch(train_data, BATCH_SIZE)

        # train network
        training_accuracy, training_logits, training_loss, summary, _ = sess.run(
            [accuracy_tensor, logits_tensor, loss_tensor, summary_tensor, train_op],
            feed_dict={input_placeholder: batch_training_data,
                       label_placeholder: batch_training_labels,
                       training_placeholder: True})

        # write data to tensorboard
        training_summary_writer.add_summary(summary, step_count)

        # every x steps check accuracy
        if step_count % 50 == 0:

            batch_test_data, batch_test_labels = get_batch(test_data, min(BATCH_SIZE, len(test_data)))
            test_accuracy, test_logits, test_loss, summary = sess.run(
                [accuracy_tensor, logits_tensor, loss_tensor, summary_tensor],
                feed_dict={input_placeholder: batch_test_data,
                           label_placeholder: batch_test_labels,
                           training_placeholder: False})

            # write data to tensorboard
            test_summary_writer.add_summary(summary, step_count)

            print("Step Count:{}".format(step_count))
            print("Training accuracy: {} loss: {}".format(training_accuracy, training_loss))
            print("Test accuracy: {} loss: {}".format(test_accuracy, test_loss))

            for i, (prediction, data, label) in enumerate(zip(training_logits, batch_training_data, batch_training_labels)):
                if i > 5:
                    break
                is_correct = "*" if tf_util.is_correct(prediction, label) else " "
                print("training:{}p{} l{:.0f} d{}".format(is_correct, prediction, label, data[:3]))

            for i, (prediction, data, label) in enumerate(zip(test_logits, batch_test_data, batch_test_labels)):
                if i > 5:
                    break
                is_correct = "*" if tf_util.is_correct(prediction, label) else " "
                print("test:    {}p{} l{:.0f} d{}".format(is_correct, prediction, label, data[:3]))

            save_path = saver.save(sess, TENSORBOARD_LOGDIR + "/model.ckpt")

            # if step_count % 10000 == 0:
            #     print("copying model")
            #
            #     copyRecursive(TENSORBOARD_LOGDIR, "{}Copy_{}".format(TENSORBOARD_LOGDIR, step_count))

    print("Finished training at step {}.".format(step_count))