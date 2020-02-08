from tensorflow import keras
import tensorflow as tf
import numpy as np
import os


def get_dataset(name):
    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    (x_train, y_train), (x_test, y_test) = getattr(keras.datasets, name).load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0
    return (x_train, y_train), (x_test, y_test)

def permute(x_, y_, seed=None):
    p = np.random.RandomState(seed=seed).permutation(len(x_))
    return x_[p], y_[p]

def input_generator(x_train, y_train, batch_size):
    while True:
        x_train, y_train = permute(x_train, y_train)
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size


def compute_metrics(logits, target, num_classes):
    target_1h = tf.one_hot(tf.cast(target, tf.int32), num_classes)
    losses = tf.losses.softmax_cross_entropy(target_1h, logits, reduction='none')
    sum_loss = tf.reduce_sum(losses)
    avg_loss = tf.reduce_mean(losses)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.cast(target, 'int64'))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    return accuracy, sum_loss, avg_loss
