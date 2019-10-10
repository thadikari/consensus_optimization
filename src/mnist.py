from tensorflow import keras
import numpy as np
import os


def get_mnist():
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

    (x_train_, y_train), (x_test_, y_test) = keras.datasets.mnist.load_data('MNIST-data')
    x_train = np.reshape(x_train_, (-1, 784)) / 255.0
    # x_test = np.reshape(x_test_, (-1, 784)) / 255.0
    assert len(x_train) == len(y_train)
    # assert len(x_test) == len(y_test)
    return x_train, y_train


def to_1hot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b


def permute(x_, y_, seed=None):
    p = np.random.RandomState(seed=seed).permutation(len(x_))
    return x_[p], y_[p]


def get_data(is_identical):
    x_train, y_train = permute(*get_mnist())
    y_train1h = to_1hot(y_train)

    def gen():
        if is_identical:
            for duo in zip(np.split(x_train, 10), np.split(y_train1h, 10)):
                yield duo
        else:
            indss = [y_train==cls for cls in range(10)]
            #count = min(inds.sum() for inds in indss)
            for inds in indss:
                #yield x_train[inds][:count], y_train1h[inds][:count]
                yield x_train[inds], y_train1h[inds]

    return gen(), (x_train, y_train1h)


def main():
    identical = 0
    assert(60000==sum(len(y_) for x_, y_ in get_data(identical)))
    for x_, y_ in get_data(identical):
        assert(len(x_)==len(y_))
        print(x_.shape, y_.shape)


if __name__ == '__main__': main()
