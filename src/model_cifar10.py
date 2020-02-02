import os
import random
import tarfile
import pickle
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm


import model

reg = model.ModelReg()


# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10.py
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

cifar10_dataset_folder_name = 'cifar-10-batches-py'
data_directory = os.path.join(os.path.expanduser('~'), '.keras', 'cifar10')
tp_ = lambda *arg_: os.path.join(data_directory, *arg_)
log = lambda arg: print(arg)

class DownloadProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def display_stats(batch_id, sample_id):
    features, labels = load_raw_batch('data_batch_%d'%batch_id)
    if not (0 <= sample_id < len(features)):
        log('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    log('\nStats of batch #{}:'.format(batch_id))
    log('# of Samples: {}\n'.format(len(features)))

    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        log('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    log('\nExample of Image {}:'.format(sample_id))
    log('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    log('Image - Shape: {}'.format(sample_image.shape))
    log('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

def normalize(x):
    min_val, max_val = np.min(x), np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))
    for idx, val in enumerate(x): encoded[idx][val] = 1
    return encoded

def load_raw_batch(file_name):
    with open(tp_(cifar10_dataset_folder_name, file_name), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1') # note the encoding type is 'latin1'
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

def load_preproc_batch(file_name):
    file_path = tp_('preprocessed_%s'%file_name)
    if not isfile(file_path):
        log('Data file not available: {}'.format(file_path))
        get_data()
    return pickle.load(open(file_path, mode='rb'))

def do_all_(file_name):
    features, labels = load_raw_batch(file_name)
    features, labels = normalize(features), one_hot_encode(labels)
    pickle.dump((features, labels), open(tp_('preprocessed_%s'%file_name), 'wb'))

def get_data():
    # Download the dataset (if not exist yet)
    tar_path = tp_('cifar-10-python.tar.gz')
    if not isfile(tar_path):
        log('CIFAR-10 tar not available: {}'.format(tar_path))
        if not isdir(data_directory):
            log('Creating data directory: {}'.format(data_directory))
            os.makedirs(data_directory)
        log('Downloading data to: {}'.format(tar_path))
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                tar_path,
                pbar.hook)

    if 1: #not isdir(tp_(cifar10_dataset_folder_name)):
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=tp_(''))
            tar.close()

    # Explore the dataset
    batch_id = 3
    sample_id = 7000
    display_stats(batch_id, sample_id)

    for batch_id in range(1,6): do_all_('data_batch_%d'%batch_id)
    do_all_('test_batch')

def permute(x_, y_, seed=None):
    p = np.random.RandomState(seed=seed).permutation(len(x_))
    return x_[p], y_[p]

def load_all_data():
    n_batches = 5
    all_features = []
    all_labels = []
    for batch_id in range(1, n_batches + 1):
        features, labels = permute(*load_preproc_batch('data_batch_%d'%batch_id))
        all_features.append(features)
        all_labels.append(labels)
    return np.vstack(all_features), np.argmax(np.vstack(all_labels), axis=1)

def to_1hot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b

def process_data():
    x_train, y_train = permute(*load_all_data())
    #x_train = np.random.normal(size=[1000, 32,32,3])
    #y_train = np.random.randint(10, size=1000)
    y_train1h = to_1hot(y_train)
    Q_global = Dist((x_train, y_train1h))
    return x_train, y_train, y_train1h, Q_global


'''
definitions for distributions
'''
from model import DistClassification as Dist
reg_dist = reg.reg_dist.reg


@reg_dist
def distinct_10():
    x_, y_, y1h_, Q_global = process_data()
    indss = [y_==cls for cls in range(10)]
    #count = min(inds.sum() for inds in indss)
    locals = [Dist((x_[inds], y1h_[inds])) for inds in indss]
    return locals, Q_global



'''
definitions for functions
'''
from model import EvalClassification as Eval
from model import params


def conv_net(x, keep_prob, f1,f2,f3,f4, w1,b1, w2,b2, w3,b3, w4,b4):
    # 1, 2
    conv1 = tf.nn.conv2d(x, f1, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv1 = tf.layers.batch_normalization(conv1)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1, f2, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv2 = tf.layers.batch_normalization(conv2)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2, f3, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv3 = tf.layers.batch_normalization(conv3)

    # 7, 8
    conv4 = tf.nn.conv2d(conv3, f4, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv4 = tf.layers.batch_normalization(conv4)

    # 9
    # flat = tf.contrib.layers.flatten(x[:,:,:,:2])
    flat = tf.contrib.layers.flatten(conv4)

    fully_connected = lambda xx,actv,ww,bb: actv(xx@ww+bb)
    # 10
    full1 = fully_connected(flat, tf.nn.relu, w1,b1)
    full1 = tf.nn.dropout(full1, keep_prob)
    # full1 = tf.layers.batch_normalization(full1)

    # 11
    full2 = fully_connected(full1, tf.nn.relu, w2,b2)
    full2 = tf.nn.dropout(full2, keep_prob)
    # full2 = tf.layers.batch_normalization(full2)

    # 12
    full3 = fully_connected(full2, tf.nn.relu, w3,b3)
    full3 = tf.nn.dropout(full3, keep_prob)
    # full3 = tf.layers.batch_normalization(full3)

    # # 13
    # full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    # full4 = tf.nn.dropout(full4, keep_prob)
    # full4 = tf.layers.batch_normalization(full4)

    # return fully_connected(x[:,:,0,0], tf.identity, w4[:32,:],b4)
    return fully_connected(full3, tf.identity, w4,b4)


def reg_func(func):
    lam = lambda: Eval(func, [32,32,3], 10)
    reg.reg_func.put(func.__name__, lam)
    return func

@reg_func
def linear(x_):
    w_, w, b = params([32*32*3,10], 10)
    return w_, tf.contrib.layers.flatten(x_)@w+b

@reg_func
def relu(x_):
    w_, w1, b1, w2, b2 = params([32*32*3,512],512, [512,10],10)
    flat = tf.contrib.layers.flatten(x_)
    return w_, tf.nn.relu(flat@w1+b1)@w2+b2

@reg_func
def conv(x_):
    w_, *weights = params([3,3,3,64], [3,3,64,128],
                          [5,5,128,256], [5,5,256,512],
                          [2048,128], 128, [128,256], 256,
                          [256,512], 512, [512,10], 10)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                                                  # train_dict    # test_dict
    return w_, conv_net(x_, keep_prob, *weights), {keep_prob:.7}, {keep_prob:1.}



if __name__ == '__main__': test_distrb()
