import numpy as np
import collections
from tensorflow.python.framework import random_seed
from tensorflow.keras.datasets import cifar100,fashion_mnist,cifar10
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import dtypes
from tensorflow.keras.datasets.fashion_mnist import load_data

Datasets = collections.namedtuple('Datasets', ['train', 'test'])
data_loder_dic = {'cifar100':cifar100,
                  'cifar10':cifar10,
                  'fashion_mnist':fashion_mnist}
class DataSet(object):
  """Container class for a dataset .
  """
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
        raise TypeError(
            'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in range(batch_size)], [
          fake_label for _ in range(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate(
          (images_rest_part, images_new_part), axis=0), np.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

class DATA_LODER:
    def __init__(self,batch_size,loder):
        (train_images,train_labels),(test_images,test_labels) = data_loder_dic[loder].load_data()
        train = DataSet(train_images,train_labels,reshape=False)
        test = DataSet(test_images,test_labels,reshape=False)
        self.data = Datasets(train=train,test=test)
        self.train_data_num = self.data.train.num_examples
        self.test_data_num = self.data.test.num_examples
        self.batch_size = batch_size
        print("data init done!")

    def next_train_batch(self):
        data_batch = self.data.train.next_batch(self.batch_size)
        data_ = data_batch[0][:,:,:,np.newaxis] if len(data_batch[0].shape) == 3 else data_batch[0]
        label_ = data_batch[1].flatten()
        return data_, label_

    def next_test_batch(self):
        test_source = self.data.test.next_batch(self.batch_size)
        test_data = test_source[0][:, :, :, np.newaxis] if len(test_source[0].shape) == 3 else test_source[0]
        test_label = test_source[1].flatten()
        return test_data,test_label

    def test_data(self):
        test_source = self.data.test
        test_data = test_source.images[:, :, :, np.newaxis] if len(test_source.images.shape) == 3 else test_source.images
        test_label = test_source.labels.flatten()
        return test_data,test_label

class DATA_FASHION:
    def __init__(self,batch_size):
        self.data = input_data.read_data_sets('/home/jimxiang/YangCheng/model_tensorflow/datasets/fashion',one_hot=True)
        self.train_data_num = self.data.train.num_examples
        self.test_data_num = self.data.test.num_examples
        self.batch_size = batch_size
        print("data init done!")

    def next_train_batch(self):
        data_batch = self.data.train.next_batch(self.batch_size)
        data_ = data_batch[0]
        data_ = np.resize(data_, (self.batch_size, 28, 28, 1))
        label_ = np.argmax(data_batch[1],axis=1)
        return data_, label_

    def next_validation_batch(self):
        val_source = self.data.validation.next_batch(self.batch_size)
        val_data = val_source[0]
        val_data = np.resize(val_data,(self.batch_size, 28, 28, 1))
        val_label = np.argmax(val_source[1],axis=1)
        return val_data,val_label

    def next_test_batch(self):
        test_source = self.data.test.next_batch(self.batch_size)
        test_data = test_source[0]
        test_data = np.resize(test_data,(self.batch_size, 28, 28, 1))
        test_label = np.argmax(test_source[1],axis=1)
        return test_data,test_label

    def test_data(self):
        test_source = self.data.test
        test_data = test_source.images
        test_data = np.resize(test_data,(test_source.num_examples, 28, 28, 1))
        test_label = np.argmax(test_source.labels,axis=1)
        return test_data,test_label

def cifar_100_loder(batch_size):
    return DATA_LODER(batch_size,'cifar100')

def cifar_10_loder(batch_size):
    return DATA_LODER(batch_size,'cifar10')

def fashion_mnist_loder(batch_size):
    return DATA_LODER(batch_size,'fashion_mnist')


if __name__=='__main__':
    data = cifar_100_loder(32)
#data = DATA_FASHION(32).next_train_batch()
