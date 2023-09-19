import numpy
import numpy as np
import torch
import tqdm
import time

DATA_ROOT = './timit_11/'

train = np.load(DATA_ROOT + 'train_11.npy')
train_label = np.load(DATA_ROOT + 'train_label_11.npy')
test = np.load(DATA_ROOT + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))


def show_data(data, label, n=50):
    # data: numpy array of shape (num_samples, num_features)
    # label: numpy array of shape (num_samples,)
    # n: number of samples to show, default is 50
    for i in range(n):
        print("Sample {}: features = {}, label = {}".format(i + 1,
                                                            np.array2string(data[i], precision=4, suppress_small=True),
                                                            label[i]))


def print_sample(sample):
    print(
        "Sample :{}, datatype = {}".format(sample[0], type(sample)))


show_data(train, train_label, 1)
print(train_label[:100])
print(type(train_label))
print(type(train))

tensor_train = torch.from_numpy(train)
print_sample(tensor_train)
tensor_train = tensor_train.float()
print_sample(tensor_train)

print_sample(train_label)
tensor_label = torch.LongTensor(np.array(train_label, dtype=np.int64))
print_sample(tensor_label)

x = torch.from_numpy(numpy.array([True, True, True, False, True]))
#y = np.array(x, dtype=np.bool_)
print(x.numpy().astype(np.int64).sum().item())

# tqdm显示有坑，实际运行没问题就好
for epoch in tqdm.tqdm(range(10),total=10):
    time.sleep(1)
    print(epoch,'\n')
