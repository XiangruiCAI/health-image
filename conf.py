import os
import random


class Conf():
    def __init__(self):
        self.num_epoch = 200
        self.batch_size = 32
        self.use_cpu = False
        self.input_folder="./data/resizejpeg2021"
        self.log_dir = 'log'
        self.train_file="./meta-process/meta_train.csv"
        self.test_file="./meta-process/meta_test.csv"
        self.input_shape = (1, 2021, 2021)

        #self.small_size = 112
        #self.large_size = 192
        #self.crop_size = 96
        self.lr = 0.01
        self.decay = 5e-4
        self.net = 'vgg'


    def dump(self, f):
        f.write('===============================================\n')
        f.write('Configuration for training:\n')
        f.write('===============================================\n')
        f.write('batch size: %d\n' % self.batch_size)
        f.write('max epoch: %d\n' % self.num_epoch)
        f.write('use cpu: %r\n' % self.use_cpu)
        f.write('input folder: %s\n' % self.input_folder)
        f.write('log dir: %s\n' % self.log_dir)
        f.write('train meta list: %s\n' % self.train_file)
        f.write('test meta list: %s\n' % self.test_file)
        f.write('input sample shape: %s\n' % str(self.input_shape))
        f.write('learning rate: %f\n' % self.lr)
        f.write('weight decay: %f\n' % self.decay)
        f.write('===============================================\n')


    def gen_conf(self):
        #small_size = [96, 112, 128, 160, 192]
        #large_size = [112, 128, 160, 192, 224]
        #while True:
        #    conf.small_size = small_size[random.randint(0, len(small_size) - 1)]
        #    conf.large_size = large_size[random.randint(0, len(large_size) - 1)]
        #    if conf.small_size < conf.large_size and 2 * conf.small_size > conf.large_size:
        #        break

        #crop_size = [96, 128, 160, 192]
        #while True:
        #    conf.size = crop_size[random.randint(0, len(crop_size) -1)]
        #    if conf.size <= conf.small_size:
        #        break


        lr = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        self.lr = lr[random.randint(0, len(lr) - 1)]

        decay = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        self.decay = decay[random.randint(0, len(decay) -1)]
