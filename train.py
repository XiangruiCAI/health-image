import numpy as np
import os
import time
import argparse

from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2

import sys
from data_loader import data as dt
from model import vgg


def vgg_lr(epoch):
    return 0.1 / float(1 << (epoch / 25))


def train(meta_train, meta_test, data, net, mean, max_epoch, get_lr,
        weight_decay, batch_size=100, use_cpu=False):

    print 'Start intialization............'
    if use_cpu:
        print 'Using CPU'
        dev = device.get_default_device()
    else:
        print 'Using GPU'
        dev = device.create_cuda_gpu()

    net.to_device(dev)
    opt = optimizer.SGD(momentum=0.9, weight_decay=weight_decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)

    dl_train = dt.MImageBatchIter(meta_train, batch_size, dt.load_from_np,
            shuffle=True, delimiter=' ', image_folder=data, capacity=200)
    dl_train.start()
    dl_test = dt.MImageBatchIter(meta_test, batch_size, dt.load_from_np,
            shuffle=False, delimiter=' ', image_folder=data, capacity=200)
    dl_test.start()
    num_train = dl_train.num_samples
    num_train_batch = num_train / batch_size
    num_test = dl_train.num_samples
    num_test_batch = num_test / batch_size
    remainder = num_test % batch_size

    tx = tensor.Tensor((batch_size, 1, 2021, 2021), dev)
    ty = tensor.Tensor((batch_size,), dev, core_pb2.kInt)
    for epoch in range(max_epoch):
        loss, acc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            t1 = time.time()
            x, y = dl_train.next()
            #print 'x.norm: ', np.linalg.norm(x)
            x -= mean
            t2 = time.time()
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            #print 'copy tx ty ok'
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s), b)
            t3 = time.time()
            # update progress bar
            info = 'batch %d: training loss = %f, accuracy = %f, load_time = %.4f, training_time = %.4f\n' % (b, l, a, t2-t1, t3-t2)

            utils.update_progress(b * 1.0 / num_train_batch, info)

        disp = '\ntraining loss = %f, training accuracy = %f, lr = %f' \
            % (loss / num_train_batch, acc / num_train_batch, get_lr(epoch))
        print disp

        if (epoch + 1) % 50 == 0:
            net.save('model-%d.bin' % epoch)

        loss, acc = 0.0, 0.0
        dominator = num_test_batch
        for b in range(num_test_batch):
            x, y = dl_test.next()
            x -= mean
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            acc += a
        if remainder > 0:
            x, y = dl_test.next()
            x -= mean
            tx_rmd = tensor.Tensor((remainder, 1, 2021, 2021), dev)
            ty_rmd = tensor.Tensor((remainder,), dev, core_pb2.kInt)
            tx_rmd.copy_from_numpy(x[0:remainder,:,:])
            ty_rmd.copy_from_numpy(y[0:remainder,])
            l, a = net.evaluate(tx_rmd, ty_rmd)
            loss += l
            acc += a
            dominator += 1

        print 'test loss = %f, test accuracy = %f' \
            % (loss / dominator, acc / dominator)
    dl_train.end()
    dl_test.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dcnn for XRay images')
    parser.add_argument('model', choices=['vgg'], default='vgg')
    parser.add_argument('train', default='meta-process/meta_train.csv')
    parser.add_argument('test', default='meta-process/meta_test.csv')
    parser.add_argument('data', default='data/resize2021/')
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()

    if args.model == 'vgg':
        mean = dt.get_mean(args.data, 'npy')
        net = vgg.create_net(args.use_cpu)
        # epoch=150 and batch_size=40
        train(args.train, args.test, args.data, net, mean, 200, vgg_lr, 0.0005,
                16, args.use_cpu)
    else:
        print 'Model not support: ', args.model
