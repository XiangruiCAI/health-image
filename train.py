import numpy as np
import os
import time
#import argparse
import sys
import logging
import datetime

from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2

from data_loader import data as dt
from model import vgg
import conf


def vgg_lr(epoch):
    return 0.05 / float(1 << (epoch / 25))


def train(meta_train, meta_test, data, net, mean, max_epoch, get_lr,
        weight_decay, input_shape, batch_size=100, use_cpu=False):

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

    dl_train = dt.MImageBatchIter(meta_train, batch_size, dt.load_from_img,
            shuffle=True, delimiter=' ', image_folder=data, capacity=10)
    dl_train.start()
    dl_test = dt.MImageBatchIter(meta_test, batch_size, dt.load_from_img,
            shuffle=False, delimiter=' ', image_folder=data, capacity=10)
    dl_test.start()
    num_train = dl_train.num_samples
    num_train_batch = num_train / batch_size
    num_test = dl_test.num_samples
    num_test_batch = num_test / batch_size
    remainder = num_test % batch_size

    best_acc = 0.0
    best_loss = 0.0
    nb_epoch_for_best_acc = 0
    tx = tensor.Tensor((batch_size,) + input_shape, dev)
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
            info = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            + ', batch %d: training loss = %f, accuracy = %f, load_time = %.4f, training_time = %.4f' % (b, l, a, t2-t1, t3-t2)
            print info
            #utils.update_progress(b * 1.0 / num_train_batch, info)

        disp = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
         + ' training loss = %f, training accuracy = %f, lr = %f' \
            % (loss / num_train_batch, acc / num_train_batch, get_lr(epoch))
        logging.info(disp)
        print disp

        if epoch % 50 == 0 and epoch > 0:
            net.save('model-%d.bin' % epoch, buffer_size=100)
            sinfo = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            + ', epoch %d: save model in model-%d.bin' % (epoch, epoch)
            logging.info(sinfo)
            print sinfo

        loss, acc = 0.0, 0.0
        dominator = num_test_batch
        #print 'num_test_batch: ', num_test_batch
        for b in range(num_test_batch):
            x, y = dl_test.next()
            x -= mean
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            acc += a
            #print datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            #+ ' batch %d, test loss = %f, test accuracy = %f' % (b, l, a)

        if remainder > 0:
            #print 'remainder: ', remainder
            x, y = dl_test.next()
            x -= mean
            tx_rmd = tensor.Tensor((remainder,) + input_shape, dev)
            ty_rmd = tensor.Tensor((remainder,), dev, core_pb2.kInt)
            tx_rmd.copy_from_numpy(x[0:remainder,:,:])
            ty_rmd.copy_from_numpy(y[0:remainder,])
            l, a = net.evaluate(tx_rmd, ty_rmd)
            loss += l
            acc += a
            dominator += 1
            #print datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            #+ ' test loss = %f, test accuracy = %f' % (l, a)

	acc /= dominator
        loss /= dominator
        disp = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
        + ' test loss = %f, test accuracy = %f' % (loss, acc)
        logging.info(disp)
        print disp
	if acc > best_acc + 0.005:
            best_acc = acc
            best_loss = loss
            nb_epoch_for_best_acc = epoch

    net.save('model.bin', buffer_size=100)
    sinfo = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') + ', save final model in model.bin'
    logging.info(sinfo)
    print sinfo

    dl_train.end()
    dl_test.end()
    return (best_acc, best_loss, nb_epoch_for_best_acc)


if __name__ == '__main__':
    cnf = conf.Conf()
    log_dir = os.path.join(cnf.log_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)

    best_acc = 0.0
    best_loss = 0
    best_idx = -1
    for i in range(1):
        #cnf.gen_conf()
        with open(os.path.join(log_dir, '%d.conf' % i), 'w') as fconf:
            cnf.dump(fconf)
        try:
            if cnf.net == 'vgg':
                net = vgg.create_net(cnf.input_shape, cnf.use_cpu)
            mean = dt.get_mean(cnf.input_folder)
            acc,loss,epoch = train(cnf.train_file, cnf.test_file, cnf.input_folder, net, mean,
                    cnf.num_epoch, vgg_lr, cnf.decay, cnf.input_shape, cnf.batch_size, cnf.use_cpu)
            logging.info('The best test accuracy for %d-th trial is %f, with loss=%f' % (i, acc, loss))
            if best_acc < acc:
                best_acc = acc
                best_loss = loss
                best_idx = i
            logging.info('The best test accuracy so far is %f, with loss=%f, from the %d-th conf'
                    % (best_acc, best_loss, best_idx))
        except Exception as e:
            print "except", e

    '''
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
    '''
