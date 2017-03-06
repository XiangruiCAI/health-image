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

#from data_loader import data as dt
from singa import data
from singa import image_tool
from model import resnet
import conf


def vgg_lr(epoch):
    return 0.05 / float(1 << (epoch / 25))


def train(cnf, ssfolder, net, mean, dl_train, dl_test):
    print 'Start intialization............'
    if cnf.use_cpu:
        print 'Using CPU'
        dev = device.get_default_device()
    else:
        print 'Using GPU'
        dev = device.create_cuda_gpu()

    net.to_device(dev)
    opt = optimizer.SGD(momentum=0.9, weight_decay=cnf.decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)

    dl_train.start()
    dl_test.start()
    num_train = dl_train.num_samples
    num_train_batch = num_train / cnf.batch_size
    num_test = dl_test.num_samples
    num_test_batch = num_test / cnf.batch_size
    remainder = num_test % cnf.batch_size

    best_acc = 0.0
    best_loss = 0.0
    nb_epoch_for_best_acc = 0
    tx = tensor.Tensor((cnf.batch_size,) + cnf.input_shape, dev)
    ty = tensor.Tensor((cnf.batch_size,), dev, core_pb2.kInt)
    for epoch in range(self.num_epoch):
        loss, acc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            t1 = time.time()
            x, y = dl_train.next()
            print x.shape()
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
                opt.apply_with_lr(epoch, lr, g, p, str(s), b)
            t3 = time.time()
            # update progress bar
            info = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            + ', batch %d: training loss = %f, accuracy = %f, load_time = %.4f, training_time = %.4f' % (b, l, a, t2-t1, t3-t2)
            print info
            #utils.update_progress(b * 1.0 / num_train_batch, info)

        disp = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
        + ', epoch %d: training loss = %f, training accuracy = %f, lr = %f' \
            % (epoch, loss / num_train_batch, acc / num_train_batch, lr)
        logging.info(disp)
        print disp

        if epoch % 50 == 0 and epoch > 0:
            try:
                net.save(os.path.join(ssfolder, 'model-%d' % epoch), buffer_size=200)
            except Exception as e:
                print e
                net.save(os.path.join(ssfolder, 'model-%d' % epoch), buffer_size=300)
            sinfo = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            + ', epoch %d: save model in %s' % (epoch, os.path.join(ssfolder, 'model-%d.bin' % epoch))
            logging.info(sinfo)
            print sinfo

        loss, acc = 0.0, 0.0
        #dominator = num_test_batch
        #print 'num_test_batch: ', num_test_batch
        for b in range(num_test_batch):
            x, y = dl_test.next()
            x -= mean
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l * cnf.batch_size
            acc += a * cnf.batch_size
            #print datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            #+ ' batch %d, test loss = %f, test accuracy = %f' % (b, l, a)

        if remainder > 0:
            #print 'remainder: ', remainder
            x, y = dl_test.next()
            x -= mean
            tx_rmd = tensor.Tensor((remainder,) + cnf.input_shape, dev)
            ty_rmd = tensor.Tensor((remainder,), dev, core_pb2.kInt)
            tx_rmd.copy_from_numpy(x[0:remainder,:,:])
            ty_rmd.copy_from_numpy(y[0:remainder,])
            l, a = net.evaluate(tx_rmd, ty_rmd)
            loss += l * remainder
            acc += a * remainder
            #dominator += 1
            #print datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            #+ ' test loss = %f, test accuracy = %f' % (l, a)
	acc /= num_test
        loss /= num_test
        disp = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
        + ', epoch %d: test loss = %f, test accuracy = %f' % (epoch, loss, acc)
        logging.info(disp)
        print disp

        if acc > best_acc + 0.005:
            best_acc = acc
            best_loss = loss
            nb_epoch_for_best_acc = 0
        else:
            nb_epoch_for_best_acc += 1
            if nb_epoch_for_best_acc > 8:
                break
            elif nb_epoch_for_best_acc % 4 ==0:
                lr /= 10
                logging.info("Decay the learning rate from %f to %f" %(lr*10, lr))

    try:
        net.save(str(os.path.join(ssfolder, 'model')), buffer_size=200)
    except Exception as e:
        net.save(str(os.path.join(ssfolder, 'model')), buffer_size=300)
    sinfo = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
    + ', save final model in %s' % os.path.join(ssfolder, 'model.bin')
    logging.info(sinfo)
    print sinfo

    dl_train.end()
    dl_test.end()
    return (best_acc, best_loss)


if __name__ == '__main__':
    cnf = conf.Conf()
    log_dir = os.path.join(cnf.log_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)
    train_tool = image_tool.ImageTool()
    validate_tool = image_tool.ImageTool()

    best_acc = 0.0
    best_loss = 0
    best_idx = -1
    for i in range(1):
        ssfolder = cnf.snapshot_folder + str(i)
        if not os.path.isdir(ssfolder):
            os.makedirs(ssfolder)
        cnf.gen_conf()
        with open(os.path.join(log_dir, '%d.conf' % i), 'w') as fconf:
            cnf.dump(fconf)

        print 'crop_size: ', cnf.crop_size
        def train_transform(path):
            global train_tool
            return train_tool.load(path, True).rotate_by_range((-5, 5)).random_crop((cnf.crop_size, cnf.crop_size)).enhance(0.1).get()
        def validate_transform(path):
            global validate_tool
            return validate_tool.load(path, True).crop5((cnf.crop_size, conf.crop_size), 5).get()

        dl_train = data.ImageBatchIter(cnf.train_file, cnf.batch_size, train_transform, shuffle=True, delimiter=' ', image_folder=cnf.input_folder, capacity=50)
        dl_test = data.ImageBatchIter(cnf.test_file, cnf.batch_size, validate_transform, shuffle=False, delimiter=' ', image_folder=cnf.input_folder, capacity=50)
        try:
            # TODO(xiangrui): Add the best vgg net.
            if cnf.net == 'resnet':
                net = resnet.create_net(cnf.net, cnf.depth, cnf.use_cpu)
            else:
                raise Exception('Unsupported net: ', cnf.net)
            logging.info('The %d-th trial' % i)
            mean = 122.4551
            acc, loss = train(cnf, ssfolder, net, mean, dl_train, dl_test)
            #acc,loss= train(cnf.lr, ssfolder, cnf.train_file, cnf.test_file, cnf.input_folder, net, mean,
            #        cnf.num_epoch, vgg_lr, cnf.decay, cnf.input_shape, cnf.batch_size, cnf.use_cpu)
            logging.info('The best test accuracy for %d-th trial is %f, with loss=%f' % (i, acc, loss))
            if best_acc < acc:
                best_acc = acc
                best_loss = loss
                best_idx = i
            logging.info('The best test accuracy so far is %f, with loss=%f, for the %d-th conf'
                    % (best_acc, best_loss, best_idx))
        except Exception as e:
            print "except", e

