import numpy as np
import os
import time
#import argparse
import sys
import csv
import logging
import datetime

from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2
from singa import data
from singa import image_tool
from sklearn import metrics
from model import resnet
import conf
import RemoteException


@RemoteException.showError
def train(cnf, dev, ssfolder, net, mean, dl_train, dl_test):
    print 'Start intialization............'
    net.to_device(dev)
    opt = optimizer.SGD(momentum=0.9, weight_decay=cnf.decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)

    dl_train.start()
    dl_test.start()
    lr = cnf.lr
    num_train = dl_train.num_samples
    num_train_batch = num_train / cnf.batch_size
    num_test = dl_test.num_samples
    num_test_batch = num_test / cnf.batch_size
    # if use data loader in singa, there should be no remainder.
    # i.e., num_test % batch_size == 0
    # remainder = num_test % cnf.batch_size

    best_acc = 0.0
    best_loss = 0.0
    nb_epoch_for_best_acc = 0
    tx = tensor.Tensor((cnf.batch_size,) + cnf.input_shape, dev)
    ty = tensor.Tensor((cnf.batch_size,), dev, core_pb2.kInt)
    for epoch in range(cnf.num_epoch):
        loss, acc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            t1 = time.time()
            x, y = dl_train.next()
            # print x.shape
            x -= mean[np.newaxis, :, np.newaxis, np.newaxis]
            t2 = time.time()
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            # print 'copy tx ty ok'
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                opt.apply_with_lr(epoch, lr, g, p, str(s), b)
            t3 = time.time()
            # update progress bar
            info = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
                + ', batch %d: training loss = %f, accuracy = %f, load_time = %.4f, training_time = %.4f' % (
                    b, l, a, t2 - t1, t3 - t2)
            print info
            #utils.update_progress(b * 1.0 / num_train_batch, info)

        disp = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            + ', epoch %d: training loss = %f, training accuracy = %f, lr = %f' \
            % (epoch, loss / num_train_batch, acc / num_train_batch, lr)
        logging.info(disp)
        print disp

        if epoch % 5 == 0 and epoch > 0:
            try:
                net.save(os.path.join(ssfolder, 'model-%d' %
                                      epoch), buffer_size=200)
            except Exception as e:
                print e
                net.save(os.path.join(ssfolder, 'model-%d' %
                                      epoch), buffer_size=300)
            sinfo = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
                + ', epoch %d: save model in %s' % (
                    epoch, os.path.join(ssfolder, 'model-%d.bin' % epoch))
            logging.info(sinfo)
            print sinfo

        loss, acc = 0.0, 0.0
        #dominator = num_test_batch
        # print 'num_test_batch: ', num_test_batch
        y_truth = []
        y_predict = []
        for b in range(num_test_batch):
            x, y = dl_test.next()
            y_truth.extend(y.tolist())
            x -= mean[np.newaxis, :, np.newaxis, np.newaxis]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            pred = tensor.to_numpy(net.predict(tx))
            pred = np.argsort(-pred)[:, 0]
            y_predict.extend(pred.tolist())
            loss += l * cnf.batch_size
            acc += a * cnf.batch_size
            # print datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            #+ ' batch %d, test loss = %f, test accuracy = %f' % (b, l, a)

        acc /= num_test
        loss /= num_test
        roc = metrics.roc_auc_score(y_truth, y_predict)
        disp = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') \
            + ', epoch %d: test loss = %f, test accuracy = %f, roc = %f' % (epoch, loss, acc, roc)
        logging.info(disp)
        print disp

        if acc > best_acc + 0.005:
            best_acc = acc
            best_loss = loss
            nb_epoch_for_best_acc = 0
            try:
                net.save(os.path.join(ssfolder, 'best_model'), buffer_size=200)
            except Exception as e:
                print e
                net.save(os.path.join(ssfolder, 'best_model'), buffer_size=300)
            sinfo = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') + \
                ', epoch %d: save best model ' % epoch
            logging.info(sinfo)
            print sinfo
        else:
            nb_epoch_for_best_acc += 1
            if nb_epoch_for_best_acc > 8:
                break
            elif nb_epoch_for_best_acc % 4 == 0:
                lr /= 10
                logging.info("Decay the learning rate from %f to %f" %
                             (lr * 10, lr))

    dl_train.end()
    dl_test.end()
    return (best_acc, best_loss)


def predict(cnf, dev, net, mean, dl_test, ssfolder, topk=5):
    '''Predict the label of each image.
    Args:
        net, a pretrained neural net without params
        dev, the training device
        topk, return the topk labels for each image.
    '''

    print 'Start prediction............'
    net.load(ssfolder, 200)
    net.to_device(dev)

    dl_test.start()
    num_test = dl_test.num_samples
    num_test_batch = num_test / cnf.batch_size
    remainder = num_test % cnf.batch_size

    tx = tensor.Tensor((cnf.batch_size,) + cnf.input_shape, dev)
    ty = tensor.Tensor((cnf.batch_size,), dev, core_pb2.kInt)
    ground_truth = []
    predict = []
    print 'num_test_batch: ', num_test_batch
    for b in range(num_test_batch):
        # print 'batch ', b
        x, y = dl_test.next()
        ground_truth.extend(y.tolist())
        x -= mean[np.newaxis, :, np.newaxis, np.newaxis]
        tx.copy_from_numpy(x)
        ty = net.predict(tx)
        ty.to_host()
        prob = tensor.to_numpy(ty)
        labels = np.fliplr(np.argsort(prob))  # sort prob in descending order
        # print labels[:, :topk]
        predict.extend(labels[:, :topk].reshape(-1).tolist())

    dl_test.end()
    return predict, ground_truth


def save_pred_errors(pred, truth, test_file, res_file):
    with open(test_file, 'rb') as tfile, open(res_file, 'w') as rfile:
        treader = csv.reader(tfile)
        rwriter = csv.writer(rfile)
        i = 0
        for line in treader:
            if pred[i] != truth[i]:
                rwriter.writerow([line[0], line[1], pred[i]])
            i += 1
    logging.info(datetime.datetime.now().strftime(
        '%b-%d-%y %H:%M:%S') + 'save predicted errors in %s\n' % res_file)
    print datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') + 'save predicted errors in %s\n' % res_file


train_tool = image_tool.ImageTool()
validate_tool = image_tool.ImageTool()


def train_transform(path):
    global train_tool
    return train_tool.load(path).resize_by_list([cnf.crop_size]).rotate_by_range((-5, 5)).enhance(0.1).get()


def validate_transform(path):
    global validate_tool
    return validate_tool.load(path).resize_by_list([cnf.crop_size]).get()


def hp_tune(cnf, dev):
    '''hyperparameter tuning'''
    log_dir = os.path.join(cnf.log_dir, cnf.mode +
                           datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(
        log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)

    best_acc = 0.0
    best_loss = 0
    best_idx = -1
    for i in range(20):
        ssfolder = os.path.join(cnf.snapshot_folder, cnf.mode + str(i))
        if not os.path.isdir(ssfolder):
            os.makedirs(ssfolder)
        cnf.gen_conf()
        with open(os.path.join(log_dir, '%d.conf' % i), 'w') as fconf:
            cnf.dump(fconf)

        dl_train = data.ImageBatchIter(cnf.train_file + 'train0.csv', cnf.batch_size, train_transform,
                                       shuffle=True, delimiter=',', image_folder=cnf.input_folder, capacity=10)
        dl_test = data.ImageBatchIter(cnf.test_file + 'test0.csv', cnf.batch_size, validate_transform,
                                      shuffle=False, delimiter=',', image_folder=cnf.input_folder, capacity=10)
        try:
            if cnf.net == 'resnet':
                net = resnet.create_net(cnf.net, cnf.depth, cnf.use_cpu)
            else:
                raise Exception('Unsupported net: ', cnf.net)
            logging.info('The %d-th trial' % i)
            #mean = 122.4551
            mean = np.asarray([124.76510401, 124.76510401, 124.76510401])
            # train
            acc, loss = train(cnf, dev, ssfolder, net, mean, dl_train, dl_test)
            # acc,loss= train(cnf.lr, ssfolder, cnf.train_file, cnf.test_file, cnf.input_folder, net, mean,
            # cnf.num_epoch, vgg_lr, cnf.decay, cnf.input_shape,
            # cnf.batch_size, cnf.use_cpu)
            logging.info(
                'The best test accuracy for %d-th trial is %f, with loss=%f' % (i, acc, loss))
            if best_acc < acc:
                best_acc = acc
                best_loss = loss
                best_idx = i
            logging.info('The best test accuracy so far is %f, with loss=%f, for the %d-th conf'
                         % (best_acc, best_loss, best_idx))
            # test
            pred, truth = predict(net, )
        except Exception as e:
            print "except", e


def cross_validate(cnf, dev):
    '''cross validation
    suppose best hyperparameters have been chosen and set in conf.py'''
    log_dir = os.path.join(cnf.log_dir, cnf.mode +
                           datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(
        log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)

    for i in range(cnf.k_fold):
        ssfolder = os.path.join(cnf.snapshot_folder, cnf.mode + str(i))
        if not os.path.isdir(ssfolder):
            os.makedirs(ssfolder)
        train_file = cnf.train_file + 'train%d.csv' % i
        test_file = cnf.test_file + 'test%d.csv' % i
        try:
            if cnf.net == 'resnet':
                net = resnet.create_net(cnf.net, cnf.depth, cnf.use_cpu)
            else:
                raise Exception('Unsupported net: ', cnf.net)
            logging.info('The %d-th fold' % i)
            dl_train = data.ImageBatchIter(train_file, cnf.batch_size, train_transform,
                                           shuffle=True, delimiter=',', image_folder=cnf.input_folder, capacity=10)
            dl_test = data.ImageBatchIter(test_file, cnf.batch_size, validate_transform,
                                          shuffle=False, delimiter=',', image_folder=cnf.input_folder, capacity=10)
            mean = np.asarray([124.76510401, 124.76510401, 124.76510401])
            # train
            acc, loss = train(cnf, dev, ssfolder, net, mean, dl_train, dl_test)
            # acc,loss= train(cnf.lr, ssfolder, cnf.train_file, cnf.test_file, cnf.input_folder, net, mean,
            # cnf.num_epoch, vgg_lr, cnf.decay, cnf.input_shape,
            # cnf.batch_size, cnf.use_cpu)
            logging.info(
                'The best test accuracy for %d-th fold is %f, with loss=%f' % (i, acc, loss))
            # test
            best_model = os.path.join(ssfolder, 'best_model')
            result_file = os.path.join(log_dir, 'results_%d.csv' % i)
            dl_test = data.ImageBatchIter(test_file, cnf.batch_size, validate_transform,
                                          shuffle=False, delimiter=',', image_folder=cnf.input_folder, capacity=10)
            pred, truth = predict(cnf, dev, net, mean,
                                  dl_test, best_model, topk=1)
            save_pred_errors(pred, truth, test_file, result_file)
        except Exception as e:
            print "except", e


if __name__ == '__main__':
    cnf = conf.Conf()
    dev = None
    if cnf.use_cpu:
        print 'Using CPU'
        dev = device.get_default_device()
    else:
        print 'Using GPU'
        #dev = device.create_cuda_gpu()
        dev = device.create_cuda_gpu_on(1)

    if not os.path.isdir(cnf.snapshot_folder):
        os.makedirs(cnf.snapshot_folder)
    if cnf.mode == 'tune':
        hp_tune(cnf, dev)
    elif cnf.mode == 'cv':
        cross_validate(cnf, dev)
    else:
        raise Exception('Undefined mode: %s' % cnf.mode)
