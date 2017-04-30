# !/usr/bin/python
# -*- encoding = utf-8 -*-
'''
preprocessing: build meta list for XRay image classification
'''
import argparse
import os
import random
import csv
import numpy as np
import PIL
from PIL import Image


class MetaBuilder(object):
    '''Build meta file for XRay image classification'''

    def __init__(self, path_train, path_test, path_log, path, label_dict, compute_mean, k, batch_size):
        # path to log
        self.path_log = path_log.strip()
        # path to meta list
        self.path_train = path_train.strip()
        self.path_test = path_test.strip()
        # path to root
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path_log):
            os.mknod(self.path_log)
        # whether to compute mean values
        self.compute_mean = compute_mean
        # number of folds for cross validation
        self.k = k
        self.batch_size = batch_size
        self.meta = []
        # label definition
        self.label = label_dict
        self.fnames = {}
        self.conflict = []

    def get_label(self, dirpath):
        '''get label of a folder'''
        key = dirpath.split('/')[0]
        if key in self.label:
            return self.label[key]
        else:
            return -1

    def gen(self):
        '''generate meta data with normal labels'''
        with open(self.path_log, 'w') as log:
            num_pos = 0
            num_neg = 0
            print 'generate meta list from %s' % self.path
            for (dirpath, dirnames, filenames) in os.walk(self.path):
                if len(filenames) > 1 and 'jpg' in filenames[0]:
                    log.write('There are %d images under %s\n' % (
                        len(filenames), dirpath.replace(self.path, '').replace('/', '', 1)))
                    continue
                if len(filenames) != 1:
                    continue
                if 'jpg' not in filenames[0]:
                    continue
                path = str(os.path.join(dirpath, filenames[0])).replace(
                    self.path, '').replace('/', '', 1)
                label = self.get_label(path)
                if label == -1:
                    continue
                if filenames[0] not in self.fnames:
                    self.fnames[filenames[0]] = label
                else:
                    if self.fnames[filenames[0]] != label:
                        self.conflict.append(filenames[0])
                        log.write('Conflict sample: %s, %d\n' % (path, label))
                    else:
                        log.write('Duplicate sample: %s, %d\n' % (path, label))
                    continue

                self.meta.append([path, label])

            for item in self.conflict:
                for i, sample in enumerate(self.meta):
                    if item in sample[0]:
                        log.write('Conflict sample: %s, %d\n' %
                                  (sample[0], sample[1]))
                        del self.meta[i]

            for sample in self.meta:
                if sample[1] == 1:
                    num_neg += 1
                elif sample[1] == 0:
                    num_pos += 1
                else:
                    log.write('%s has no label\n' % path)

                '''# dirpath: root, dirnames: normal/abnormal
                print (dirpath, dirnames, filenames)
                for folder in dirnames:
                    if folder not in self.label:
                        log.write('Subfolder has no labels: %s\n' % folder)
                        continue
                    for (_, img_folders, _) in os.walk(os.path.join(dirpath, folder)):
                        # img_folders: all folders under normal
                        for ifd in img_folders:
                            images = os.listdir(
                                os.path.join(dirpath, folder, ifd))
                            if len(images) != 1:
                                log.write('There are %d images under %s\n' % (
                                    len(images), os.path.join(folder, ifd)))
                            else:
                                #meta.write('%s, %d\n' % (os.path.join(
                                #    folder, ifd, images[0]), self.label[folder]))
                                self.meta.append([os.path.join(folder, ifd, images[0]), self.label[folder]])
                                if self.label[folder] == 1:
                                    num_neg += 1
                                else:
                                    num_pos += 1
                '''
            print 'number of positive samples (normal): %d' % num_pos
            print 'number of negative samples (abnormal): %d' % num_neg
            log.write('number of positive samples (normal): %d\n' % num_pos)
            log.write('number of negative samples (abnormal): %d\n' % num_neg)
            self.save_(log)

    def save_k_fold(self, meta):
        '''save k fold train and test list'''
        meta_fold = []
        slice = int(len(meta) / self.k)
        # split to k slice
        for i in range(self.k):
            if i == self.k - 1:
                meta_fold.append(meta[i * slice:])
            else:
                meta_fold.append(meta[i * slice: (i + 1) * slice])

        # save meta list
        for i in range(self.k):
            train = self.path_train + str(i) + '.csv'
            test = self.path_test + str(i) + '.csv'
            with open(train, 'w') as f1, open(test, 'w') as f2:
                wtrain = csv.writer(f1)
                wtest = csv.writer(f2)
                len_test = int(len(meta_fold[i]) / self.batch_size) * self.batch_size
                for item in meta_fold[i][:len_test]:
                    wtest.writerow(item)
                for j in range(self.k):
                    if j == i:
                        for item in meta_fold[j][len_test:]:
                            wtrain.writerow(item)
                    else:
                        for item in meta_fold[j]:
                            wtrain.writerow(item)

    def save_(self, log):
        '''save train and test list'''
        with open(self.path_train, 'w') as train, open(self.path_test, 'w') as test:
            wtrain = csv.writer(train, delimiter=' ')
            wtest = csv.writer(test, delimiter=' ')
            # currently, we use 3 times augmentation for neg samples
            meta_over = []
            for i, item in enumerate(self.meta):
                if item[-1] == 1:
                    meta_over.extend([item, item, item])
            meta = self.meta + meta_over
            random.shuffle(meta)
            self.save_k_fold(meta)
            if self.compute_mean is True:
                self.mean_(log, meta)

    def mean_(self, log, meta):
        img_list = []
        for item in meta:
            img_path = os.path.join(self.path, item[0])
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize((224, 224), PIL.Image.ANTIALIAS)
            img_list.append(np.asarray(img))
        img_array = np.asarray(img_list)
        print img_array.shape
        mean_pixel = np.mean(img_array, axis=(0, 1, 2))
        print 'mean of pixel grayscale: ', mean_pixel
        log.write('mean of pixel grayscale: (%.4f, %.4f, %.4f)\n' %
                  tuple(mean_pixel.tolist()))


def gen_meta(args):
    '''generate meta list for XRay folder'''
    result_dir = args.res
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    path_train = 'train'
    path_test = 'test'
    path_log = 'log.csv'
    label_dict = {'abnormal_nodule': 1, 'normal': 0,
                  'abnormal_nodule_0228': 1, 'normal_0228': 0}
    path = '../data/xray'
    meta_gen = MetaBuilder(os.path.join(result_dir, path_train), os.path.join(result_dir, path_test), os.path.join(
        result_dir, path_log), path, label_dict, args.mean, args.k, args.bs)
    meta_gen.gen()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess images and generate meta list')

    parser.add_argument('-img', default='../data/xray',
                        help='input image folder')
    parser.add_argument('-res', default='metalist', help='output dir')
    parser.add_argument('-mean', action='store_true',
                        help='whether to compute mean values of the images, default is false')
    parser.add_argument('-k', default=10, type=int,
                        help='number of folds for cross validation')
    parser.add_argument('-bs', default=80, type=int,
                        help='batch size of data feed to singa')
    ARGS = parser.parse_args()
    gen_meta(ARGS)
