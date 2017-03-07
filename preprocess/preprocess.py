# !/usr/bin/python
# -*- encoding = utf-8 -*-
'''
preprocessing: build meta list for XRay image classification
'''
import os
import random
import csv
import numpy as np
import PIL
from PIL import Image


class MetaBuilder(object):
    '''Build meta file for XRay image classification'''

    def __init__(self, path_train, path_test, path_log, path, label_dict, ratio = 0.8):
        # path to log
        self.path_log = path_log.strip()
        # path to meta list
        self.path_train = path_train.strip()
        self.path_test = path_test.strip()
        # path to root
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path_log):
            os.mknod(self.path_log)
        self.meta = []
        # label definition
        self.label = label_dict
        self.r = ratio
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
                    log.write('There are %d images under %s\n' % (len(filenames), dirpath.replace(self.path, '').replace('/', '', 1)))
                    continue
                if len(filenames) != 1:
                    continue
                if 'jpg' not in filenames[0]:
                    continue
                path = str(os.path.join(dirpath, filenames[0])).replace(self.path, '').replace('/', '', 1)
                label = self.get_label(path)
                if label == -1:
                    continue
                if filenames[0] not in self.fnames:
                    self.fnames[filenames[0]] = label
                else:
                    if self.fnames[filenames[0]] != label:
                        self.conflict.append(filenames[0])
                        log.write('Conflict sample: %s, %d\n' %(path, label))
                    else:
                        log.write('Duplicate sample: %s, %d\n' %(path, label))
                    continue

                self.meta.append([path, label])

            for item in self.conflict:
                for i, sample in enumerate(self.meta):
                    if item in sample[0]:
                        log.write('Conflict sample: %s, %d\n' %(sample[0], sample[1]))
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

    def save_(self, log):
        '''save train and test list'''
        with open(self.path_train, 'w') as train, open(self.path_test, 'w') as test:
            wtrain = csv.writer(train, delimiter = ' ')
            wtest = csv.writer(test, delimiter= ' ')
            meta_over = []
            for i, item in enumerate(self.meta):
                if item[-1] == 1:
                    meta_over.extend([item, item, item])
            meta = self.meta + meta_over
            random.shuffle(meta)
            spoint = int(len(meta) * self.r)
            for i, item in enumerate(meta):
                if i < spoint:
                    wtrain.writerow(item)
                else:
                    wtest.writerow(item)
            print 'number of training samples: %d' % spoint
            print 'number of test samples: %d' % (len(meta) - spoint)
            log.write('number of training samples: %d\n' % spoint)
            log.write('number of test samples: %d\n' % (len(meta) - spoint))
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
        log.write('mean of pixel grayscale: (%.4f, %.4f, %.4f)\n' % tuple(mean_pixel.tolist()))

def gen_meta():
    '''generate meta list for XRay folder'''
    result_dir = 'metalist'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    path_train = 'train.csv'
    path_test = 'test.csv'
    path_log = 'log.csv'
    label_dict = {'abnormal_nodule': 1, 'normal': 0,
                  'abnormal_nodule_0228': 1, 'normal_0228': 0}
    path = '../data/xray'
    meta_gen = MetaBuilder(os.path.join(result_dir, path_train), os.path.join(result_dir, path_test), os.path.join(
        result_dir, path_log), path, label_dict)
    meta_gen.gen()

if __name__ == '__main__':
    gen_meta()
