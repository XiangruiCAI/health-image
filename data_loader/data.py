import numpy as np
import os
import random
import time
from multiprocessing import Process, Queue
from singa import data
import dicom


# medical image iter: dicom image
class MImageBatchIter(data.ImageBatchIter):
    def run(self):
        img_list = []
        for line in open(self.img_list_file, 'r'):
            item = line.split(self.delimiter)
            img_path = item[0]
            img_label = int(item[1])
            img_list.append((img_label, img_path))
        index = 0  # index for the image
        while not self.stop:
            if index == 0 and self.shuffle:
                random.shuffle(img_list)
            if not self.queue.full():
                x = []
                y = np.empty(self.batch_size, dtype=np.int32)
                i = 0
                while i < self.batch_size:
                    img_label, img_path = img_list[index]
                    # print "self.image_folder: ", self.image_folder
                    # print "img_path: ", img_path
                    #print "os.path.join(self.image_folder, img_path): ", os.path.join(self.image_folder, img_path)
                    aug_images = self.image_transform(os.path.join(self.image_folder, img_path))
                    #print 'aug_images.norm: ', np.linalg.norm(aug_images)
                    # check, what is the len(aug_images)?
                    assert i + len(aug_images) <= self.batch_size, \
                        'too many images (%d) in a batch (%d)' % \
                        (i + len(aug_images), self.batch_size)
                    for img in aug_images:
                        ary = np.asarray(img)
                        x.append(ary)
                        y[i] = img_label
                        i += 1
                    index += 1
                    if index == self.num_samples:
                        index = 0  # reset to the first image
                # enqueue one mini-batch
                self.queue.put((np.asarray(x, np.float32), y))
            else:
                time.sleep(0.1)
        return


def load_from_dicom(img_path):
    new_imgs = []
    ary = np.asarray(dicom.read_file(img_path).pixel_array, dtype=dicom.read_file(img_path).pixel_array.dtype)
    new_imgs.append(ary)
    return new_imgs


def load_from_csv(img_path):
    img_path = img_path.replace('dcm', 'csv')
    new_imgs = []
    #print 'img path: ', img_path
    ary = np.genfromtxt(img_path, delimiter = ',')
    #print 'ary.norm: ', np.linalg.norm(ary)
    new_imgs.append(ary)
    return new_imgs


def load_from_np(img_path):
    img_path = img_path.replace('dcm', 'npy')
    new_imgs = []
    ary = np.fromfile(img_path)
    new_imgs.append(ary)
    return new_imgs


def get_mean(mean_dir, dtype = 'npy'):
    if dtype == 'npy':
        mean_path = os.path.join(mean_dir, 'mean.npy')
        mean = np.fromfile(mean_path)
        return mean
    elif dtype == 'csv':
        mean_path = os.path.join(mean_dir, 'mean.csv')
        mean = np.genfromtxt(mean_path, delimiter = ',')
        return np.asarray(mean, np.float32)
    else:
        raise Exception('Unsupported dtype: ', dtype)
