from singa import device
from singa import tensor
from singa.proto import core_pb2
import numpy as np

from model import vgg_BNDrop2
from data_loader import data as dt
import conf


def predict(net, cfg, dev, topk=5):
    '''Predict the label of each image.
    Args:
        net, a pretrained neural net
        dev, the training device
        topk, return the topk labels for each image.
    '''

    dl_test = dt.MImageBatchIter(cfg.test_file, cfg.batch_size, dt.load_from_img,
            shuffle=False, delimiter=' ', image_folder=cfg.input_folder, capacity=10)
    dl_test.start()
    mean = dt.get_mean(cfg.input_folder)
    num_test = dl_test.num_samples
    num_test_batch = num_test / cfg.batch_size
    remainder = num_test % cfg.batch_size

    tx = tensor.Tensor((cfg.batch_size,) + cfg.input_shape, dev)
    ty = tensor.Tensor((cfg.batch_size,), dev, core_pb2.kInt)
    ground_truth = []
    predict = []
    print 'num_test_batch: ', num_test_batch
    for b in range(num_test_batch):
        print 'batch ', b
        x, y = dl_test.next()
        ground_truth.extend(y.tolist())
        x -= mean
        tx.copy_from_numpy(x)
        ty = net.predict(tx)
        ty.to_host()
        prob = tensor.to_numpy(ty)
        labels = np.fliplr(np.argsort(prob))  # sort prob in descending order
        print labels[:,:topk]
        predict.extend(labels[:,:topk].tolist())
    if remainder > 0:
        print 'remainder: ', remainder
        x, y = dl_test.next()
        ground_truth.extend(y[0:remainder,].tolist())
        x -= mean
        tx_rmd = tensor.Tensor((remainder,) + cfg.input_shape, dev)
        ty_rmd = tensor.Tensor((remainder,), dev, core_pb2.kInt)
        tx_rmd.copy_from_numpy(x[0:remainder,])
        ty_rmd = net.predict(tx_rmd)
        ty_rmd.to_host()
        prob = tensor.to_numpy(ty_rmd)
        labels = np.fliplr(np.argsort(prob))  # sort prob in descending order
        print labels[:,:topk]
        predict.extend(labels[:,:topk].tolist())

    dl_test.end()
    return predict, ground_truth


if __name__ == '__main__':
    cfg = conf.Conf()
    if cfg.net == 'vgg_BNDrop2':
        model = vgg_BNDrop2.create_net(cfg.input_shape, True)
    else:
        raise Exception('Unsupported net: ', cnf.net)
    model.load(cfg.best_model, 200)
    dev = device.get_default_device()
    model.to_device(dev)

    p, g = predict(model, cfg, dev, 1)
    print p
    print g
