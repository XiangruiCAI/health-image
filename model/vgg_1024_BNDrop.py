#! /usr/bin/python
#encoding=utf-8

'''
Follow vgg_BNDrop2.py
Add batchnorm and dropout to avoid overfitting.

image size = 1024
'''

from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet

def ConvBnReLUPool(net, name, nb_filers, sample_shape=None):
    net.add(layer.Conv2D(name + '_conv', nb_filers, 3, 1, pad=1, input_sample_shape=sample_shape))
    net.add(layer.BatchNormalization(name + '_bn'))
    net.add(layer.Activation(name + '_relu'))
    net.add(layer.MaxPooling2D(name + '_pool', 2, 2, border_mode='valid'))

def create_net(input_shape, use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())

    ConvBnReLUPool(net, 'conv1', 32, input_shape)
    ConvBnReLUPool(net, 'conv2', 64)
    ConvBnReLUPool(net, 'conv3', 128)
    ConvBnReLUPool(net, 'conv4', 128)
    ConvBnReLUPool(net, 'conv5', 256)
    ConvBnReLUPool(net, 'conv6', 256)
    ConvBnReLUPool(net, 'conv7', 512)
    ConvBnReLUPool(net, 'conv8', 512)

    net.add(layer.Flatten('flat'))

    net.add(layer.Dense('ip1', 256))
    net.add(layer.BatchNormalization('bn1'))
    net.add(layer.Activation('relu1'))
    net.add(layer.Dropout('dropout1', 0.2))

    net.add(layer.Dense('ip2', 16))
    net.add(layer.BatchNormalization('bn2'))
    net.add(layer.Activation('relu2'))
    net.add(layer.Dropout('dropout2', 0.2))

    net.add(layer.Dense('ip3', 2))

    print 'Parameter intialization............'
    for (p, name) in zip(net.param_values(), net.param_names()):
        print name, p.shape
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv' in name:
                initializer.gaussian(p, 0, p.size())
            else:
                p.gaussian(0, 0.02)
        else:
            p.set_value(0)
        print name, p.l1()

    return net
