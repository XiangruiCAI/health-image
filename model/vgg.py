#! /usr/bin/python
#encoding=utf-8

from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet


def create_net(input_shape, use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())

    net.add(layer.Conv2D('conv1', nb_kernels=32, kernel=7, stride=3, pad=1,
        input_sample_shape=input_shape))
    net.add(layer.Activation('relu1'))
    net.add(layer.MaxPooling2D('pool1', 2, 2, border_mode='valid'))

    net.add(layer.Conv2D('conv2', nb_kernels=64, kernel=5, stride=3))
    net.add(layer.Activation('relu2'))
    net.add(layer.MaxPooling2D('pool2', 2, 2, border_mode='valid'))

    net.add(layer.Conv2D('conv3', nb_kernels=128, kernel=3, stride=1, pad=2))
    net.add(layer.Activation('relu3'))
    net.add(layer.MaxPooling2D('pool3', 2, 2, border_mode='valid'))

    net.add(layer.Conv2D('conv4', nb_kernels=256, kernel=3, stride=1))
    net.add(layer.Activation('relu4'))
    net.add(layer.MaxPooling2D('pool4', 2, 2, border_mode='valid'))

    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('ip5', 256))
    net.add(layer.Activation('relu5'))
    net.add(layer.Dense('ip6', 16))
    net.add(layer.Activation('relu6'))
    net.add(layer.Dense('ip7', 2))

    print 'Start intialization............'
    for (p, name) in zip(net.param_values(), net.param_names()):
        print name, p.shape
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv1' in name:
                initializer.gaussian(p, 0, p.size() / 10)
            elif 'conv' in name:
                initializer.gaussian(p, 0, p.size())
            else:
                p.gaussian(0, 0.02)
        else:
            p.set_value(0)
        print name, p.l1()

    return net
