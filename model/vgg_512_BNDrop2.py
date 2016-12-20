#! /usr/bin/python
#encoding=utf-8

'''
Follow vgg_BNDrop2.py
Add batchnorm and dropout to avoid overfitting.

image size = 512
'''

from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet

def ConvBnReLU(net, name, nb_filers, sample_shape=None):
    net.add(layer.Conv2D(name + '_1', nb_filers, 3, 1, pad=1, input_sample_shape=sample_shape))
    net.add(layer.BatchNormalization(name + '_2'))
    net.add(layer.Activation(name + '_3'))

def create_net(input_shape, use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())

    ConvBnReLU(net, 'conv1_1', 64, input_shape)
    #net.add(layer.Dropout('drop1', 0.3))
    net.add(layer.MaxPooling2D('pool0', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv1_2', 128)
    net.add(layer.MaxPooling2D('pool1', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv2_1', 128)
    net.add(layer.Dropout('drop2_1', 0.4))
    ConvBnReLU(net, 'conv2_2', 128)
    net.add(layer.MaxPooling2D('pool2', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv3_1', 256)
    net.add(layer.Dropout('drop3_1', 0.4))
    ConvBnReLU(net, 'conv3_2', 256)
    net.add(layer.Dropout('drop3_2', 0.4))
    ConvBnReLU(net, 'conv3_3', 256)
    net.add(layer.MaxPooling2D('pool3', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv4_1', 256)
    net.add(layer.Dropout('drop4_1', 0.4))
    ConvBnReLU(net, 'conv4_2', 256)
    net.add(layer.Dropout('drop4_2', 0.4))
    ConvBnReLU(net, 'conv4_3', 256)
    net.add(layer.MaxPooling2D('pool4', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv5_1', 512)
    net.add(layer.Dropout('drop5_1', 0.4))
    ConvBnReLU(net, 'conv5_2', 512)
    net.add(layer.Dropout('drop5_2', 0.4))
    ConvBnReLU(net, 'conv5_3', 512)
    net.add(layer.MaxPooling2D('pool5', 2, 2, border_mode='valid'))
    #ConvBnReLU(net, 'conv6_1', 512)
    #net.add(layer.Dropout('drop6_1', 0.4))
    #ConvBnReLU(net, 'conv6_2', 512)
    #net.add(layer.Dropout('drop6_2', 0.4))
    #ConvBnReLU(net, 'conv6_3', 512)
    #net.add(layer.MaxPooling2D('pool6', 2, 2, border_mode='valid'))
    #ConvBnReLU(net, 'conv7_1', 512)
    #net.add(layer.Dropout('drop7_1', 0.4))
    #ConvBnReLU(net, 'conv7_2', 512)
    #net.add(layer.Dropout('drop7_2', 0.4))
    #ConvBnReLU(net, 'conv7_3', 512)
    #net.add(layer.MaxPooling2D('pool7', 2, 2, border_mode='valid'))

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
