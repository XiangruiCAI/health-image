import numpy as np
from singa.proto import core_pb2
from singa import device
from singa import tensor
from singa import image_tool
from model import resnet
import conf


# path to the best model (snapshot)
path_model = '/home/xiangrui/health-image_0429/snapshot/cv8/best_model'
# image mean, obtained from preprocessing
mean = np.asarray([124.76510401, 124.76510401, 124.76510401])


validate_tool = image_tool.ImageTool()
def load_img(path_img, size):
    '''
    path_img: path to the image to predict
    size: input size of the model
    '''
    global validate_tool
    images = validate_tool.load(path_img).resize_by_list([size]).get()
    x = []
    for img in images:
        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
        x.append(ary.transpose(2, 0, 1))
    x = np.asarray(x) - mean[np.newaxis, :, np.newaxis, np.newaxis]
    return np.asarray(x, dtype=np.float32)


def predict(path_img, topk=2):
    '''
    path_img: path to the image to predict
    topk: return topk tuples in terms of probability

    return: list of topk tuples, sorted by the probability.
    ABOUT THE CLASSES: 0 refers to normal and 1 refers to abnormal.
    '''
    cnf = conf.Conf()
    # use cpu
    dev = device.get_default_device()
    # load the best model
    net = resnet.create_net(cnf.net, cnf.depth, True)
    net.load(path_model, 200)
    net.to_device(dev)
    # load an image
    x = load_img(path_img, cnf.crop_size)
    tx = tensor.Tensor((1,) + cnf.input_shape, dev)
    ty = tensor.Tensor((1,), dev, core_pb2.kInt)
    tx.copy_from_numpy(x)
    # predict
    ty = net.predict(tx)
    ty.to_host()
    prob = tensor.to_numpy(ty)
    order = np.argsort(-prob)
    res = []
    for i in order[0]:
        res.append((i, prob[0, i]))
    size = min(2, topk)
    return res[:size]

if __name__ == '__main__':
    # test
    path_img = 'data/xray/normal/8580297/I00007391013.jpg'
    #path_img = 'data/xray/abnormal_nodule_0228/81129121/I00128946508.jpg'
    print predict(path_img)

