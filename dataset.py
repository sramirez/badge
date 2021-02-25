import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import pytorch_mask_rcnn as pmr
from pytorch_mask_rcnn.datasets.voc_dataset import VOC_CLASSES


def get_dataset(name, path):
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(path)
    elif name == 'SVHN':
        return get_SVHN(path)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)
    elif name == 'VOC':
        return get_VOC_detection(path)
    elif name == 'COCO':
        return get_COCO_detection(path)

def get_COCO_detection(path):
    raw_tr = datasets.CocoDetection(path + '/COCO', image_set='train', download=True)
    raw_te = datasets.CocoDetection(path + '/COCO', image_set='val', download=True)
    X_tr = raw_tr.images
    Y_tr = raw_tr.annotations
    X_te = raw_te.images
    Y_te = raw_te.annotations
    return X_tr, Y_tr, X_te, Y_te


def get_VOC_classification(path):
    raw_tr = datasets.VOCDetection(path + '/VOC', image_set='train', download=True)
    raw_te = datasets.VOCDetection(path + '/VOC', image_set='val', download=True)
    X_tr = raw_tr.images
    Y_tr = raw_tr.annotations
    X_te = raw_te.images
    Y_te = raw_te.annotations
    return X_tr, Y_tr, X_te, Y_te


def _transform_voc(ds):
    X_tr = []
    Y_tr = []
    Y_cls_tr = []
    for x in ds:
        img = torch.nn.functional.interpolate(x[0].unsqueeze(0), size=(128, 128), mode='bilinear')
        X_tr.append(np.transpose(img.squeeze(0).numpy(), (1, 2, 0)))
        found_class_indices = [i for i, x in enumerate(x[1]['labels']) if x == VOC_CLASSES.index('person')]
        Y_cls_tr.append(1 if len(found_class_indices) > 0 else 0)
        Y_tr.append({'image_id': x[1]['image_id'], 'boxes': x[1]['boxes'][found_class_indices],
            'labels': x[1]['labels'][found_class_indices], 'masks': x[1]['masks'][found_class_indices]})

    return np.array(X_tr), Y_tr, torch.tensor(Y_cls_tr)


def get_VOC_detection(path):
    dataset_train = pmr.datasets("voc", "data/VOC/VOCdevkit/VOC2012", "train", train=True)
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    d_test = pmr.datasets("voc", "data/VOC/VOCdevkit/VOC2012", "val", train=True)  # set train=True for eval and to get labels

    X_tr, Y_tr_detection, Y_tr_cls = _transform_voc(d_train)
    X_te, Y_te_detection, Y_te_cls = _transform_voc(d_test)
    return X_tr, d_train, Y_tr_cls, X_te, d_test, Y_te_cls

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te


def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te


def get_SVHN(path):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te


def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'VOC':
        return DataHandlerVOC
    elif name == 'VOC_detection':
        return DataHandlerVOCDetection
    else:
        return DataHandler4


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandlerVOC(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray((x * 255).astype(np.uint8))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandlerVOCDetection(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

