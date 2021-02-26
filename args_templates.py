# non-openml data defaults

from torchvision import transforms

args_template = {'MNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
                 'nClasses': 10},
            'FashionMNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
                 'nClasses': 10}, # TODO: change it
            'SVHN':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
                 'nClasses': 10},
            'CIFAR10':
                {'n_epoch': 3, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.3},
                 'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'nClasses': 2},
             'VOC':
                 {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize(
                                                                      (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
                  'nClasses': 2},
             'COCO':
                 {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize(
                                                                      (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                  'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.01, 'momentum': 0.5},
                  'nClasses': 2}
                }