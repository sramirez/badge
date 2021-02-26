import numpy as np
import sys
import os
import argparse
from args_templates import args_template
from dataset import get_dataset, get_handler, get_VOC_detection
import vgg
import resnet
from torchvision import transforms
import torch
from mlp_mod import mlpMod
from openml_dataset import load_openml_data
from query_strategies import RandomSampling, BadgeSampling, \
                                BaselineSampling, LeastConfidence, MarginSampling, \
                                EntropySampling, LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning

# code based on https://github.com/ej0cl6/deep-active-learning"
from train_object_detector import train_object_detector

parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=10)
parser.add_argument('--nStart', help='number of points to start', type=int, default=10)
parser.add_argument('--nEnd', help='total number of points to query', type=int, default=50000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)
# object detection parameters
parser.add_argument("--seed", type=int, default=3)
parser.add_argument('--lr-steps', nargs="+", type=int, default=[22, 26])
parser.add_argument("--lr_object", type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--iters", type=int, default=200, help="max iters per epoch, -1 denotes auto")
parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
parser.add_argument("--test_iter", type=int, default=100, help="number of iterations during evaluation time")
opts = parser.parse_args()

if opts.lr_object is None:
    opts.lr_object = 0.02 * 1 / 16  # lr should be 'batch_size / 16 * 0.02'

# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB)/ opts.nQuery)
DATA_NAME = opts.data

opts.nClasses = 10
args_pool = args_template.copy()
args_pool['CIFAR10']['transform'] = args_pool['CIFAR10']['transformTest'] # remove data augmentation
args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']
args_pool['VOC']['transformTest'] = args_pool['VOC']['transform']
args_pool['COCO']['transformTest'] = args_pool['COCO']['transform']

if opts.did == 0:
    args = args_pool[DATA_NAME]
    opts.nClasses = args['nClasses']

if not os.path.exists(opts.path):
    os.makedirs(opts.path)

# load openml dataset if did is supplied
if opts.did > 0:
    X_tr, Y_tr, X_te, Y_te, nClasses = load_openml_data(opts)
    opts.nClasses = nClasses
    args = {'transform':transforms.Compose([transforms.ToTensor()]),
            'n_epoch':10,
            'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
            'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
            'optimizer_args':{'lr': 0.01, 'momentum': 0},
            'transformTest':transforms.Compose([transforms.ToTensor()])}
    handler = get_handler('other')

# load non-openml dataset
else:
    if DATA_NAME == 'VOC':
        X_tr, d_train, Y_tr, X_te, d_test, Y_te = get_VOC_detection(opts.path)
    else:
        X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)

    opts.dim = np.shape(X_tr)[1:]
    handler = get_handler(opts.data)

args['lr'] = opts.lr
# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# load specified network
if opts.model == 'mlp':
    net = mlpMod(opts.dim, nClasses=opts.nClasses, embSize=opts.nEmb)
elif opts.model == 'resnet':
    net = resnet.ResNet18()
elif opts.model == 'vgg':
    net = vgg.VGG('VGG16')
else: 
    print('choose a valid model - mlp, resnet, or vgg', flush=True)
    raise ValueError

if opts.did > 0 and opts.model != 'mlp':
    print('openML datasets only work with mlp', flush=True)
    raise ValueError

if type(X_tr[0]) is not np.ndarray and not list:
    X_tr = X_tr.numpy()

# set up the specified sampler
if opts.alg == 'rand': # random sampling
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'conf': # confidence-based sampling
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'marg': # margin-based sampling
    strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'badge': # batch active learning by diverse gradient embeddings
    strategy = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'coreset': # coreset sampling
    strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'entropy': # entropy-based sampling
    strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'deepfool':
    strategy = AdversarialDeepFool(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'baseline': # badge but with k-DPP sampling instead of k-means++
    strategy = BaselineSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'albl': # active learning by learning
    albl_list = [LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args),
        CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)]
    strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
else: 
    print('choose a valid acquisition function', flush=True)
    raise ValueError

# print info
if opts.did > 0:
    DATA_NAME ='OML' + str(opts.did)
print(DATA_NAME, flush=True)
print(type(strategy).__name__, flush=True)

# round 0 accuracy
strategy.train()
output = ''
if DATA_NAME == 'VOC':
    subset = torch.utils.data.Subset(d_train, np.arange(len(d_train))[strategy.idxs_lb])
    opts.iters = len(subset)
    ap = train_object_detector(subset, d_test, opts)
    output += str(opts.nStart) + '\ttesting mAP {}'.format(ap) + '\n'
    print(output, flush=True)

P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
output += str(opts.nStart) + '\ttesting accuracy {}'.format(acc[0]) + '\n'
print(output, flush=True)

# Write to file
file_object = open('results_plain.txt', 'x')
file_object.write(output)
file_object.close()


file_object = open('results_plain.txt', 'a')

for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd), flush=True)

    # query
    q_idxs = strategy.query(NUM_QUERY)
    idxs_lb[q_idxs] = True

    print('====================== Training set size: {}'.format(idxs_lb.sum()))

    # report weighted accuracy
    corr = (strategy.predict(X_tr[q_idxs], torch.Tensor(Y_tr.numpy()[q_idxs]).long())).numpy() == Y_tr.numpy()[q_idxs]

    # update
    strategy.update(idxs_lb)
    strategy.train()
    output = ''
    if DATA_NAME == 'VOC':
        subset = torch.utils.data.Subset(d_train, np.arange(len(d_train))[strategy.idxs_lb])
        opts.iters = len(subset)
        ap = train_object_detector(subset, d_test, opts)
        output += str(opts.nStart) + '\ttesting mAP {}'.format(ap) + '\n'
        print(output, flush=True)

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    output += str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(acc[rd]) + '\n'
    print(output, flush=True)
    if sum(~strategy.idxs_lb) < opts.nQuery: 
        sys.exit('too few remaining points to query')
    file_object.write(output)

file_object.close()
