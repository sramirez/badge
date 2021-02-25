import bisect
import time
import torch
import pytorch_mask_rcnn as pmr


def train_object_detector(d_train, d_test, args) -> float:
    # init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.warmup_iters = max(1000, len(d_train))
    num_classes = len(d_train.dataset.classes) + 1  # including background class
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr_object, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect([22, 26], x)
    start_epoch = 0
    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))

    # ------------------------------- train ------------------------------------ #
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr_object
        print("lr_epoch: {:.4f}, factor: {:.4f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        B = time.time()
        eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        B = time.time() - B
        trained_epoch = epoch + 1
        print("training: {:.2f} s, evaluation: {:.2f} s".format(A, B))
        pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        print(eval_output.get_AP())

    return eval_output.get_AP()