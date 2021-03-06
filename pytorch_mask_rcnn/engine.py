import time

import torch
import sys

from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    iter_performed = 0
    for _, (image, target) in enumerate(data_loader):
        if target['boxes'].shape[0] > 0:
            T = time.time()
            num_iters = epoch * len(data_loader) + iter_performed
            if num_iters <= args.warmup_iters:
                r = num_iters / args.warmup_iters
                for j, p in enumerate(optimizer.param_groups):
                    p["lr"] = r * args.lr_epoch

            image = image.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            S = time.time()

            losses = model(image, target)
            total_loss = sum(losses.values())
            m_m.update(time.time() - S)

            S = time.time()
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            b_m.update(time.time() - S)

            if num_iters % args.print_freq == 0:
                print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

            t_m.update(time.time() - T)
            iter_performed += 1
        if iter_performed >= iters:
            break

    print("Model trained with {} images".format(iter_performed))
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters, 1000*t_m.avg,
                                                                                1000*m_m.avg,
                                                                                1000*b_m.avg))
    return A / iters
            

def evaluate(model, data_loader, device, args, generate=True):
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader #
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load("results.pth", map_location="cpu")

    S = time.time()
    if len(results) > 0:
        coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()
    if len(results) > 0:
        coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = args.test_iter # TODO: why test has the same iters as train -> if args.iters < 0 else args.iters
    ann_labels = data_loader.ann_labels
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    n_iters = 0
    for i, (image, target) in enumerate(data_loader):
        if target['boxes'].shape[0] > 0:
            T = time.time()

            image = image.to(device)
            target = {k: v.to(device) for k, v in target.items()}

            S = time.time()
            torch.cuda.synchronize()
            output = model(image)
            m_m.update(time.time() - S)

            prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
            coco_results.extend(prepare_for_coco(prediction, ann_labels))

            t_m.update(time.time() - T)
            n_iters += 1
        if n_iters >= iters:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, num_iter: {}".format(1000*A/iters, 1000*t_m.avg,1000*m_m.avg,
                                                                            args.test_iter))
    
    S = time.time()
    print("all gather: {:.1f}s".format(time.time() - S))
    torch.save(coco_results, "results.pth")
        
    return A / iters
    

