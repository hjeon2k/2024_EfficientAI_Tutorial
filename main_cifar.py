from __future__ import print_function

import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import time
import models
from utils import AverageMeter
from utils import accuracy
from utils.progress.progress.bar import Bar as Bar
from utils import get_cifar10_100_train_valid_loader
from utils import get_cifar10_100_test_loader

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--gpu-id', default='0', type=str,
                                   help='ID(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-d', '--data', default='/home/intern1/data/cifar10/', type=str, help='path to dataset')
parser.add_argument('--valid-size', default=0.1, type=float,
                    help='Ratio of valid set split from training set of CIFAR \
                    dataset. If 0, no validation set used.')
parser.add_argument('-o', '--optimizer', default='sgd', type=str,
                    choices=['sgd', 'adam', 'adamax', 'adamw'],
                    help='Optimizer to be used. (default: sgd)')
parser.add_argument('--lr-method', default='lr_step', type=str,
                    choices=['lr_step', 'lr_linear', 'lr_exp', 'lr_cosineanneal'],
                    help='Set learning rate scheduling method.')
parser.add_argument('--schedule', nargs='+', default=[100, 150], type=int,
                    help='Decrease learning rate at these epochs when using step method')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='LR is multiplication factor')
parser.add_argument('--T0', default=10, type=int,
                    help='Number of steps for the first restart in SGDR')
parser.add_argument('--T-mult', default=1, type=int,
                    help='A factor increases T_{i} after restart in SGDR')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

best_valid_top1 = 0
best_valid_top5 = 0
best_test_top1 = 0
test_top1 = 0

def train(train_loader, model, criterion, optimizer, epoch):

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    len_trainloader = len(train_loader)
    bar = Bar('Processing', max=len_trainloader)

    for batch_idx, data in enumerate(train_loader):                                # repeat training for each mini-batch
        inputs, targets = data
        inputs = inputs.cuda()
        targets = targets.cuda()

        data_time.update(time.time() - end)

        batch_time.update(time.time() - end)
        end = time.time()

        outputs = model(inputs)                                                     # outputs: predicted target value

        loss = criterion(outputs, targets)                                          # calculate loss
                                                                                    # Note that criterion is set to CrossEntropyLoss in the main function
        optimizer.zero_grad()                                                       # make gradient=0 before backpropagation

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))            # this parts calculate the accuracy

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        loss.backward()                                                             # backpropagation: calculate gradients for all hyperparameters
        optimizer.step()                                                            # Update the parameters
                                                                                    # Note that optimizer is set to SGD in the main function

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len_trainloader,
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def test(val_loader, model, criterion, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()

    len_valloader = len(val_loader)
    bar = Bar('Processing', max=len_valloader)
    with torch.no_grad():                                                           # we don't need to calculate gradient for the validation/test part
        for batch_idx, data in enumerate(val_loader):
            inputs, targets = data
            inputs = inputs.cuda()
            targets = targets.cuda()
            data_time.update(time.time() - end)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len_valloader,
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def main():
    global best_valid_top1, best_valid_top5, best_test_top1, test_top1
    args = parser.parse_args()

    train_loader, train_sampler, valid_loader = get_cifar10_100_train_valid_loader(
            dataset='cifar10', data_dir=args.data, batch_size=args.batch_size,
            augment=True, random_seed=42, valid_size=args.valid_size, shuffle=True,
            num_workers=args.workers, distributed=False,
            pin_memory=False) # True for CUDA?
    test_loader = get_cifar10_100_test_loader(
            dataset='cifar10', data_dir=args.data, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=False) # True for CUDA?

    model = models.__dict__[args.arch]().cuda()                                             # you can change model's hidden size

    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_method == 'lr_step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, args.gamma)
    elif args.lr_method == 'lr_linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1.0-step/args.epochs))
    elif args.lr_method == 'lr_exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    elif args.lr_method == 'lr_cosineanneal':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.T0, args.T_mult)

    num_params = 0
    for params in model.parameters():
        num_params += params.view(-1).size(0)
    print("# of parameters : " + str(num_params))

    for epoch in range(1, args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch: [{epoch} | {args.epochs}] LR: {current_lr:.3e}")

        train_loss, train_top1, train_top5 = train(
            train_loader, model, criterion,
            optimizer, epoch)
        lr_scheduler.step()

        valid_loss, valid_top1, _ = test(
                    valid_loader, model, criterion, epoch)
        test_loss, test_top1, test_top5 = test(
                    test_loader, model, criterion, epoch)

        if valid_top1 > best_valid_top1:
            best_valid_top1 = valid_top1
            best_test_top1 = test_top1

    print('Test top1 @ best valid top1:')
    print(f"{best_test_top1:.2f}")
    print('Test top1 @ last epoch:')
    print(f"{test_top1:.2f}")


if __name__ == '__main__':
    main()
