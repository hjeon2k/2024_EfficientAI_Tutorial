import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import time
from utils import AverageMeter
from utils import accuracy
from utils.progress.progress.bar import Bar as Bar
from utils import get_mnist_train_valid_loader
from utils import get_mnist_test_loader

from models import MLP
from quant import lsq_prepare

os.environ['CUDA_VISIBLE_DEVICES'] = '0'                                            # GPU number that you will use

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
    start_epoch = 0

    train_loader, valid_loader = get_mnist_train_valid_loader(                      # you can change Batch_size, valid_size (ratio)
            data_dir='/home/intern1/data/', batch_size=64,
            random_seed=42, valid_size=0.1, shuffle=True, show_sample=False,
            num_workers=4, pin_memory=False)
    test_loader = get_mnist_test_loader(
            data_dir='/home/intern1/data/', batch_size=64, shuffle=False,
            num_workers=4, pin_memory=False)

    model = MLP(input_size=32*32, hidden_dim=128, output_class=10).cuda()           # you can change model's hidden size

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)                # you can change a optimizer. (adam, rmsprop ...)
                                                                                    # you can also change learning rate.
    # optimizer = optim.Adam(model.parameters(), lr=2e-3)

    num_params = 0
    for params in model.parameters():
        num_params += params.view(-1).size(0)
    print("# of parameters : " + str(num_params))

    total_epochs = 20                                                               # you can change total epochs
    lsq_start = 15

    for epoch in range(1, total_epochs):
        if epoch == lsq_start:
            model = lsq_prepare(model, w_bits=4, quant_inference=True, per_channel=False, batch_init=20)
            print("\n***LSQ Training Prepared***")

            num_params = 0
            for params in model.parameters():
                num_params += params.view(-1).size(0)
            print("# of parameters : " + str(num_params))

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch: [{epoch} | {total_epochs}] LR: {current_lr:.3e}")

        train_loss, train_top1, train_top5 = train(
            train_loader, model, criterion,
            optimizer, epoch)

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
