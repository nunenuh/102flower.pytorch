import time
import numpy as np
import torch
import metrics


def getsteplr(base_lr=0.001, max_lr=0.1, step=4):
    lr = base_lr
    hlr = max_lr
    step = hlr/(step-1)
    step_lr = np.arange(lr, hlr+step, step).tolist()
    return step_lr


def adjust_learning_rate(optimizer, epoch, decay, lrate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrate * (0.1 ** (epoch // decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


batch_history = {
    'train': {'epoch': [], 'loss': [], 'acc_topk1': [], 'acc_topk5': []},
    'valid': {'epoch': [], 'loss': [], 'acc_topk1': [], 'acc_topk5': []}
}


def train(train_loader, model, criterion, optimizer, epoch, print_freq, save_history=True, ngpu=1):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    batch_time = metrics.AverageMeter()
    data_time = metrics.AverageMeter()
    losses = metrics.AverageMeter()
    top1 = metrics.AverageMeter()
    top5 = metrics.AverageMeter()
    history = {'epoch': [], 'loss': [], 'acc_topk1': [], 'acc_topk5': []}

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure metrics.accuracy and record loss
        acc1, acc5 = metrics.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            history['epoch'].append(epoch)
            history['loss'].append(losses.avg)
            history['acc_topk1'].append(top1.avg)
            history['acc_topk5'].append(top5.avg)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    return history


def validate(val_loader, model, criterion, epoch, print_freq, save_history=True, ngpu=1):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    batch_time = metrics.AverageMeter()
    losses = metrics.AverageMeter()
    top1 = metrics.AverageMeter()
    top5 = metrics.AverageMeter()
    history = {'epoch': [], 'loss': [], 'acc_topk1': [], 'acc_topk5': []}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure metrics.accuracy and record loss
            acc1, acc5 = metrics.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                history['epoch'].append(epoch)
                history['loss'].append(losses.avg)
                history['acc_topk1'].append(top1.avg)
                history['acc_topk5'].append(top5.avg)
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return history

