import numpy as np
import torch
import torch.nn as nn
from dataloader import DataLoader
from model import Model
from utils import AverageMeter, accuracy, F1_Score
from dice_loss import DiceLoss
import argparse
import time
import warnings
import mmcv
warnings.filterwarnings("ignore")
try:
    import wandb
except:
    pass

def adjust_learning_rate(optimizer, dataloader, epoch, iter):
    cur_iter = epoch * len(dataloader) + iter
    max_iter_num = args.epoch * len(dataloader)
    lr = args.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

def valid(valid_loader, model, epoch):
    model.eval()
    
    for iter, (x, y) in enumerate(valid_loader):
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs, y)
            acc = ((outputs > 0) == y).sum(dim=0).float() / args.valid_batch_size

    mean_acc = acc.mean()
    output_log = '(Valid)  Loss: {loss:.3f} | Mean Acc: {acc:.3f}'.format(
        loss=loss.item(),
        acc=mean_acc.item()
    )
    print(output_log)
    print(acc)
    if args.wandb:
        wandb.log({'epoch': epoch,
                   'Caco-2': acc[0].item(),
                   'CYP3A4': acc[1].item(),
                   'hERG': acc[2].item(),
                   'HOB': acc[3].item(),
                   'MN': acc[4].item(),
                   'Mean': mean_acc.item()})
    return mean_acc

def train(train_loader, model, optimizer, epoch):
    model.train()

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # start time
    start = time.time()
    for iter, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter)

        outputs = model(x)
        loss = criterion(outputs, y)
        with torch.no_grad():
            acc = ((outputs > 0) == y).sum(dim=0).float() / args.batch_size
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 10 == 0:
            output_log = '({batch}/{size}) LR: {lr:.6f} | Batch: {bt:.3f}s | Total: {total:.0f}min | ' \
                         'ETA: {eta:.0f}min | Loss: {loss:.3f} | ' \
                         'Mean Acc: {acc:.3f}'.format(
                batch=iter + 1,
                size=len(train_loader),
                lr=optimizer.param_groups[0]['lr'],
                bt=batch_time.avg,
                total=batch_time.avg * iter / 60.0,
                eta=batch_time.avg * (len(train_loader) - iter) / 60.0,
                loss=loss.item(),
                acc=acc.mean().item()
            )
            
            print(output_log)
            print(acc)
        # if args.wandb:
        #     wandb.log({'epoch': epoch,
        #                'Caco-2': acc[0].item(),
        #                'CYP3A4': acc[1].item(),
        #                'hERG': acc[2].item(),
        #                'HOB': acc[3].item(),
        #                'MN':acc[4].item()})

def main():
    train_loader = torch.utils.data.DataLoader(
        DataLoader(split="train"), batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=True, pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        DataLoader(split="valid"), batch_size=args.valid_batch_size,
        shuffle=False, num_workers=0, drop_last=False, pin_memory=True
    )
    model = Model().cuda()
    model_structure(model)
    if args.wandb:
        wandb.watch(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch, start_iter, best_mean_acc = 0, 0, 0
    for epoch in range(start_epoch, args.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epoch))
        train(train_loader, model, optimizer, epoch)
        mean_acc = valid(valid_loader, model, epoch)
        if mean_acc >= best_mean_acc:
            best_mean_acc = mean_acc
            
            torch.save(model.state_dict(), "checkpoint/checkpoint.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--epoch', default=1000, type=int, help='epoch')
    parser.add_argument('--batch_size', default=1776, type=int, help='batch size')
    parser.add_argument('--valid_batch_size', default=198, type=int, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='batch size')
    parser.add_argument('--wandb', action='store_true', help='use wandb')

    mmcv.mkdir_or_exist("checkpoint/")
    args = parser.parse_args()
    print(args)
    # torch.backends.cudnn.benchmark = True
    
    if args.wandb:
        wandb.init(project="math-model")
        
    criterion = DiceLoss(loss_weight=1.0)
    main()