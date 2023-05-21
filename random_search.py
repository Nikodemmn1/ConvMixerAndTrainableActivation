import warnings
import torch
import random
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import loguniform
####
import gc
import time
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
####
import time
import sys
import numpy as np
import pandas as pd
import os

from activation import CosLU


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        CosLU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size,
                          groups=dim, padding="same"),
                CosLU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            CosLU(),
            nn.BatchNorm2d(dim)
        ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


try:
    sys.path.insert(0, '.')
    from model.base import Model
    from dataset.base import Dataset
    from metric import metrics, accuracy
    from optimizer import Optimizer
    from scheduler import Scheduler, print_scheduler
    from tools import parse, get_logger, print_dct
finally:
    pass
warnings.filterwarnings('ignore')


TRAINING_NUMBER = 15


def train(args):
    args_str = print_dct(args)

    torch.manual_seed(args['train']['seed'])
    random.seed(args['train']['seed'])

    epochs = args['train']['epochs']
    bs = args['dataset']['batch_size']
    device_str = args['train']['device']
    device = torch.device(args['train']['device'])

    optimizer_name = args['optimizer']['name']
    optimizer_params = args['optimizer']['parameters']

    model = ConvMixer(args['model']['architecture']['h_dim'],
                      args['model']['architecture']['depth'],
                      args['model']['architecture']['conv_ks'],
                      args['model']['architecture']['psize'],
                      args['model']['architecture']['out_channels'])

    optimizer = Optimizer(optimizer_name, optimizer_params, model.parameters())
    scheduler = Scheduler(optimizer, args['scheduler'])

    activation_name = args['model']['architecture']['activation']
    lr = args['optimizer']['parameters']['lr']
    batch_size = args['dataset']['batch_size']
    weight_decay = args['optimizer']['parameters']['weight_decay']
    gamma = args['scheduler']['scheduler1']['parameters']['gamma']
    momentum = args['optimizer']['parameters']['momentum']
    conv_ks = args['model']['architecture']['conv_ks']
    psize = args['model']['architecture']['psize']
    ra_m = args['dataset']['augmentations']['ra_m']
    comment = f" activation_name = {activation_name} lr = {lr} batch_size = {batch_size} weight_decay = {weight_decay} momentum = {momentum} gamma = {gamma} psize = {psize} conv_ks = {conv_ks} ra_m = {ra_m}"

    tb = SummaryWriter(comment=comment)

    loss_fn = CrossEntropyLoss()

    if 'checkpoint' in args['train'].keys():
        model.load_state_dict(torch.load(args['train']['checkpoint']))

    log_path = f"{args['train']['log_path']}{args['model']['name'].lower()}/"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = get_logger(log_path + f"train{comment}.log")

    save_path = f"{args['train']['save_path']}{args['model']['name'].lower()}/"

    model = model.to(device)

    # logger.info(f'\n{args_str}')
    # logger.info(f'\n{str(model)}')
    # logger.info(f'\n{str(optimizer)}')
    # scheduler_str = print_scheduler(scheduler, args['scheduler'])
    # logger.info(scheduler_str)
    # logger.info(f'\n{str(loss_fn)}')

    train_dataset, valid_dataset, _ = Dataset(args['dataset'])

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=bs,
                                   shuffle=True)

    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=bs,
                                   shuffle=True)

    init_time = time.time()
    best_loss = 99999999.0

    epochs_npz = np.array([ep for ep in range(epochs)])
    train_loss_npz = []
    valid_loss_npz = []

    for epoch in range(epochs):
        start = time.time()
        model.train()

        train_loss = 0.0
        valid_loss = 0.0

        for batch in train_data_loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch in valid_data_loader:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs)
                loss = loss_fn(preds, labels)
                valid_loss += loss.item()

        train_loss /= len(train_data_loader)
        valid_loss /= len(valid_data_loader)

        train_accuracy = accuracy(model, train_data_loader, device)
        valid_accuracy = accuracy(model, valid_data_loader, device)

        tb.add_scalar("Train Loss", train_loss, epoch)
        tb.add_scalar("Validation Loss", valid_loss, epoch)
        tb.add_scalar("Train Accuracy", train_accuracy, epoch)
        tb.add_scalar("Validation Accuracy", valid_accuracy, epoch)

        train_loss_npz.append(train_loss)
        valid_loss_npz.append(valid_loss)

        cond1 = valid_loss < best_loss
        cond3 = epoch == epochs - 1

        if cond1:
            best_loss = valid_loss

        if cond3:
            state_dict = model.state_dict()
            save_name = f'ep={epoch + 1}_lv={valid_loss:.2f}_{comment}.pth'
            torch.save(state_dict, save_path + save_name)

        end = time.time()

        print_str = f'{device_str} '
        print_str += f'epoch: {epoch + 1}/{epochs} '
        print_str += f'train_loss: {train_loss:.4f} '
        print_str += f'valid_loss: {valid_loss:.4f} '
        print_str += f'epoch_time: {(end - start):.3f} sec'
        logger.info(print_str)

    last_time = time.time()

    train_metrics = metrics(model, train_data_loader, device)
    valid_metrics = metrics(model, valid_data_loader, device)

    print_str = '\n'

    for name, val in train_metrics:
        print_str += f'train_{name}: {val:.3f} '

    for name, val in valid_metrics:
        print_str += f'valid_{name}: {val:.3f} '

    print_str += f'total_time: {(last_time - init_time):.3f} sec'
    logger.info(print_str)

    tb.close()

    np.savez_compressed(log_path + f"losses{comment}.npz",
                        epochs_npz,
                        train_loss_npz,
                        valid_loss_npz)

    valid_accuracy = accuracy(model, valid_data_loader, device)

    import gc
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return valid_accuracy


if __name__ == '__main__':
    args = parse()

    output = pd.DataFrame()

    best_validation_accuracy = 0

    for i in range(20):
        print(
            f"####################################### Sample {i+1} ############################################")
        lr = round(loguniform.rvs(1e-4, 5e-1, size=1)[0], 5)
        batch_size = int(np.random.choice([16, 32, 64, 128, 256, 512], size=1)[0])
        weight_decay = round(loguniform.rvs(1e-4, 1, size=1)[0], 5)
        momentum = round(np.random.uniform(0, 1, size=1)[0], 5)
        gamma = round(loguniform.rvs(1e-5, 5e-1, size=1)[0], 5)
        batch_size = 32

        psize = int(np.random.choice([1, 2], size=1)[0])
        conv_ks = int(np.random.choice([3, 5, 7, 9, 11], size=1)[0])
        ra_m = random.randint(2, 16)

        if batch_size == 512 and psize == 1:
            psize = 2
        if batch_size == 512 and conv_ks > 5:
            conv_ks = int(np.random.choice([3, 5], size=1)[0])

        if batch_size == 256 and psize == 1:
            psize = 2
        if batch_size == 256 and conv_ks > 7:
            conv_ks = int(np.random.choice([3, 5, 7], size=1)[0])
        
        print(
            f"lr = {lr} batch_size = {batch_size} weight_decay = {weight_decay} momentum = {momentum} gamma = {gamma} "+
            f"psize = {psize} conv_ks = {conv_ks} ra_m = {ra_m}")

        args['dataset']['batch_size'] = batch_size
        args['optimizer']['parameters']['lr'] = lr
        args['optimizer']['parameters']['weight_decay'] = weight_decay
        args['optimizer']['parameters']['momentum'] = momentum
        args['scheduler']['scheduler1']['parameters']['gamma'] = gamma

        args['model']['architecture']['conv_ks'] = conv_ks
        args['model']['architecture']['psize'] = psize

        args['dataset']['augmentations']['ra_m'] = ra_m

        validation_accuracy = train(args)

        dictionary = {
            "validation_accuracy": validation_accuracy,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "momentum": momentum,
            "psize": psize,
            "conv_ks": conv_ks,
            "ra_m": ra_m
        }
        df_dictionary = pd.DataFrame([dictionary])
        output = pd.concat([output, df_dictionary], ignore_index=True)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_batch_size = batch_size
            best_lr = lr
            best_weight_decay = weight_decay
            best_gamma = gamma
            best_momentum = momentum

    print(
        f"Best validation accuracy: {best_validation_accuracy} for lr = {best_lr} batch_size = {best_batch_size} "
        f"weight_decay = {best_weight_decay} momentum = {best_momentum} gamma = {gamma} psize = {psize} conv_ks = {conv_ks} ra_m = {ra_m}")
    output.to_csv("result.csv")
