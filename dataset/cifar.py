from torchvision import transforms
from torchvision.datasets import CIFAR10 as cifar10
from torch.utils.data import random_split


def CIFAR10(params):
    mean_val = [0.4914, 0.4822, 0.4465]
    std_val = [0.2470, 0.2435, 0.2616]
    save_path = './data'

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(
            params['augmentations']['scale'], 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=params['augmentations']['ra_n'], magnitude=params['augmentations']['ra_m']),
        transforms.ColorJitter(params['augmentations']['jitter'],
                               params['augmentations']['jitter'],
                               params['augmentations']['jitter']),
        transforms.ToTensor(),
        transforms.Normalize(mean_val, std_val),
        transforms.RandomErasing(p=params['augmentations']['reprob'])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_val, std_val)
    ])

    train_dataset = cifar10(save_path,
                            train=True,
                            download=True,
                            transform=train_transform)

    test_dataset = cifar10(save_path,
                           train=False,
                           download=True,
                           transform=test_transform)

    train_size = int(params['split']['train'] * len(train_dataset))
    valid_size = int(params['split']['valid'] * len(train_dataset))
    train_other_size = len(train_dataset) - train_size - valid_size

    train_datasets = random_split(train_dataset, [train_size,
                                                  valid_size,
                                                  train_other_size])

    test_size = int(params['split']['test'] * len(test_dataset))
    test_other_size = len(test_dataset) - test_size

    test_datasets = random_split(test_dataset, [test_size,
                                                test_other_size])

    train_dataset, valid_dataset, _ = train_datasets
    test_dataset, _ = test_datasets

    return train_dataset, valid_dataset, test_dataset
