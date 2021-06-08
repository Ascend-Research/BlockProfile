# Using the exact code provided by the Once-For-All tutorial for transforms to ensure consistency
import torch
from torchvision import transforms, datasets
import math
import os
import numpy as np


def get_imagenet_calib_loader(imagenet_path="/data/ImageNet/", batch_size=250, workers=8, size=224, num_images=2000):
    calib_data = datasets.ImageFolder(
        os.path.join(
            imagenet_path,
            'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406],
                std=[
                    0.229,
                    0.224,
                    0.225]
            ),
        ])
    )
    chosen_indexes = np.random.choice(list(range(len(calib_data))), num_images)
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    calib_loader = torch.utils.data.DataLoader(
        calib_data,
        sampler=sub_sampler,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return calib_loader


def get_imagenet_val_loader(imagenet_path="/data/ImageNet/", batch_size=250, workers=8, size=224):

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_path, 'val'),
            transform=_build_val_transform(size)
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    return val_loader


def _build_val_transform(size):
    return transforms.Compose([
        transforms.Resize(int(math.ceil(size / 0.875))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class ImageNetRAMDataset(torch.utils.data.Dataset):

    # Transforms are all performed when the RAM data is made, not here.
    def __init__(self, tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index].long()

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def get_imagenet_RAM_val_loader(batch_size=500, num_workers=6, res=224,
                            imagenet_path='models/ImageNetRAM/'):

    # Replace this directory with the location of your information
    content = torch.load(os.path.join(imagenet_path, 'val/val_data_%d' % res))

    tensor_x = torch.Tensor(content['data'])
    tensor_y = torch.LongTensor(content['labels'])
    print(tensor_x.shape)
    print(tensor_y.shape)

    inet_val_dset = ImageNetRAMDataset((tensor_x, tensor_y))

    val_loader = torch.utils.data.DataLoader(inet_val_dset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=True,
                                             pin_memory=True,
                                             drop_last=False
                                             )

    return val_loader
