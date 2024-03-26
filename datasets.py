# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split



class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        #dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'GCANCER':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 3
    elif args.data_set == 'HER2-4':
        validation_split = 0.15
        ds = datasets.ImageFolder(args.data_path, transform=transform)
        print(ds)
        nb_classes = 4
        # generate indices: instead of the actual data we pass in integers instead
        train_indices, test_indices, _, _ = train_test_split(
            range(len(ds)),
            ds.targets,
            stratify=ds.targets,
            test_size=validation_split,
            random_state=42
        )

        # generate subset based on indices
        train_set = Subset(ds, train_indices)
        val_set = Subset(ds, test_indices)

        if is_train:
            dataset = train_set
        else:
            dataset = val_set
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes

#image = torch.zeros(3, 224, 224)

# Distribute pixel indices across all channels
import numpy as np
channel_indices = None
pixel_indices = None

torch.manual_seed(0)


patch_size = 16
drop_ratio = 0

#def drop_patch(img):
#    global patch_size
#    global drop_ratio
#    #img: tensor of shape (3, 224, 224)
#    #patch_size: size of the patch to drop
#    #drop_ratio: ratio dropped
#    img = img.permute(1, 2, 0).numpy()
#    h, w, _ = img.shape
#    npatches = (h//patch_size) * (w//patch_size)
#    #seed
#    #np.random.seed(abs(int(img[0,0,0])))
#    permutation = np.random.permutation(npatches)[:int(drop_ratio*npatches)]
#    for p in permutation:
#        y = p // (w//patch_size)
#        x = p % (w//patch_size)
#        img[y*patch_size:y*patch_size+patch_size, x*patch_size:x*patch_size+patch_size] = 0
#    return torch.tensor(img).permute(2, 0, 1)
def drop_patch(img):
    global patch_size
    global drop_ratio
    #img: tensor of shape (3, 224, 224)
    #patch_size: size of the patch to drop
    #drop_ratio: ratio dropped
    img = img.permute(1, 2, 0).numpy()
    h, w, _ = img.shape
    npatches = (h//patch_size) * (w//patch_size)
    npatches_x = (w - patch_size)
    npatches_y = (h - patch_size)
    #seed
    np.random.seed(abs(int(img[0,0,0])))
    permutation_x = np.random.permutation(npatches_x)[:int(drop_ratio*npatches)]
    permutation_y = np.random.permutation(npatches_y)[:int(drop_ratio*npatches)]
    for py,px in zip(permutation_y, permutation_x):
        y = py
        x = px
        img[y:y+patch_size, x:x+patch_size] = 0
    return torch.tensor(img).permute(2, 0, 1)

def build_transform(is_train, args):
    global channel_indices
    global pixel_indices
    global patch_size
    global drop_ratio
    resize_im = args.input_size > 32
    if is_train:
        if args.classic == 'no':
            noaug = True
        else:
            noaug = False
        if args.aa == 'none':
            args.aa = None
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            no_aug=noaug,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)

        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    if (not is_train) and (args.eval_drop_ratio > 0):
        if args.eval_drop_mode == 'pixel':
            patch_size = 1
        elif args.eval_drop_mode == 'patch':
            patch_size = 16
        else:
            patch_size = 0
        drop_ratio = args.eval_drop_ratio
        t.append(drop_patch)
    return transforms.Compose(t)
