import os
from .dataset import (FaceDataset, RandomCutout, Identity,  RoundRobinDataset)
from torch.utils.data import  DataLoader
import torchvision.transforms as transforms
import torch 
import numpy as np
import random
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
    
def get_single_dataset(data_dir, 
                       FaceDataset, 
                       data_name, 
                       train=True,
                       label=None, 
                       img_size=256, 
                       map_size=32, 
                       transform=None, 
                       UUID=-1,
                       test_per_video=1):
    if train:
        if data_name in ["OULU"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'oulu'), is_train=True, label=label,
                                      transform=transform, UUID=UUID,img_size=img_size )
        elif data_name in ["CASIA_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'casia'), is_train=True, label=label,
                                      transform=transform, UUID=UUID,img_size=img_size )
        elif data_name in ["Replay_attack"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'replay'), is_train=True, label=label,
                                      transform=transform,  UUID=UUID,img_size=img_size )
        elif data_name in ["MSU_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'msu'), is_train=True, label=label,
                                      transform=transform,  UUID=UUID,img_size=img_size )

    else:
        if data_name in ["OULU"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'oulu'), is_train=False, label=label,
                                      transform=transform, map_size=map_size, UUID=UUID, img_size=img_size, 
                                       test_per_video=test_per_video)
        elif data_name in ["CASIA_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'casia'), is_train=False, label=label,
                                      transform=transform, map_size=map_size, UUID=UUID,img_size=img_size, 
                                       test_per_video=test_per_video)
        elif data_name in ["Replay_attack"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'replay'), is_train=False, label=label,
                                      transform=transform, map_size=map_size, UUID=UUID,img_size=img_size, 
                                       test_per_video=test_per_video)
        elif data_name in ["MSU_MFSD"]:
            
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'msu'), is_train=False, label=label,
                                      transform=transform, map_size=map_size, UUID=UUID, img_size=img_size, 
                                       test_per_video=test_per_video)

    return data_set

def get_datasets(data_dir, 
                 FaceDataset, 
                 data_list, 
                 train=True, 
                 img_size=256, 
                 map_size=32, 
                 transform=None, 
                 test_per_video=1,
                 balance=False):


    sum_n = 0
    if train:
        if not balance:
            data_name_list_train = data_list
            data_set_sum = get_single_dataset(data_dir, 
                                            FaceDataset, 
                                            data_name=data_name_list_train[0], 
                                            train=True, 
                                            img_size=img_size, 
                                            map_size=map_size, 
                                            transform=transform, 
                                            UUID=0 )
            sum_n = len(data_set_sum)
            for i in range(1, len(data_name_list_train)):
                data_tmp = get_single_dataset(data_dir, FaceDataset, 
                                            data_name=data_name_list_train[i], 
                                            train=True, img_size=img_size, 
                                            map_size=map_size, transform=transform, 
                                            UUID=i)
                data_set_sum += data_tmp
                sum_n += len(data_tmp)
        else:
             print("Balanced loader for each class and domain")
             data_name_list_train = data_list
             data_set_list = []
             sum_n = 0
             for i in range(len(data_name_list_train)):
                for label in ['live', 'spoof']:
                    data_tmp = get_single_dataset(data_dir, FaceDataset, 
                                                data_name=data_name_list_train[i], 
                                                train=True, img_size=img_size, 
                                                map_size=map_size, transform=transform, 
                                                UUID=i, label=label)
                    data_set_list.append(data_tmp)
                    sum_n += len(data_tmp)
             data_set_sum = RoundRobinDataset(data_set_list)
    else:
        data_name_list_test = data_list
        data_set_sum = {}
        for i in range(len(data_name_list_test)):
            data_tmp = get_single_dataset(data_dir, 
                                          FaceDataset, 
                                          data_name=data_name_list_test[i], 
                                          train=False, 
                                          img_size=img_size, 
                                          map_size=map_size, 
                                          transform=transform, 
                                          UUID=i,
                                          test_per_video=test_per_video)
            data_set_sum[data_name_list_test[i]] = data_tmp
            sum_n += len(data_tmp)
    print("{} videos: {}".format('Train' if train else 'Test',sum_n))
    return data_set_sum

def get_train_test_loader(config):
    
    """
    Return the train and test data loader
    
    """
    if  config.TRAIN.get('auto_augment'): 
        print("Apply AUTO AUGMENTATION")
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    train_transform =  transforms.Compose([
            transforms.RandomResizedCrop((config.MODEL.image_size, config.MODEL.image_size), 
                                         scale=(0.08, 1.0), 
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-180, 180), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            RandomCutout(1, 0.5) if config.TRAIN.cutout else Identity(),
            normalization
        ])

    test_transform = transforms.Compose([
            transforms.Resize((config.MODEL.image_size, config.MODEL.image_size)),
            transforms.ToTensor(),
            normalization
        ])
    train_set = get_datasets(config.PATH.data_folder, FaceDataset, 
                             train=True, data_list=config.train_set, 
                             img_size= config.MODEL.image_size, map_size=32, 
                             transform=train_transform, balance=config.TRAIN.get('balance_loader'))
    
    test_set = get_datasets(config.PATH.data_folder, FaceDataset, 
                            train=False, data_list=config.test_set, 
                            img_size= config.MODEL.image_size, map_size=32, 
                            transform=test_transform,
                            test_per_video=config.TEST.get('n_frames') if config.TEST.get('n_frames') else 1)
    
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    train_loader = DataLoader(train_set, batch_size=config.TRAIN.batch_size, 
                              shuffle=True, num_workers=config.SYS.num_workers,
                            worker_init_fn=seed_worker,  generator=g)

    test_loader = DataLoader(test_set[config.test_set[0]], batch_size=config.TRAIN.batch_size, 
                             shuffle=False, num_workers=config.SYS.num_workers,
                            worker_init_fn=seed_worker, generator=g)

    return train_loader,test_loader