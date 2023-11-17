import os 
import io
import torch
from torch.utils.data import Dataset
import math
from glob import glob
import re
from .meta import DEVICE_INFOS
import numpy as np
from PIL import Image
import random

def list_dirs_at_depth(root_dir, depth):
    if depth < 0:
        return []
    elif depth == 0:
        return [root_dir]
    else:
        sub_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        return [d for sub_dir in sub_dirs for d in list_dirs_at_depth(sub_dir, depth-1)] 
    
class FaceDataset(Dataset):    
    def __init__(self, 
                 dataset_name, 
                 root_dir, 
                 is_train=True, 
                 label=None, 
                 transform=None, 
                 map_size=32, 
                 UUID=-1,
                 img_size=256,
                 test_per_video=1):
        self.is_train = is_train
        self.video_list = [folder for folder in list_dirs_at_depth(os.path.join(root_dir, 'train' if is_train else 'test'), 2) if len(os.listdir(folder)) > 0] 
        if label is not None and label != 'all':
            self.video_list = list(filter(lambda x: label in x, self.video_list))
        print(f"({root_dir.split('/')[-1]}) Total video: {len(self.video_list)}: {len([u for u in self.video_list  if 'live' in u])} vs. {len([u for u in self.video_list  if 'live' not in u])}" )
            
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transform
        self.map_size = map_size
        self.UUID = UUID
        self.image_size = img_size

        if not is_train:
            self.frame_per_video = test_per_video
            self.video_list = sum([self.video_list]*test_per_video, [])
        else:
            self.frame_per_video = 1

        self.init_frame_list()
    def __len__(self):
        return len(self.video_list)
    
    def shuffle(self):
        if self.is_train:
            random.shuffle(self.video_list)
        
    def init_frame_list(self):
        """
        Create dictionary of 
        """
        self.video_frame_list = dict(zip([os.path.join(self.root_dir, video_name) for video_name in self.video_list],
                                     [[] for _ in self.video_list]))
        for  video_path in self.video_frame_list:

            if not self.is_train:
                """
                In the mode test, we only need on face per video
                """
                all_crop_faces = glob(os.path.join(video_path, "crop_*.jpg"))
                assert len(all_crop_faces) > 2, f"Cannot find the image in folder {video_path}"
                # all_crop_faces.sort()
                self.video_frame_list[video_path]  =  all_crop_faces # [len(all_crop_faces)//2:len(all_crop_faces)//2+1] # Select only one middle frame for reproducible
            else:
                all_crop_faces = glob(os.path.join(video_path, "crop_*.jpg"))
                assert len(all_crop_faces) > 2, f"Cannot find the image in folder {video_path}"
                self.video_frame_list[video_path]  = all_crop_faces
            
        return True
    
    def get_client_from_video_name(self, video_name):
        video_name = video_name.split('/')[-1]
        if 'msu' in self.dataset_name.lower() or 'replay' in self.dataset_name.lower():
            match = re.findall('client(\d\d\d)', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'oulu' in self.dataset_name.lower():
            match = re.findall('(\d+)_\d$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'casia' in self.dataset_name.lower():
            
            match = re.findall('(\d+)_[H|N][R|M]_\d$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                print(f"Cannot find client from : {video_name}")
                raise RuntimeError('no client')
        else:
            raise RuntimeError("no dataset found")
        return client_id
    
    def __getitem__(self, idx):
        idx = idx % len(self.video_list) # Incase testing with many frame per video
        video_name = self.video_list[idx]
        spoofing_label = int('live' in video_name)
        if self.dataset_name in DEVICE_INFOS:
            if 'live' in video_name:
                patterns = DEVICE_INFOS[self.dataset_name]['live']
            elif 'spoof' in video_name:
                patterns = DEVICE_INFOS[self.dataset_name]['spoof']
            else:
                raise RuntimeError(f"Cannot find the label infor from the video: {video_name}")
            device_tag = None
            for pattern in patterns:
                if len(re.findall(pattern, video_name)) > 0:
                    if device_tag is not None:
                        raise RuntimeError("Multiple Match")
                    device_tag = pattern
            if device_tag is None:
                raise RuntimeError("No Match")
        else:
            device_tag = 'live' if spoofing_label else 'spoof'

        client_id = self.get_client_from_video_name(video_name)

        image_dir = os.path.join(self.root_dir, video_name)

        if self.is_train:
            image_x, _, _, = self.sample_image(image_dir, is_train=True)
            transformed_image1 = self.transform(image_x)           
            transformed_image2 = self.transform(image_x, )


        else:
            image_x, _, _ = self.sample_image(image_dir, is_train=False, rep=None)
            transformed_image1 = transformed_image2 = self.transform(image_x)



        sample = {"image_x_v1": transformed_image1,
                  "image_x_v2": transformed_image2,
                  "label": spoofing_label,
                  "UUID": self.UUID,
                  'device_tag': device_tag,
                  'video': video_name,
                  'client_id': client_id}
        return sample


    def sample_image(self, image_dir, is_train=False, rep=None):
        """
        rep is the parameter from the __getitem__ function to reduce randomness of test phase
        
        """
        image_path = np.random.choice(self.video_frame_list[image_dir]) 
        image_id = int(image_path.split('/')[-1].split('_')[-1].split('.')[0])

        info_name = f"infov1_{image_id:04d}.npy"
        info_path = os.path.join(image_dir, info_name)

        try: 
            info = None
            image = Image.open(image_path)
        except:
            if is_train: 
                return self.sample_image(image_dir, is_train)
            else:
                raise ValueError(f"Error in the file {info_path}")
        return image, info, image_id * 5

class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im
    
class RandomCutout(object):
    def __init__(self, n_holes, p=0.5):
        """
        Args:
            n_holes (int): Number of patches to cut out of each image.
            p (int): probability to apply cutout
        """
        self.n_holes = n_holes
        self.p = p

    def rand_bbox(self, W, H, lam):
        """
        Return a random box
        """
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if  np.random.rand(1) > self.p:
            return img
        
        h = img.size(1)
        w = img.size(2)
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(w, h, lam)
        for n in range(self.n_holes):
            img[:,bby1:bby2, bbx1:bbx2] = img[:,bby1:bby2, bbx1:bbx2].mean(dim=[-2,-1],keepdim=True)
        return img
    
class RandomJPEGCompression(object):
    def __init__(self, quality_min=30, quality_max=90, p=0.5):
        assert 0 <= quality_min <= 100 and 0 <= quality_max <= 100
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p
    def __call__(self, img):
        if  np.random.rand(1) > self.p:
            return img
        # Choose a random quality for JPEG compression
        quality = np.random.randint(self.quality_min, self.quality_max)
        
        # Save the image to a bytes buffer using JPEG format
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        
        # Reload the image from the buffer
        img = Image.open(buffer)
        return img
    
class RoundRobinDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_len = sum(self.lengths)
        
    def __getitem__(self, index):
        # Determine which dataset to sample from
        dataset_id = index % len(self.datasets)
        
        # Adjust index to fit within the chosen dataset's length
        inner_index = index // len(self.datasets)
        inner_index = inner_index % self.lengths[dataset_id]
        return self.datasets[dataset_id][inner_index]
    
    def shuffle(self):
        for dataset in self.datasets:
            dataset.shuffle()

    def __len__(self):
        # Return the length of the largest dataset times the number of datasets
        return max(self.lengths) * len(self.datasets)