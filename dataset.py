from typing import Any
import torch
import torchvision
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

class MLDataSet(Dataset):
    def __init__(self, args, config, mode: str='train') -> None:
        super().__init__()
        
        image_size = config['train']['image_size']
        print(image_size)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        self.data_path = Path(args.data)
        
        if mode == 'train':
            self.data_path = self.data_path / '3-Saliency-TrainSet'
        else:
            self.data_path = self.data_path / '3-Saliency-TestSet'

        img_dir = self.data_path / 'Stimuli'
        label_dir = self.data_path / 'FIXATIONMAPS'
        
        self.cls_idx, self.img_paths, self.cls = self.load_imgs(img_dir)
        cls_idx, self.label_paths, _ = self.load_imgs(label_dir)
        
        # 顺序一致性
        assert cls_idx == self.cls_idx
        assert len(self.img_paths) == len(self.label_paths)
        
        print(f'>total data: {len(self.img_paths)}')
        
        self.N = len(self.img_paths)
        
        # pre load imgs
        # self.imgs = [cv2.imread(str(img)) for img in tqdm(self.img_paths, desc='loding_imgs:', total=self.N)]
        # self.labels = [cv2.imread(str(label)) for label in tqdm(self.label_paths, desc='loding_labels', total=self.N)]

    def __getitem__(self, index):
        raw_img = cv2.imread(str(self.img_paths[index]))
        raw_label = cv2.imread(str(self.label_paths[index]), flags=cv2.IMREAD_GRAYSCALE)
        
        
        img = self.transform(Image.fromarray(raw_img))
        label = self.transform_mask(Image.fromarray(raw_label))
        # label = label.squeeze()
        
        return {
            'image': img.float(),
            'label': label.float(), 
            'cls': self.cls[index]
        }
    
    def __len__(self, ):
        return self.N
    
    def load_imgs(self, paths):
        img_paths = []
        img_cls = []
        cls_idxs = dict()
        cnt_idx = -1
        
        for dir_path in paths.iterdir():
            if dir_path.is_dir():
                cls_idxs[dir_path.name] = []
                for img_path in dir_path.iterdir():
                    if img_path.is_file():
                        cnt_idx = cnt_idx + 1
                        img_paths.append(img_path)
                        cls_idxs[dir_path.name].append(cnt_idx)
                        img_cls.append(dir_path.name)

        return cls_idxs, img_paths, img_cls
                
    
    