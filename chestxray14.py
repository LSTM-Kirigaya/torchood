import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import cv2


label_to_id = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
    "No Finding": -1  # 如果没有发现疾病，标记为 -1
}

class DimTransform:
    def __init__(self, target_dim, class_split):
        self.target_dim = target_dim
        self.class_split = class_split

    def __call__(self, x):
        if np.argmax(x) in self.class_split[0]:
            label = torch.zeros(self.target_dim)
            label[self.class_split[0].index(np.argmax(x))] = 1
        else:
            label = torch.ones(self.target_dim)
            label = -1. / self.target_dim * label
        return label


class Subset(Dataset):
    def __init__(self, data, label, data_split, transform=None, label_transform=None):
        self.data = data
        self.label = label
        self.data_split = data_split
        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
    
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data)


class Base(Dataset):
    def __init__(self, img_dir, label_dir, cache_dir, transform, aug_transform, label_transform):
        self.data_dir = img_dir
        self.label_dir = label_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.aug_transform = aug_transform
        self.label_transform = label_transform

        self.data = []
        self.label = []
        self.load_data()

        self.idx_by_class = []
        self.train_idx = []
        self.valid_idx_id = []
        self.valid_idx_ood = []

    def load_data(self):
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        data = self.data[index]
        label = self.label[index]
    
        if self.transform is not None:
            data = self.transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)
            
        return data, label

    def __len__(self) -> int:
        return len(self.data)

    def split(self, fold: int,
              class_split: tuple,
              use_ood_during_training: bool = False,
              data_transform: torchvision.transforms = None,
              aug_transform: torchvision.transforms = None,
              split_label_transform: torchvision.transforms = None) -> tuple[Subset, Subset, Subset]:
        r"""Split ISIC 2019 dataset
        
        Args:
            fold(int): fold
            use_ood_during_training(bool): if true, a little ood data is mixed into output train_set
        
        """
        assert len(class_split) == 2, f'Wrong split setting is given! expect 2, given {len(class_split)}.'

        if data_transform is None:
            data_transform = self.transform
        if aug_transform is None:
            aug_transform = self.aug_transform
        if split_label_transform is None:
            split_label_transform = DimTransform(len(class_split[0]), class_split=class_split)

        # split data according to categories
        max_label = max(self.label)
        self.idx_by_class = [ [] for _ in range(max_label + 1) ]
        for idx, l in enumerate(self.label):
            self.idx_by_class[l].append(idx)
        
        for class_id in class_split[1]:
            self.valid_idx_ood += self.idx_by_class[class_id]

        idx_id = np.setdiff1d(np.arange(len(self.data)), self.valid_idx_ood)
        self.valid_idx_id = idx_id[np.linspace(fold, len(idx_id), len(idx_id) // 5, endpoint=False, dtype=np.int)]
        self.train_idx = np.setdiff1d(idx_id, self.valid_idx_id)
        
        # 训练集
        train_set = Subset(
            [self.data[idx] for idx in self.train_idx],
            [self.label[idx] for idx in self.train_idx],
            data_split=class_split,
            transform=aug_transform,
            label_transform=split_label_transform
        )
        
        # 测试集
        valid_set_id = Subset(
            [self.data[idx] for idx in self.valid_idx_id],
            [self.label[idx] for idx in self.valid_idx_id],
            data_split=class_split,
            transform=data_transform,
            label_transform=split_label_transform
        )
                
        # OOD 数据
        if use_ood_during_training:
            
            # 分配 20% 的数据作为 training ood
            ood_sample_num = len(self.valid_idx_ood)
            valid_ood_sample = int(ood_sample_num * 0.2)
            np.random.shuffle(self.valid_idx_ood)
            
            valid_set_ood = Subset(
                [self.data[idx] for idx in self.valid_idx_ood[:valid_ood_sample]],
                [self.label[idx] for idx in self.valid_idx_ood[:valid_ood_sample]],
                data_split=class_split,
                transform=data_transform,
                label_transform=split_label_transform
            )
            
            # 将 ood 数据混入 train set            
            train_set.data.extend([self.data[idx] for idx in self.valid_idx_ood[valid_ood_sample:]])
            train_set.label.extend([self.label[idx] for idx in self.valid_idx_ood[valid_ood_sample:]])
            
            return train_set, valid_set_id, valid_set_ood
        else: 
            valid_set_ood = Subset([self.data[idx] for idx in self.valid_idx_ood],
                                [self.label[idx] for idx in self.valid_idx_ood],
                                data_split=class_split,
                                transform=data_transform,
                                label_transform=split_label_transform)

            return train_set, valid_set_id, valid_set_ood


class ISIC(Base):
    def __init__(self, img_dir, label_dir, cache_dir, transform=None, aug_transform=None, label_transform=None, args=None):
        self.args = args
        
        super(ISIC, self).__init__(img_dir, label_dir, cache_dir, transform, aug_transform, label_transform)
        # if transform is None:
        #     self.transform = transforms.Compose([
        #         transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #     ])
        # if aug_transform is None:
        #     self.aug_transform = transforms.Compose([
        #         transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
        #         transforms.RandomCrop((224, 224), padding=4),
        #         transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #     ])
                
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        if aug_transform is None:
            self.aug_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop((224, 224), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
    def load_data(self):
        data_npy = os.path.join(self.cache_dir, prebuild_data_file)
        label_npy = os.path.join(self.cache_dir, prebuild_label_file)
        
        if os.path.exists(data_npy) and os.path.exists(label_npy):
            self.data = np.load(data_npy)
            self.label = np.load(label_npy)
            print('reuse processed data in', self.cache_dir)
            return
        
        if not os.path.exists(self.data_dir):
            raise RuntimeError('data path does not exist! ' + self.data_dir)
        if not os.path.exists(self.label_dir):
            raise RuntimeError('label path does not exist! ' + self.label_dir)

        
        # 读取 csv 并建立索引
        label_csv = pd.read_csv(self.label_dir, header=0, index_col='Image Index')
        def get_label_ids(image_name):
            if image_name in label_csv.index:
                labels = label_csv.loc[image_name, 'Finding Labels']
                label_list = labels.split('|')
                # 将标签名称转换为 ID
                label_ids = [label_to_id[label] for label in label_list if label in label_to_id]
                return label_ids[0]
            else:
                return None

        image_paths = []
        # data_dir is something like /data/zhelonghuang/datasets/chestxray-14/images
        for image in tqdm(os.listdir(self.data_dir), ncols=100, colour='green'):
            if image.endswith('.png'):
                label = get_label_ids(image)
                if label:
                    image_paths.append(
                        (os.path.join(self.data_dir, image), int(label))
                    )

        print("get image pairs num: {}".format(len(image_paths)))

        with tqdm(total=len(image_paths), ncols=100, colour='green') as _tqdm:
            for step, (p_img, label) in enumerate(image_paths):
                data = io.imread(p_img)
                data = cv2.resize(data, (224, 224))
                
                self.data.append(data)
                self.label.append(label)
                    
                _tqdm.update(1)
        
        from collections import Counter
        print(Counter(self.label))
        
        self.label = np.array(self.label)
        print('Finish loading data!')


import yaml
config = yaml.load(open('./config/chestxray14.yaml', 'r', encoding='utf-8'), Loader=yaml.Loader)
p_train_img = config['data']['train_image_dir']
p_train_label = config['data']['train_image_label']
data_dir = config['data']['cache_dir']
data_name = config['data']['name']

prebuild_data_file = data_name + '.data.npy'
prebuild_label_file = data_name + '.label.npy'


if __name__ == '__main__':    
    train_npy = os.path.join(data_dir, prebuild_data_file)
    label_npy = os.path.join(data_dir, prebuild_label_file)
    
    os.makedirs(data_dir, exist_ok=True)
    isic = ISIC(img_dir=p_train_img, label_dir=p_train_label, cache_dir=data_dir, args=None)
    np.save(train_npy, isic.data)
    print('save data to', train_npy)
    
    np.save(label_npy, isic.label)
    print('save label to', label_npy) 
