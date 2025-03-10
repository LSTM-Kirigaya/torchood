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
            
        # 标签映射，确保 ID 是连续的，OOD 也是连续的
        if label in id_labels_mapper:
            label = id_labels_mapper[label]
        elif label in ood_labels_mapper:
            label = ood_labels_mapper[label]

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

        # 标签映射，确保 ID 是连续的，OOD 也是连续的
        if label in id_labels_mapper:
            label = id_labels_mapper[label]
        elif label in ood_labels_mapper:
            label = ood_labels_mapper[label]
            
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

        for class_id in class_split[1]:
            self.valid_idx_ood.append(self.label[class_id])

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
            raise RuntimeError('data path does not exist!')
        if not os.path.exists(self.label_dir):
            raise RuntimeError('label path does not exist!')

        if os.path.isfile(self.label_dir):  # 读取csv标签
            csv_reader = pd.read_csv(self.label_dir, header=0, index_col='image')
        else:
            raise RuntimeError('wrong label path is given!')
        print(f'Start loading ISIC from {self.data_dir}')

        # for debug
        if self.args is None:
            debug_mode = False
        else:
            debug_mode =  self.args.debug

        with tqdm(total=len(os.listdir(self.data_dir)), ncols=100) as _tqdm:
            for step, img in enumerate(os.listdir(self.data_dir)):
                p_img = os.path.join(self.data_dir, img)
                if p_img.endswith('jpg'):
                    data = io.imread(p_img)
                    data = cv2.resize(data, (224, 224))
                    
                    id = img.split('.')[0]
                    label = np.array(csv_reader.loc[id])
                    self.data.append(data)
                    self.label.append(label.argmax(axis=0))
                    
                    # if debug_mode and len(self.data) > 200:
                    #     break
                    
                _tqdm.update(1)
        
        from collections import Counter
        print(Counter(self.label))
        
        self.label = np.array(self.label)
        print('Finish loading data!')


import yaml
config = yaml.load(open('./config/isic2019.yaml', 'r', encoding='utf-8'), Loader=yaml.Loader)
p_train_img = config['data']['train_image_dir']
p_train_label = config['data']['train_image_label']
data_dir = config['data']['cache_dir']
data_name = config['data']['name']

id_labels = config['model']['ID_labels']
ood_labels = config['model']['OOD_labels']

# 构建 label 映射（训练加载数据时使用）
id_labels_mapper = {}
ood_labels_mapper = {}
id_counter = 0
for label_id in id_labels:
    id_labels_mapper[label_id] = id_counter
    id_counter += 1
for label_id in ood_labels:
    ood_labels_mapper[label_id] = id_counter
    id_counter += 1

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
