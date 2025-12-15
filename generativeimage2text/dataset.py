import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_pil_image
from PIL import Image

# 모든 이미지에 적용될 기본적인 전처리 작업들
common_transforms = transforms.Compose([
    transforms.Resize((256, 256)), # 모든 이미지를 일정 크기로 조정
    transforms.CenterCrop((224, 224)), # 중앙에서 일정 크기로 크롭
    transforms.ToTensor(), # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 정규화
])

class ICAA17KDataset(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.if_train = if_train
        # 모든 경우에 공통적으로 적용할 전처리 파이프라인 사용
        self.transform = common_transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score_names = ['MOS', 'color']
        y = np.array([row[k] / 10 for k in score_names])

        image_id = row['ID']
        image_path = os.path.join(self.images_path, f'{image_id}') # 파일 이름 형식 확인 필요
        image = default_loader(image_path)

        x = self.transform(image)

        return x, y.astype('float32')


# 기존 
# IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
# IMAGE_NET_STD = [0.229, 0.224, 0.225]
# normalize = transforms.Normalize(
#             mean=IMAGE_NET_MEAN,
#             std=IMAGE_NET_STD)

# class ICAA17KDataset(Dataset):
#     def __init__(self, path_to_csv, images_path, if_train):
#         self.df = pd.read_csv(path_to_csv)
#         self.images_path =  images_path
#         self.if_train = if_train
#         if if_train:
#             self.transform = transforms.Compose([
#                 transforms.Resize((256, 256)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomCrop((224, 224)),
#                 transforms.ToTensor(),
#                 normalize])
#         else:
#             self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             normalize])

#     def __len__(self):
#         return self.df.shape[0]

#     def __getitem__(self, item):
#         row = self.df.iloc[item]
#         score_names = ['MOS', 'color']
#         y = np.array([row[k] / 10 for k in score_names] )

#         image_id = row['ID']
#         image_path = os.path.join(self.images_path, f'{image_id}')
#         image = default_loader(image_path)

#         x = self.transform(image)
#         print("img", x.shape)

#         return x, y.astype('float32')

class TodayhouseDataset(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

        image_id = row['ID']
        image_path = os.path.join(self.images_path, f'{image_id}')
        image = default_loader(image_path)

        x = self.transform(image)

        return x, image_id

