import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_pil_image
from PIL import Image

# 기존 
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class ICAA17KDataset(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path =  images_path
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
        score_names = ['MOS', 'color']
        y = np.array([row[k] / 10 for k in score_names] )

        image_id = row['ID']
        image_path = os.path.join(self.images_path, f'{image_id}')
        image = default_loader(image_path)

        x = self.transform(image)
        # print("img", x.shape)
        # return x, y.astype('float32') # 기존 
        return x, image_id, y.astype('float32')

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

        return x , image_id
    
class CustomImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        # image_path는 단일 이미지 파일 경로를 저장합니다.
        self.image_path = image_path
        self.transform = transform
    
    def __len__(self):
        # 단일 이미지이므로 길이는 1입니다.
        return 1
    
    def __getitem__(self, idx):
        # image_path에서 이미지를 불러옵니다.
        image = Image.open(self.image_path).convert('RGB')
        original_size = image.size
        if self.transform:
            image = self.transform(image)
        # 파일명에서 확장자를 제외한 부분을 image_id로 사용합니다.
        image_id = os.path.splitext(os.path.basename(self.image_path))[0]
        return image, image_id, original_size
