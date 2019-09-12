import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import pandas as pd
import numpy as np
from PIL import Image
import os
import random

disease_categories = {
        'Atelectasis': 0,
        'Cardiomegaly': 1,
        'Effusion': 2,
        'Infiltrate': 3,
        'Mass': 4,
        'Nodule': 5,
        'Pneumonia': 6,
        'Pneumothorax': 7,
        'Consolidation': 8,
        'Edema': 9,
        'Emphysema': 10,
        'Fibrosis': 11,
        'Pleural_Thickening': 12,
        'Hernia': 13,
        }

class CXRDataset(Dataset):

    # FIXME: transform can't be None, since the return type must be a Tensor (not Pillow.Image)
    # can use: torchvision.transforms.ToTensor() by default
    def __init__(self, root_dir, dataset_type = 'train', transform = None, n_diseases=8):
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError("No such type, must be 'train', 'val', or 'test'")
        
        self.image_dir = os.path.join(root_dir, 'images')
        self.transform = transform

        # Load csv files
        labels_fname = os.path.join(root_dir, dataset_type + '_label.csv')
        self.label_index = pd.read_csv(labels_fname, header=0)
        self.bbox_index = pd.read_csv(os.path.join(root_dir, 'BBox_List_2017.csv'), header=0)
        
        # Drop Bbox file unnamed columns (are empty)
        drop_unnamed = [col for col in self.bbox_index.columns if col.startswith("Unnamed")]
        self.bbox_index.drop(drop_unnamed, axis=1, inplace=True)
        
        # Get classes names
        self.classes = list(self.label_index.columns)[1:1+n_diseases]
        self.n_diseases = n_diseases

        # Keep only the images in the directory
        available_images = os.listdir(self.image_dir)
        self.label_index = self.label_index.loc[self.label_index['FileName'].isin(available_images)]
        self.bbox_index = self.bbox_index.loc[self.bbox_index['Image Index'].isin(available_images)]
        
    def size(self):
        return self.label_index.shape

    def __len__(self):
        n_samples, _ = self.label_index.shape
        return n_samples

    def __getitem__(self, idx):
        row = self.label_index.iloc[idx]
        
        # Load image
        image_name = row[0]        
        image_fname = os.path.join(self.image_dir, image_name)
        image = Image.open(image_fname).convert('L')
        if self.transform:
            image = self.transform(image)

        # Extract labels
        labels = row[1:1+self.n_diseases].to_numpy().astype('int')
        
        # Array indicating if the bbox is valid for each disease
        bbox_valid = np.zeros(self.n_diseases)
        for i in range(self.n_diseases):
            if labels[i] == 0:
                   bbox_valid[i] = 1
        
        # Save bboxes
        bbox = np.zeros([self.n_diseases, 512, 512])
        rows = self.bbox_index.loc[self.bbox_index['Image Index']==image_name]
        for index, row in rows.iterrows():
            image_name, disease_name, x, y, w, h = row
            y = int(y)
            h = int(h)

            x = int(x)
            w = int(w)

            disease_index = disease_categories[disease_name]
            
            bbox[disease_index, y:y+h, x:x+w] = 1
            bbox_valid[disease_index] = 1
        
        return image, labels, image_name, bbox, bbox_valid
    
    
class CXRDataset_BBox_only(Dataset):
    # TODO: reuse code with previous class

    def __init__(self, root_dir, transform = None, data_arg=True): 
        self.image_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.index_dir = os.path.join(root_dir, 'test'+'_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None,nrows=1).ix[0, :].to_numpy()[1:9]
        self.label_index = pd.read_csv(self.index_dir, header=0)
        self.bbox_index = pd.read_csv(os.path.join(root_dir, 'BBox_List_2017.csv'), header=0)
        self.data_arg = data_arg
        
        # Keep only the images in the directory
        available_images = os.listdir(self.image_dir)
        self.label_index = self.label_index.loc[self.label_index['FileName'].isin(available_images)]
        self.bbox_index = self.bbox_index.loc[self.bbox_index['Image Index'].isin(available_images)]
        
    def __len__(self):
        return len(self.bbox_index)

    def __getitem__(self, idx):
        name = self.bbox_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('L')
        label = self.label_index.loc[self.label_index['FileName']==name].iloc[0, 1:9].to_numpy().astype('int')
            
        
        # bbox
        bbox = np.zeros([8, 512, 512])
        bbox_valid = np.zeros(8)
        for i in range(8):
            if label[i] == 0:
               bbox_valid[i] = 1
        
        cols = self.bbox_index.loc[self.bbox_index['Image Index']==name]
        if len(cols)>0:
            for i in range(len(cols)):
                bbox[
                    disease_categories[cols.iloc[i, 1]], #index
                    int(cols.iloc[i, 3]/2): int(cols.iloc[i, 3]/2+cols.iloc[i, 5]/2), #y:y+h
                    int(cols.iloc[i, 2]/2): int(cols.iloc[i, 2]/2+cols.iloc[i, 4]/2) #x:x+w
                ] = 1
                bbox_valid[disease_categories[cols.iloc[i, 1]]] = 1
        
        #argumentation
        if self.data_arg:
            image = F.resize(image, 600, Image.BILINEAR)
            angle = random.uniform(-20, 20)
            image = F.rotate(image, angle, resample=False, expand=False, center=None)
            crop_i = random.randint(0, 600 - 512)
            crop_j = random.randint(0, 600 - 512)
            image = F.crop(image, crop_i, crop_j, 512, 512)
            bbox_list = []
            for i in range(8):
                bbox_img = Image.fromarray(bbox[i])
                bbox_img = F.resize(bbox_img, 600, Image.BILINEAR)
                bbox_img = F.rotate(bbox_img, angle, resample=False, expand=False, center=None)
                bbox_img = F.crop(bbox_img, crop_i, crop_j, 512, 512)
                bbox_img = transforms.ToTensor()(bbox_img)
                bbox_list.append(bbox_img)
            bbox = torch.cat(bbox_list)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, name, bbox, bbox_valid
