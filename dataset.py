import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import pandas as pd
import numpy as np
from PIL import Image
from filelock import FileLock
import os
import random

import utils


class CXRDataset(Dataset):

    def __init__(self, root_dir, dataset_type='train', image_format="L", transform=None, diseases=None, max_images=None):
        """Create a Dataset object."""
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError("No such type, must be 'train', 'val', or 'test'")
        
        self.dataset_type = dataset_type
        
        self.image_dir = os.path.join(root_dir, 'images')

        self.image_format = image_format
        
        self.transform = transform if transform is not None else transforms.ToTensor()

        # Load csv files
        labels_fname = os.path.join(root_dir, dataset_type + '_label.csv')
        self.label_index = pd.read_csv(labels_fname, header=0)
        
        bbox_fname = os.path.join(root_dir, 'BBox_List_2017.csv')
        self.bbox_index = pd.read_csv(bbox_fname, header=0)
        
        # Drop Bbox file unnamed columns (are empty)
        drop_unnamed = [col for col in self.bbox_index.columns if col.startswith("Unnamed")]
        self.bbox_index.drop(drop_unnamed, axis=1, inplace=True)
        
        # Choose diseases names
        if not diseases:
            # TODO: rename to diseases (is more clear, they are not actually classes)
            self.classes = list(utils.ALL_DISEASES)
        else:
            # Keep only the valid ones
            all_diseases_set = set(utils.ALL_DISEASES)
            diseases_set = set(diseases)
            self.classes = list(diseases_set.intersection(all_diseases_set))
            
            not_found_diseases = list(diseases_set - all_diseases_set)
            if not_found_diseases:
                print("Diseases not found: ", not_found_diseases, "(ignoring)")
            

        self.n_diseases = len(self.classes)
        
        # Filter labels DataFrame
        columns = ["FileName"] + self.classes
        self.label_index = self.label_index[columns]

        # Keep only the images in the directory # and max_images
        available_images = set(os.listdir(self.image_dir))
        labeled_images = set(self.label_index['FileName']).intersection(available_images)
        if max_images:
            labeled_images = set(list(labeled_images)[:max_images])

        self.label_index = self.label_index.loc[self.label_index['FileName'].isin(labeled_images)]
        
        # The bbox_index is always kept full (all images)
        self.bbox_index = self.bbox_index.loc[self.bbox_index['Image Index'].isin(available_images)]
        
        # Precompute items
        self.precomputed = None
        self.precompute()
        
    def size(self):
        n_images, _ = self.label_index.shape
        return (n_images, self.n_diseases)

    def get_by_name(self, image_name):
        idx = self.names_to_idx[image_name]

        return self[idx]
        
    
    def __len__(self):
        n_samples, _ = self.label_index.shape
        return n_samples

    def __getitem__(self, idx):
        image_name, labels, bboxes, bbox_valid = self.precomputed[idx]
        
        # Load the image with a lock
        image_fname = os.path.join(self.image_dir, image_name)
        try:
            # REVIEW: Is lock necessary for read operation?
            with FileLock(image_fname + ".lock"):
                image = Image.open(image_fname).convert(self.image_format)
        except OSError as e:
            print(e)
            print("({}) Failed to load image, may be broken: ".format(self.dataset_type, image_fname))

            # FIXME: a way to ignore the image during training? (though it may broke other things)
            raise

        if self.transform:
            image = self.transform(image)
        
        return image, labels, image_name, bboxes, bbox_valid
    
    def precompute(self):
        self.precomputed = []
        self.names_to_idx = dict()
        for idx in range(len(self)):
            item = self.precompute_item(idx)
            image_name = item[0]

            self.precomputed.append(item)
            
            self.names_to_idx[image_name] = idx


    def precompute_item(self, idx):
        row = self.label_index.iloc[idx]
        
        # Image name
        image_name = row[0]

        # Extract labels
        labels = row[self.classes].to_numpy().astype('int')
        
        # Get bboxes
        bboxes = torch.zeros(self.n_diseases, 4) # 4: x, y, w, h
        bbox_valid = torch.zeros(self.n_diseases)

        rows = self.bbox_index.loc[self.bbox_index['Image Index']==image_name]
        for _, row in rows.iterrows():
            _, disease_name, x, y, w, h = row
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            
            # HACK: this should be fixed in the BBox csv file
            if disease_name == "Infiltrate":
                disease_name = "Infiltration"

            if disease_name not in self.classes:
                continue

            disease_index = self.classes.index(disease_name)
            for j, value in enumerate([x, y, w, h]):
                bboxes[disease_index, j] = value

            bbox_valid[disease_index] = 1
        
        return image_name, labels, bboxes, bbox_valid
    
    def get_items_old(self, idx):
        # TODO: delete this (is deprecated)
        return None

        row = self.label_index.iloc[idx]
        
        # Load image
        image_name = row[0]        
        image_fname = os.path.join(self.image_dir, image_name)
        image = Image.open(image_fname).convert(self.image_format)
        if self.transform:
            image = self.transform(image)

        # Extract labels
        labels = row[self.classes].to_numpy().astype('int')
        
        # Get bboxes
        bboxes = torch.zeros(self.n_diseases, 4) # 4: x, y, w, h
        bbox_valid = torch.zeros(self.n_diseases)

        rows = self.bbox_index.loc[self.bbox_index['Image Index']==image_name]
        for _, row in rows.iterrows():
            _, disease_name, x, y, w, h = row
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            disease_index = utils.DISEASE_INDEX[disease_name]
            for j, value in enumerate([x, y, w, h]):
                bboxes[disease_index, j] = value

            bbox_valid[disease_index] = 1
        
        return image, labels, image_name, bboxes, bbox_valid
        
        # Save bboxes

        # TODO: make this a get_bbox_as_image() function, if necessary
#         bbox = np.zeros([self.n_diseases, 512, 512])
#         rows = self.bbox_index.loc[self.bbox_index['Image Index']==image_name]
#         for index, row in rows.iterrows():
#             image_name, disease_name, x, y, w, h = row
#             y = int(y)
#             h = int(h)

#             x = int(x)
#             w = int(w)

#             disease_index = disease_categories[disease_name]
            
#             bbox[disease_index, y:y+h, x:x+w] = 1
#             bbox_valid[disease_index] = 1
        
        # return image, labels, image_name, bbox, bbox_valid
    
    
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
