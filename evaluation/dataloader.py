import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import json
import numpy as np
import scipy.ndimage
import random
import time

import numpy as np
from scipy.ndimage import sobel
import numpy as np
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    EnsureTyped,
    # AsChannelFirstd,
    Rand3DElasticd,
)
from monai.transforms import  EnsureChannelFirstd as AsChannelFirstd
from scipy.ndimage import binary_dilation, binary_erosion


import warnings
warnings.simplefilter("ignore")

# dataset class
class MetaDataset_Multi(Dataset):
    def __init__(self, dataset_dir, 
                 data_loading_config=None, 
                 transform=None, 
                 resize_image=128, 
                 train_or_val = 'train', 
                 train_or_val_split_rate = 0.1,
                 data_split_seed = 0,
                 DDPM=False,
                ):
        """
        Load dataset of nnUNet format.
        Load multiple datasets at the same time.
        
        Args:
            image_dir (string): Path to the directory of dataset.
            data_loading_config (dict): Configuration of loading dataset including dataset ID, task, input, output, and sampling rate.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert data_loading_config is not None, 'Please input the data_loading_config'
        assert train_or_val=='train' or train_or_val=='val', 'Please input the correct train_or_val variable.'
        print('-'*10,f'Load the Meta dataset of {train_or_val} set.','-'*10)
        
        self.dataset_dir = dataset_dir
        self.data_loading_config = data_loading_config
        self.transform = transform
        self.resize_image = resize_image
        
        self.data_samples = []
        self.task_data_num = []
        self.task_sample_rate = []
        
        self.train_or_val = train_or_val
        self.train_or_val_split_rate = train_or_val_split_rate
        self.data_split_seed = data_split_seed
        
        self.DDPM = DDPM
        if self.DDPM: print("Load data in DDPM's scale. "*3)
        
        for i in data_loading_config:
            data_samples_tmp = self.loaddata_from_config(dataset_dir,i)
            self.data_samples+=data_samples_tmp
            self.task_data_num.append(len(data_samples_tmp))
            self.task_sample_rate.append(i['sample rate'])
    
    def loaddata_from_config(self, dataset_dir,i):
        image_dir = os.path.join(dataset_dir,i['name'],'imagesTr')
#         samples = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and '0000' in f]
        samples = [f for f in os.listdir(image_dir) if '0000' in f]
        samples = ['_'.join(f.split('_')[:-1]) for f in samples]
        samples = sorted(list(set(samples)))
        
        with open(os.path.join(dataset_dir,i['name'],'dataset.json'), 'r', encoding='utf-8') as file:
                config = json.load(file)
        data_samples_tmp = [{'sample': os.path.join(dataset_dir, i['name'], 'imagesTr', k) , 'input':i['input'], 'output':i['output'],\
          'task': i['task'],'task_config':i['task_config'],'sample rate': i['sample rate'], 'weight': i['weight'],
                             'config':config} for k in samples]
        
        # Split the train and val set by order.
        split_point = int(len(data_samples_tmp)-len(data_samples_tmp) * self.train_or_val_split_rate)
        if self.train_or_val=='train':
            data_samples_split = data_samples_tmp[:split_point]
        elif self.train_or_val=='val':
            data_samples_split = data_samples_tmp[split_point:]
            
        print(f"Load {i['name']}\t{i['task']} sample rate: {i['sample rate']}", f"Total length is {len(samples)}\t",
             f"Length after spliting is {len(data_samples_split)}")
        return data_samples_split

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        # Load images for the given input_
        image = self.load_data(sample,'input')

        # Load label
        label = self.load_data(sample, 'output')

        
        # crop
        image,crop_coords = self.crop_image(image)
        label,_ = self.crop_image(label,crop_coords)
        
        # padding
        image,padding_config =  self.pad_image(image, padding_config = None)
        label,_ =  self.pad_image(label, padding_config = padding_config)
        
        # resize
        image = self.resize_image_func(image, new_A=self.resize_image,order = 3)
        if sample['output']=='Seg':
            label = self.resize_image_func(label, new_A=self.resize_image,order = 0)
        else:
            label = self.resize_image_func(label, new_A=self.resize_image,order = 3)
        
        # adjust depth to 128*128*128
        image = self.adjust_image_depth(image)
        label = self.adjust_image_depth(label)
        
        # z-score again
        image = self.z_score_normalization(image)
        if sample['output']!='Seg':
            label = self.z_score_normalization(label)
        if sample['output']=='Seg':
            label = self.set_foreground_background(label,**sample['task_config'])
        
        # Task specific modified
        image, label = self.task_specific_transform( image, label, sample)
        
        # Apply transform (if any)
        if self.transform:
            image, label = self.transform(image, label)
            
        

        return {'image': image, 'label': label, 'name': idx}
    
    def task_specific_transform_pre(self, image, label, sample):
    
        if sample['task']=='SupRes':
            factor = 2 if 'factor' not in  sample['task_config'].keys() else sample['task_config']['factor']
            downsampled_image = image[::factor,::factor,::factor]
            zoom_factor = [o/d for o, d in zip(image.shape, downsampled_image.shape)]
            image = scipy.ndimage.zoom(downsampled_image, zoom_factor, order=0)
            
        elif sample['task']=='Bias':
            image = rand_bias_field(image,**sample['task_config'])
        
        elif sample['task']=='Inp':
            image = img_2_painting(image,**sample['task_config'])
            
        elif sample['task']=='Denoi':
            image = add_noise(image=image,**sample['task_config'])
            
        elif sample['task']=='2D23D':
            image = retain_slice_and_next(image=image,**sample['task_config'])
        

        return image, label
    def task_specific_transform(self, image, label, sample):
        """
        {
            'name':'Dataset601_TopCow_MRA',
            'input':0,
            'output':0,
            'task':'Bias',
            'task_config':{'coeff_range' :(0.3,0.5), 'prob' :1},
            'sample rate':1,
            'weight': 1
        }
        {
            'name':'Dataset601_TopCow_MRA',
            'input':0,
            'output':0,
            'task':['Bias','Denoi'],
            'task_config':[
                {'coeff_range' :(0.3,0.5), 'prob' :1},
                {'noise_type' :"gaussian", 'std':0.25}
            ],
            'sample rate':1,
            'weight': 1
        }
        """

        tasks = sample.get('task')
        task_configs = sample.get('task_config')

        if not isinstance(tasks, list):
            tasks = [tasks]
            task_configs = [task_configs]
 
        else:
            if not isinstance(task_configs, list):
                raise ValueError("")

            if len(tasks) != len(task_configs):
                raise ValueError("")

        for task, config in zip(tasks, task_configs):
            
            if task=='SupRes':
                factor = 2 if 'factor' not in  config.keys() else config['factor']
                downsampled_image = image[::factor,::factor,::factor]
                zoom_factor = [o/d for o, d in zip(image.shape, downsampled_image.shape)]
                image = scipy.ndimage.zoom(downsampled_image, zoom_factor, order=0)
            
            elif task=='Bias':
                image = rand_bias_field(image,**config)

            elif task=='Inp':
                image = img_2_painting(image,**config)

            elif task=='Denoi':
                image = add_noise(image=image,**config)

            elif task=='2D23D':
                image = retain_slice_and_next(image=image,**config)

        return image, label 
    
    def load_data(self,sample, key):
        # Load images for the given input_
        images = []
        label_dir = sample['sample'].replace('imagesTr','labelsTr')
        image_dir = sample['sample']
        config = sample['config']
        mod = sample[key]
        
        if mod == 'Seg':
            tmp = nib.load(os.path.join(f"{label_dir}{config['file_ending']}")).get_fdata()
        else:
            tmp = nib.load(os.path.join(f"{image_dir}_{str(mod).zfill(4)}{config['file_ending']}")).get_fdata()
            tmp = self.z_score_normalization(self.clip_percentiles(tmp))
        return tmp
    
    def resample_to_voxel_size(self, image_path, new_voxel_size=(1, 1, 1)):
        print('Warning! use resample_to_1mm1mm1mm function which is slow!!')
        # Load the image
        img = nib.load(image_path)
        img_data = img.get_fdata()
        original_affine = img.affine

        # Get the current voxel sizes from the affine matrix
        current_voxel_size = np.sqrt(np.sum(original_affine[:3, :3] ** 2, axis=0))

        # Calculate the zoom factors for each dimension
        zoom_factors = current_voxel_size / np.array(new_voxel_size)

        # Resample the image data
        resampled_data = zoom(img_data, zoom=zoom_factors, order=3)  # using cubic interpolation

        # Calculate the new affine matrix
        new_affine = original_affine.copy()
        new_affine[:3, :3] = np.diag(new_voxel_size)

        # Create a new NIfTI image
        resampled_img = nib.Nifti1Image(resampled_data, new_affine)

        return resampled_img
    
    # Data preprocessing
    def clip_percentiles(self, img):
        p5 = np.percentile(img, 0.5)
        p95 = np.percentile(img, 99.5)

        img_clipped = np.clip(img, p5, p95)
        return img_clipped

    def z_score_normalization(self, img):
        mean = np.mean(img)
        std = np.std(img)

        img_normalized = (img - mean) / std
        return img_normalized
    
    def crop_image(self, image, crop_coords = None):
        if crop_coords is None:
            threshold= np.percentile(image, 35)
            
            coords = np.array(np.nonzero((image > threshold).astype('int')))
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            z_min, z_max = coords[2].min(), coords[2].max()
            crop_coords = {'x_min':x_min,'x_max':x_max,'y_min':y_min,'y_max':y_max,'z_min':z_min, 'z_max':z_max}
        else: 
            x_min, x_max = crop_coords['x_min'],crop_coords['x_max']
            y_min, y_max = crop_coords['y_min'],crop_coords['y_max']
            z_min, z_max = crop_coords['z_min'],crop_coords['z_max']

        cropped_image = image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        return cropped_image, crop_coords
    
    def pad_image(self, image, pad_type='constant',padding_config = None):
        '''
        Make dim 0,1 equal
        '''
        pad_value = image.min()
        
        if padding_config is None:
            # Calculate the dimensions to pad to (make A and B dimensions equal)
            A, B, C = image.shape
            max_dim = max(A, B)

            # Calculate padding amounts
            pad_A = (max_dim - A) // 2
            pad_A_remainder = (max_dim - A) % 2
            pad_B = (max_dim - B) // 2
            pad_B_remainder = (max_dim - B) % 2

            # Create a padding configuration
            padding_config = (
                (pad_A, pad_A + pad_A_remainder),  # Padding for A dimension
                (pad_B, pad_B + pad_B_remainder),  # Padding for B dimension
                (0, 0)  # No padding for C dimension
            )
            

        # Pad the image
        padded_image = np.pad(image, padding_config, mode=pad_type, constant_values=pad_value)
        return padded_image, padding_config
    
    def resize_image_func(self, image, new_A=128,order = 3,nii_scale = [1,1,1], return_factor = False):
        A, _, C = image.shape

        scale_A = new_A / A
        scale_C = scale_A

        resized_image = scipy.ndimage.zoom(image, (scale_A, scale_A, 1), order=order)
        resized_image = scipy.ndimage.zoom(resized_image, (1, 1, scale_C), order=0)

        return resized_image
    
    def expand_image_dim3(self, image, target_size=40):
        # Get the current size of the third dimension
        current_size = image.shape[2]

        # Check if the third dimension is less than the target size
        if current_size < target_size:
            # Calculate the zoom factor for the third dimension
            zoom_factor = target_size / current_size
            # Apply nearest-neighbor interpolation to expand the third dimension
            expanded_image = zoom(image, (1, 1, zoom_factor), order=0)
        else:
            expanded_image = image
        return expanded_image
    
    def adjust_image_depth(self, image):
        target_depth = self.resize_image
        current_depth = image.shape[2]
        min_value = image.min()

        if current_depth < target_depth:
            pad_size = (target_depth - current_depth) // 2
            pad_before = pad_size + (target_depth - current_depth) % 2
            pad_after = pad_size

            padding = ((0, 0), (0, 0), (pad_before, pad_after))
            padded_image = np.pad(image, padding, mode='constant', constant_values=min_value)
            return padded_image
        elif current_depth > target_depth:
            start = (current_depth - target_depth) // 2
            end = start + target_depth
            cropped_image = image[:, :, start:end]
            return cropped_image
        else:
            return image
        
    def set_foreground_background(self, mask, foreground_classes=None):
        """
        Set the specified classes as foreground (1) and others as background (0).

        Parameters:
        - mask: numpy array of shape (128, 128, 128) with integer values representing different classes
        - foreground_classes: list or set of integers representing the classes to set as foreground.
                              If None, randomly select one or more classes as foreground.

        Returns:
        - modified_mask: numpy array with the specified classes set as foreground (1) and others as background (0)
        """
        unique_classes = np.unique(mask)

        if foreground_classes is None:
            raise ValueError("You must set the foreground class!!!")
            # Randomly select one or more classes as foreground
            # num_classes_to_select = np.random.randint(1, len(unique_classes) + 1)
            # foreground_classes = set(np.random.choice(unique_classes, num_classes_to_select, replace=False))
        elif foreground_classes=='random':
            return mask
        else:
            foreground_classes = set(foreground_classes)

        # Create a new mask with the same shape
        modified_mask = np.zeros_like(mask, dtype=float)
        
        # Set foreground classes to 1
        mask = np.round(mask)
        for class_value in foreground_classes:
            modified_mask[mask == class_value] = 1.

        return modified_mask

class MetaDataset_Multi_Extended(MetaDataset_Multi):
    def __init__(self, dataset_dir, 
                 data_loading_config=None, 
                 transform=None, 
                 resize_image=128, 
                 train_or_val='train', 
                 train_or_val_split_rate=0.1,
                 data_split_seed=0,
                 group_size=3,
                DDPM = False,
                skip_resize = False,
                len_divid = None,
                cut_length = False,
                random_context_size = False,
                in_order = False):
        """
        Extend the MetaDataset_Multi class to include a group size variable.
        
        Args:
            dataset_dir (string): Path to the directory of dataset.
            data_loading_config (dict): Configuration of loading dataset including dataset ID, task, input, output, and sampling rate.
            transform (callable, optional): Optional transform to be applied on a sample.
            resize_image (int): Size to resize the images to.
            train_or_val (string): Specify whether the dataset is for training or validation.
            train_or_val_split_rate (float): Split rate for training and validation datasets.
            data_split_seed (int): Seed for random data splitting.
            group_size (int): Size of the support set size+1.
        """
        super().__init__(dataset_dir, 
                         data_loading_config, 
                         transform, 
                         resize_image, 
                         train_or_val, 
                         train_or_val_split_rate, 
                         data_split_seed,
                        DDPM)
        
        self.group_size = group_size
        
        self.len_divid = len_divid
        self.skip_resize = skip_resize
        if self.skip_resize:
            print('---Do skip_resize!---'*3)
            
        self.cut_length = cut_length # cut the validation length
        
        self.random_context_size = random_context_size
        if self.random_context_size:
            print('---Do random_context_size!---'*3)
        self.in_order = in_order
        
    def get_single_item(self, idx):
        sample = self.data_samples[idx]
        
        # Load images for the given input_
        image = self.load_data(sample,'input')

        # Load label
        label = self.load_data(sample, 'output')
        
        if self.skip_resize:
            pass
        else:
            # crop
            image,crop_coords = self.crop_image(image)
            label,_ = self.crop_image(label,crop_coords)

            # padding
            image,padding_config =  self.pad_image(image, padding_config = None)
            label,_ =  self.pad_image(label, padding_config = padding_config)

            # resize
            image = self.resize_image_func(image, new_A=self.resize_image,order = 3)
            if sample['output']=='Seg':
                label = self.resize_image_func(label, new_A=self.resize_image,order = 0)
            else:
                label = self.resize_image_func(label, new_A=self.resize_image,order = 3)

            # resize the third dimension if it is smaller than 40
            image = self.expand_image_dim3(image, target_size=40)
            label = self.expand_image_dim3(label, target_size=40)

            # adjust depth to 128*128*128
            image = self.adjust_image_depth(image)
            label = self.adjust_image_depth(label)
        
        # normalize again
        image = self.min_max_normalize_with_clipping(image, clip = False)
        if sample['output']!='Seg':
            label = self.min_max_normalize_with_clipping(label, clip = False)
        if sample['output']=='Seg':
            label = np.round(label)
        
        # Task specific modified
        image, label = self.task_specific_transform( image, label, sample)
        
        # rotate if it is from synthetic data
        
            
        return {'image': image, 'label': label, 'idx': idx, 'task': sample['task'],
                'dataset': sample['sample'].split('/')[-3], 'weight': sample['weight'],
               'input':sample['input'],'output':sample['output'],'sample':sample}
    
    def __len__(self):
        if self.train_or_val == 'val':
            if self.len_divid is None:
                len_divid = 4
            else:
                len_divid = self.len_divid/10*4
            len_ = int(len(self.data_samples)/self.group_size/len_divid)
            
            if self.cut_length and len_ > 200/self.group_size: # if validation is too long cut it!
                len_ = int(200/self.group_size)
            
        elif self.train_or_val == 'train': 
            if self.len_divid is None:
                len_divid = 10
            else:
                len_divid = self.len_divid
            len_ = int(len(self.data_samples)/self.group_size/len_divid)
            
        else:
            assert False, "Did not set the right Key for train_or_val"
        if len_<1: len_=1
        return len_
    def min_max_normalize_with_clipping(self, image, min_percentile=0.1, max_percentile=99.9, clip = True):
        """
        Perform Min-Max normalization on an input image array with clipping to handle outliers.

        Parameters:
        image (numpy.ndarray): Input image array of shape (H, W) or (C, H, W).
        min_percentile (float): The lower percentile to clip the pixel values.
        max_percentile (float): The upper percentile to clip the pixel values.

        Returns:
        numpy.ndarray: Normalized image array with pixel values in the range [0, 1].
        """
        if clip:
            # Calculate the min and max values based on the given percentiles
            min_val = np.percentile(image, min_percentile)
            max_val = np.percentile(image, max_percentile)

            # Clip the image values to the calculated min and max
            clipped_image = np.clip(image, min_val, max_val)

            # Normalize the clipped image to [0, 1]
            normalized_image = (clipped_image - min_val) / (max_val - min_val + 1e-5)  # Add a small epsilon to avoid division by zero
        else:
            min_ = image.min()
            max_ = image.max()
            normalized_image = (image - min_) / (max_ - min_ + 1e-5)  # Add a small epsilon to avoid division by zero

        return normalized_image
    
    def __getitem__(self, idx):
        if len(self.task_data_num) != len(self.task_sample_rate):
            raise ValueError("List self.task_data_num and self.task_sample_rate must have the same length")
        if not self.in_order:
            self.task_idx  = random.choices(list(range(len(self.task_data_num))), weights=self.task_sample_rate, k=1)[0]
        else:
            self.task_idx = idx%len(self.task_data_num)

        start_idx = sum(self.task_data_num[:self.task_idx])
        end_idx = sum(self.task_data_num[:self.task_idx+1])
        
        # add context_size randomness
        if self.random_context_size and random.random() < 0.25:
            group_size = random.choice([2, 3, 4, 5, 6, 7, 8, 9])
        else:
            group_size =  self.group_size
        
        if not self.in_order:
            self.all_idx = [random.randint(start_idx, end_idx-1) for _ in range(group_size)]
        else:
            self.all_idx = [i for i in range(start_idx,start_idx+self.group_size)]
            print("self.all_idx:",self.all_idx)
        
        # load the image one by one
        item = {'image': [], 'label': [], 'idx':[],'task': None,'samples':[]}
        for i in self.all_idx:
            tmp = self.get_single_item(i)
            item['image'].append(tmp['image'])
            item['label'].append(tmp['label'])
            item['idx'].append(tmp['idx'])
            item['task']=tmp['task']
            item['dataset']=tmp['dataset']
            item['weight']=tmp['weight']
            item['input']=tmp['input']
            item['output']=tmp['output']
            item['samples'].append(tmp['sample'])
            
        item['image'] = np.stack(item['image'], axis=0)
        item['label'] = np.stack(item['label'], axis=0)
        
        # load label setting of segmentation 
        if item['task']=='Seg':
            task_config = item['samples'][0]['task_config']
        elif 'Seg' in item['task']:
            task_config = item['samples'][0]['task_config'][item['task'].index('Seg')]
        if 'Seg' in item['task']:
            item['label'] = self.set_foreground_background(item['label'],
                                                   task_config,
                                                   config = item['samples'][0]['config'],)
            
        
        if 'task_aug' in item['samples'][0]['task_config']:
            sample_tmp = self.get_task_config(task_aug=item['samples'][0]['task_config']['task_aug'])
            for index in range(item['label'].shape[0]):
                item['image'][index,:], item['label'][index,:] = self.task_specific_transform( item['image'][index,:], item['label'][index,:], sample_tmp)
        if self.transform:
            item = self.transform(item,transform_individual_rotate = '_syn_type' in item['dataset'])
        
        if not isinstance(item['label'], torch.Tensor):
            item['label'] = torch.tensor(item['label'])
        if not isinstance(item['image'], torch.Tensor):
            item['image'] = torch.tensor(item['image'])
        
        # Normalize the data
        item['label'] = self.normalize_3d_volume(item['label'])
        item['image'] = self.normalize_3d_volume(item['image'])
        
        if self.DDPM:
            item['image'] = item['image']*2-1
            item['label'] = item['label']*2-1
            
        return item
    def set_foreground_background(self, mask, task_config, config = None):
        """
        Set the specified classes as foreground (1) and others as background (0).

        Parameters:
        - mask: numpy array of shape (128, 128, 128) with integer values representing different classes
        - foreground_classes: list or set of integers representing the classes to set as foreground.
                              If None, randomly select one or more classes as foreground.

        Returns:
        - modified_mask: numpy array with the specified classes set as foreground (1) and others as background (0)
        """
        foreground_classes = task_config.get('foreground_classes', None)
        remove = task_config.get('remove', None)
        

        if foreground_classes is None:
            raise ValueError("You must set the foreground class!!!")
#             unique_classes = np.unique(mask)
            # Randomly select one or more classes as foreground
            # num_classes_to_select = np.random.randint(1, len(unique_classes) + 1)
            # foreground_classes = set(np.random.choice(unique_classes, num_classes_to_select, replace=False))
        elif foreground_classes=='random':
            class_all = [int(i) for i in list(config['labels'].keys())]
            # remove specific class
            if remove is not None:
                remove = [int(x) for x in remove]
                class_all = [x for x in class_all if x not in remove]
                
            max_length = len(class_all)-1 if len(class_all)-1 < 5 else 5
            length = random.randint(1, max_length)
            foreground_classes = random.sample(class_all, length)
            foreground_classes = set(foreground_classes)
        else:
            foreground_classes = set(foreground_classes)

        # Create a new mask with the same shape
        modified_mask = np.zeros_like(mask, dtype=float)
        
        # Set foreground classes to 1
        mask = np.round(mask)
        for class_value in foreground_classes:
            modified_mask[mask == class_value] = 1.
        return modified_mask
    
    def get_task_config(self, task_aug):
        sample_tmp = {'task':[],'task_config':[]}
        # shuffled the order of the task aug
        shuffled_task_aug = dict(random.sample(task_aug.items(), len(task_aug)))
        for task, prob in shuffled_task_aug.items():
            if np.random.rand() < prob:
                sample_tmp['task'].append(task)
                if task in 'Denoi':
                    if np.random.rand() < 0.5:
                        sample_tmp['task_config'].append({'noise_type' :"gaussian" ,'random':True})
                    else:
                        sample_tmp['task_config'].append({'noise_type' :"salt_pepper" ,'random':True})
                else:
                    sample_tmp['task_config'].append({})
        return sample_tmp
    def normalize_3d_volume(self, target_in: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalizes a batch of 3D volumes (shape: [..., D, H, W]) to the range [0, 1] independently for each sample.

        Args:
            target_in (torch.Tensor): Input tensor of shape [..., 1, D, H, W], where N is the batch size,
                                      and (D, H, W) are the depth, height, and width of the 3D volume.
            eps (float): A small value to prevent division by zero. Default is 1e-8.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input, where each sample is scaled to [0, 1].
        """
        # Ensure the input tensor is of type float32 to avoid integer division issues
        if target_in.dtype != torch.float32:
            target_in = target_in.to(torch.float32)

        # Compute the minimum and maximum values for each sample independently
        # Input shape: [N, 1, D, H, W] → Output shape: [N, 1, 1, 1, 1]
        min_vals = torch.amin(target_in, dim=(-3, -2, -1), keepdim=True)
        max_vals = torch.amax(target_in, dim=(-3, -2, -1), keepdim=True)

        # Compute the dynamic range and prevent division by zero
        dynamic_range = max_vals - min_vals
        dynamic_range[dynamic_range < eps] = eps  # Replace small ranges with eps to avoid division by zero

        # Normalize the input tensor to the range [0, 1]
        normalized = (target_in - min_vals) / dynamic_range

        return normalized



class MetaDataset_Multi_Extended_ordered_SAM(MetaDataset_Multi_Extended):
    def __init__(self, dataset_dir, 
                        data_loading_config=None, 
                        transform=None, 
                        resize_image=128, 
                        train_or_val='train', 
                        train_or_val_split_rate=0.1,
                        data_split_seed=0,
                        group_size=3,
                        DDPM = False,
                        skip_resize = False,
                        len_divid = None,
                        cut_length = False,
                        random_context_size = False,
                        task_idx=None,
                        all_idx_relative=None):
        """
        Allow the dataloader to output fix data
        """
        super().__init__(dataset_dir, 
            data_loading_config, 
            transform, 
            resize_image, 
            train_or_val, 
            train_or_val_split_rate,
            data_split_seed,
            group_size,
            DDPM,
            skip_resize,
            len_divid,
            cut_length,
            random_context_size)
        
        # Allow task_idx and all_idx to be set manually
        self.task_idx = task_idx
        self.all_idx_relative = all_idx_relative
        

    def __getitem__(self, idx):
        if len(self.task_data_num) != len(self.task_sample_rate):
            raise ValueError("List self.task_data_num and self.task_sample_rate must have the same length")
            
        # Use provided task_idx if set, otherwise select randomly
        if self.task_idx is None:
            self.task_idx = random.choices(list(range(len(self.task_data_num))), weights=self.task_sample_rate, k=1)[0]
            print('You randomly choose the task id!!!!',"self.task_idx:",self.task_idx)
        
        start_idx = sum(self.task_data_num[:self.task_idx])
        end_idx = sum(self.task_data_num[:self.task_idx+1])
        
        self.all_idx = [i+idx for i in range(self.group_size)]
        
        
        
        
        item = {'image': [], 'label': [], 'idx':[],'task': None,'samples':[]}
        for i in self.all_idx:
            tmp = self.get_single_item(i)
            item['image'].append(tmp['image'])
            item['label'].append(tmp['label'])
            item['idx'].append(tmp['idx'])
            item['task']=tmp['task']
            item['dataset']=tmp['dataset']
            item['weight']=tmp['weight']
            item['input']=tmp['input']
            item['output']=tmp['output']
            item['samples'].append(tmp['sample'])
            
        item['image'] = np.stack(item['image'], axis=0)
        item['label'] = np.stack(item['label'], axis=0)

#         if item['task']=='Seg':
#             item['label'] = self.set_foreground_background(item['label'],
#                                                    **item['samples'][0]['task_config'],
#                                                    config = item['samples'][0]['config'],)
#       # load label setting of segmentation 
        if item['task']=='Seg':
            task_config = item['samples'][0]['task_config']
        elif 'Seg' in item['task']:
            task_config = item['samples'][0]['task_config'][item['task'].index('Seg')]
        if 'Seg' in item['task']:
            item['label'] = self.set_foreground_background(item['label'],
                                                   task_config,
                                                   config = item['samples'][0]['config'],)
        
        # Apply transform (if any)
        if self.transform:
            item = self.transform(item,transform_individual_rotate = '_syn_' in item['dataset'])
            
        if not isinstance(item['label'], torch.Tensor):
            item['label'] = torch.tensor(item['label'])
        if not isinstance(item['image'], torch.Tensor):
            item['image'] = torch.tensor(item['image'])
        
            
        return item
    
    def __len__(self):
        return len(self.data_samples)-self.group_size





class MetaDatasetf_transform_1channel(object):
    def __init__(self, 
                 flip_prob=0.05,
                 sobel_prob = 0.05,
                 dilation_errosion_prob = 0.05,
                 individual_rotate_prob = 0.25,
                 augmentation_seg = True,
                 augmentation_gen = True,
                 ):
        """
        Parameters:
        flip_prob (float): The probability of flipping the mask values.
        sobel_prob (float): The probability of applying sobel edge detection.
        augmentation_seg (bool): Whether apply augmentation for segmentation.
        augmentation_gen (bool): Whether apply augmentation for generation task.
        """
        
        self.flip_prob = flip_prob
        self.sobel_prob = sobel_prob
        self.dilation_errosion_prob = dilation_errosion_prob
        self.individual_rotate_prob = individual_rotate_prob
        
        
        # Build augmentation object
        self.transform_rotate = Compose([
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,1)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(1,2)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,2)),
        ])
        
        self.transform_seg = Compose([
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=2),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,1)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(1,2)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,2)),
                RandomResample(keys=["image", "label"], prob=0.15),
                RandShiftIntensityd(keys='image', prob=0.2, offsets=0.2),
                RandScaleIntensityd(keys='image', prob=0.2, factors=0.1),
                RandGaussianNoised(keys='image', prob=0.1, mean=0.0, std=0.02),
                Rand3DElasticd(
                        keys=['image', 'label'],
                        prob=0.05,
                        sigma_range=(7, 8),
                        magnitude_range=(100, 200),
                        translate_range=(0, 0, 0),
                        rotate_range=(0.3, 0.3,0.3),
                        scale_range=(0.2, 0.2, 0.2),
                        mode=('bilinear', 'nearest')  # Specify different modes for image and label
                    ),
                EnsureTyped(keys=['image', 'label'])
            ])
        self.augmentation_seg = augmentation_seg
        
        self.transform_gen = Compose([
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=2),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,1)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(1,2)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,2)),
                RandomResample(keys=["image", "label"], prob=0.15),
                RandShiftIntensityd(keys=['image', 'label'], prob=0.2, offsets=0.2),
                RandScaleIntensityd(keys=['image', 'label'], prob=0.2, factors=0.1),
                RandGaussianNoised(keys='image', prob=0.1, mean=0.0, std=0.02),
                Rand3DElasticd(
                        keys=['image', 'label'],
                        prob=0.05,
                        sigma_range=(7, 8),
                        magnitude_range=(100, 200),
                        translate_range=(0, 0, 0),
                        rotate_range=(0.3, 0.3,0.3),
                        scale_range=(0.1, 0.1, 0.1),
                        mode=('bilinear', 'bilinear')  # Specify different modes for image and label
                    ),
                EnsureTyped(keys=['image', 'label'])
            ])
        self.augmentation_gen = augmentation_gen

    
    def sobel_edge_detection_3d(self, image):
        sobel_edges = np.zeros_like(image)

        for axis in range(3):
            sobel_edges += np.abs(sobel(image, axis=axis))/3

        return sobel_edges
    def sobel_edge_detection_batch(self, mask):
        for i in range(mask.shape[0]):
            mask[i,:] = self.sobel_edge_detection_3d(mask[i,:])
        mask = mask/mask.max()
        mask = (mask>0.3).astype(float)
        return mask
    
    def dilate_mask(self, mask, structure_size=3, iterations=1):
        structure = np.ones((structure_size, structure_size, structure_size), dtype=bool)
        dilated = binary_dilation(mask, structure=structure, iterations=iterations)
        return dilated.astype(mask.dtype)

    def dilate_batch_masks(self, batch_mask, structure_size=3, iterations=1):
        dilated_masks = [self.dilate_mask(mask, structure_size, iterations) for mask in batch_mask]
        return np.stack(dilated_masks, axis=0)
    
    def shrink_mask(self, mask, structure_size=3, iterations=1):
        structure = np.ones((structure_size, structure_size, structure_size), dtype=bool)
        shrunk = binary_erosion(mask, structure=structure, iterations=iterations)
        return shrunk.astype(mask.dtype)

    def shrink_batch_masks(self, batch_mask, structure_size=3, iterations=1):
        shrunk_masks = [self.shrink_mask(mask, structure_size, iterations) for mask in batch_mask]
        return np.stack(shrunk_masks, axis=0)
    
    def process_item_individually(self, item, transform_fn):
        new_item = {'label': np.zeros_like(item['label']), 'image': np.zeros_like(item['image'])}

        for i in range(item['label'].shape[0]):
            label_temp = np.expand_dims(item['label'][i], axis=0)  # (1, 128, 128, 128)
            image_temp = np.expand_dims(item['image'][i], axis=0)  # (1, 128, 128, 128)
            item_temp = {'label': label_temp, 'image': image_temp}
            item_temp = transform_fn(item_temp)

            new_item['label'][i] = np.squeeze(item_temp['label'])
            new_item['image'][i] = np.squeeze(item_temp['image'])

        return new_item
    
    def __call__(self, item, transform_individual_rotate = False):
        '''
        Item includes keys: image, label, idx, task.
        Transforms for ICL segmentation task. All transforms are on item['label'].
        Shape of item['label'] is: (B,H,W,D)

        '''
#         print("item['label']:",item['label'].shape,type(item['label']))
        if item['task'] == 'Seg' or 'Seg' in item['task']:
            # flip
            if np.random.rand() < self.flip_prob:
                item['label'] = 1-item['label']
            if np.random.rand() < self.flip_prob:
                item['image'] = 1-item['image']
                
            # edge detection
            if np.random.rand() < self.sobel_prob:
                item['label'] = self.sobel_edge_detection_batch(item['label'])
            
            if np.random.rand() < self.dilation_errosion_prob:
                # Inflation
                item['label'] = self.dilate_batch_masks(item['label'],iterations=np.random.randint(1,2))
            elif np.random.rand() < self.dilation_errosion_prob:
                # Shrinkage
                item['label'] = self.shrink_batch_masks(item['label'],iterations=1)
            
            # Augmentation
            if self.augmentation_seg:
                item = self.transform_seg(item)
                
            if transform_individual_rotate and (np.random.rand() < self.individual_rotate_prob):
                new_item = self.process_item_individually(item, self.transform_rotate)
                item['label'],item['image'] = new_item['label'],new_item['image']
                
        elif item['task'] in ['ModTran','Bias','Denoi','2D23D','Inp','SupRes'] or isinstance(item['task'], list):
            if np.random.rand() < self.flip_prob:
                item['label'] = 1-item['label']
            if np.random.rand() < self.flip_prob:
                item['image'] = 1-item['image']
                
            # Augmentation
            if self.augmentation_gen:
                item = self.transform_gen(item)
#         print("item['label']:",item['label'].shape,type(item['label']))
        return item


import numpy as np
import torch
import torch.nn.functional as F
from typing import Mapping, Hashable, Sequence, Union, Optional

from monai.transforms import MapTransform, InvertibleTransform, LazyTransform


class RandomResample(MapTransform, InvertibleTransform, LazyTransform):
    backend = ["numpy", "torch"]

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        prob: float = 0.05,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)

        object.__setattr__(self, "_lazy_evaluation", bool(lazy))
        # ------------------------------------------------------

        self.prob = float(prob)
        self.rng = np.random.default_rng()

    # ----------------- internal helpers ----------------- #
    @staticmethod
    def _downsample(x, dim: int, factor: int):
        slicer = [slice(None)] * x.ndim
        slicer[dim] = slice(None, None, factor)
        return x[tuple(slicer)]

    @staticmethod
    def _pad_to_size(x, dim: int, target: int):
        size = x.shape[dim]
        if size >= target:
            return x
        pad_total = target - size
        pad_pre = pad_total // 2
        pad_post = pad_total - pad_pre

        if isinstance(x, torch.Tensor):
            # (D2, D1, W2, W1, H2, H1) 的顺序
            pad_cfg = [0, 0, 0, 0, 0, 0]
            if dim == 3:          # D
                pad_cfg[0:2] = [pad_post, pad_pre]
            elif dim == 2:        # W
                pad_cfg[2:4] = [pad_post, pad_pre]
            elif dim == 1:        # H
                pad_cfg[4:6] = [pad_post, pad_pre]
            return F.pad(x, pad_cfg, mode="constant", value=x.min().item())

        pad_width = [(0, 0)] * x.ndim
        pad_width[dim] = (pad_pre, pad_post)
        return np.pad(x, pad_width, mode="constant", constant_values = x.min())

    @staticmethod
    def _resize_nn(x, dim: int, target: int):
        if x.shape[dim] == target:
            return x
        rep = target // x.shape[dim] + 1
        if isinstance(x, torch.Tensor):
            x = x.repeat_interleave(rep, dim=dim)
        else:
            x = np.repeat(x, rep, axis=dim)
        slicer = [slice(None)] * x.ndim
        slicer[dim] = slice(0, target)
        return x[tuple(slicer)]

    # ------------------- forward ------------------- #
    def __call__(
        self,
        data: Mapping[Hashable, Union[torch.Tensor, np.ndarray]],
        lazy: Optional[bool] = None,
    ):
        if np.random.rand() >= self.prob:
            return data
        
        d = dict(data)

        axis = int(self.rng.choice([1, 2, 3]))      # H / W / D
        factor = int(self.rng.choice([2, 3, 4]))    # 2, 3, 4
        use_padding = bool(self.rng.random() < 0.5) # True=pad, False=resize

        for key in self.key_iterator(d):
            x = d[key]
            if x.ndim != 4:
                raise ValueError(
                    f"{self.__class__.__name__}: 期望 4-D (B,H,W,D)，收到 {x.shape}"
                )
            orig_len = x.shape[axis]
            x_ds = self._downsample(x, axis, factor)
            x_out = (
                self._pad_to_size(x_ds, axis, orig_len)
                if use_padding
                else self._resize_nn(x_ds, axis, orig_len)
            )
            d[key] = x_out

        return d

    # ------------------- inverse ------------------- #
    def inverse(self, data):
        raise NotImplementedError(
            ""
        )





import random
from torch.utils.data import Sampler    
class Meta_dataset_Sampler(Sampler):
    def __init__(self, batchsize=4, task_data_num = [10,10]):
        """
            batchsize (int): batchsize
        """
        self.n = batchsize
        self.count = 0
        self.current_idx = None
        self.task_data_num = task_data_num
        self.sample_count = 0
        
        tmp = sum(self.task_data_num)
        self.epoch_len = tmp-(tmp%self.n)

    def __iter__(self):
        self.sample_count = 0 
        return self
    
    def __len__(self):
        return self.epoch_len

    def __next__(self):

        
        if self.sample_count >=self.epoch_len:
            raise StopIteration
            
            
        if self.count == 0:
            self.task_idx = random.randint(0, len(self.task_data_num)-1)
            self.count = self.n
        
        self.count -= 1
        
        start_idx = sum(self.task_data_num[:self.task_idx])
        end_idx = sum(self.task_data_num[:self.task_idx+1])
        self.current_idx = random.randint(start_idx, end_idx-1)
        
        self.sample_count += 1
        
        return self.current_idx



if __name__ == "__main__":    
    """
    Configuration of dataloader
    task abbreviation:
        Seg: Segmentation
        ModTran: Modality Transfer
        SupRes: Super Resolution
        Denoi: Denoising
        Inp: Inpainting
    """

    data_loading_config = [
        {'name':'Dataset668_BraTS2021_MR_1251', 'input':1,'output':3,'task':'ModTran','task_config':[],'sample rate':1},
        {'name':'Dataset602_ADNI', 'input':0,'output':2,'task':'ModTran','task_config':[],'sample rate':1},
        {'name':'Dataset601_TopCow_MRA', 'input':0,'output':'Seg','task':'Seg','task_config':[],'sample rate':1},
        {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0,'task':'SupRes','task_config':{'factor':2},'sample rate':1},
            ]

    """
    [
    {'sample': 'image_path', 'input':0, 'output':'Seg', 'Task': 'Seg'},
    {'sample': 'image_path', 'input':0, 'output':'Seg', 'Task': 'Seg'},
    ]
    """


    dataset = MetaDataset_Multi(
        dataset_dir = 'dataset/', 
        data_loading_config = data_loading_config
    )
    
    from torch.utils.data import DataLoader, Dataset
    batchsize = 4
    sampler = Meta_dataset_Sampler(batchsize = 4,task_data_num = dataset.task_data_num)
    print('len(sampler)',len(sampler))
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=4,num_workers=16)

    


'''
Other function for task specific transform
'''

import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import json
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

'''
Mask 3D image to three 2D slice
'''
def retain_slice_and_next(image, axis, slice_index, **kwargs):
    """
    Retain the given slice and the next slice along the specified axis,
    setting all other values in the image to the minimum value of the input image.
    
    Parameters:
    - image: numpy array of shape (128, 128, 128)
    - axis: int, the axis along which to retain the slices (0, 1, or 2)
    - slice_index: int, the index of the slice to retain
    
    Returns:
    - modified_image: numpy array with the specified slices retained and others set to the minimum value
    """
    # Validate input
    if axis not in (0, 1, 2):
        raise ValueError("Axis must be 0, 1, or 2")
    if not (0 <= slice_index < image.shape[axis] - 1):
        raise ValueError(f"slice_index must be between 0 and {image.shape[axis] - 2}")
        
    random_flag = kwargs.get("random", False)
    if random_flag: slice_index+=np.random.randint(-10, 11)
    
    # Find the minimum value in the image
    min_value = np.min(image)
    
    # Create a copy of the image to modify
    modified_image = np.full(image.shape, min_value, dtype=image.dtype)
    
    # Retain the specified slice and the next slice
    if axis == 0:
        modified_image[slice_index-1:slice_index+2, :, :] = image[slice_index-1:slice_index+2, :, :]
    elif axis == 1:
        modified_image[:, slice_index-1:slice_index+2, :] = image[:, slice_index-1:slice_index+2, :]
    else:
        modified_image[:, :, slice_index-1:slice_index+2] = image[:, :, slice_index-1:slice_index+2]
    
    return modified_image

if __name__ == "__main__":  
    # Example usage
    image = np.random.rand(128, 128, 128)  # Example 3D image
    
    config = {'axis': 0, 'slice_index' : 64}
    
    modified_image = retain_slice_and_next(image, **config)
    


    # Display a slice of the original and inpainted image
    slice_index = image.shape[2] // 2
    slice_index = config['slice_index']+1
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image[:, :, slice_index], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('2D Mask Image')
    plt.imshow(modified_image[:, :, slice_index], cmap='gray')
    plt.show()

'''
Function for adding noise to a 3D medical image.
'''
def add_noise(image, noise_type="gaussian", **kwargs):
    """
    Add noise to a 3D medical image.

    Parameters:
    - image: numpy array of shape (128, 128, 128)
    - noise_type: str, type of noise to add ("gaussian", "salt_pepper", "poisson")
    - kwargs: additional arguments for noise generation

    Returns:
    - noisy_image: numpy array with added noise
    """
    if noise_type == "gaussian":
        mean = kwargs.get("mean", 0)
        std = kwargs.get("std", 0.10)
        random_flag = kwargs.get("random", False)
        if random_flag: 
            std = np.random.uniform(0.15*std, std)

#         print('Gaussian std',std)
        gaussian_noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + gaussian_noise
    
    elif noise_type == "salt_pepper":
        salt_prob = kwargs.get("salt_prob", 0.04)
        pepper_prob = kwargs.get("pepper_prob", 0.04)
        random_flag = kwargs.get("random", False)
        if random_flag: 
            salt_prob = np.random.uniform(0.25*salt_prob, salt_prob)
            pepper_prob = np.random.uniform(0.25*pepper_prob, pepper_prob)
#         print('salt_prob pepper_prob',salt_prob,pepper_prob)
        
        
        noisy_image = image.copy()
        
        range_ = image.max()-image.min()
        salt = image.max()-0.01*range_
        pepper = image.min()+0.01*range_
        
        # Salt noise
        num_salt = np.ceil(salt_prob * image.size)
        coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = salt
        
        # Pepper noise
        num_pepper = np.ceil(pepper_prob * image.size)
        coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = pepper
    
    elif noise_type == "poisson":
        noisy_image = np.random.poisson(image)
    
    else:
        raise ValueError("Unsupported noise type. Choose from 'gaussian', 'salt_pepper', or 'poisson'.")
    
    return_ = noisy_image
    # norm
    return_ = (return_-return_.min())/(return_.max()-return_.min())
    return return_

if __name__ == "__main__":  
    # Example usage
    image = np.ones((128, 128, 128))  # Example image
    image[0,0,0] = 0
    image[0,0,2] = 2

    # Add Gaussian noise
    noisy_image_gaussian = add_noise(image, noise_type="gaussian", mean=0, std=0.05)

    # Add Salt and Pepper noise
    noisy_image_salt_pepper = add_noise(image, noise_type="salt_pepper", salt_prob=0.02, pepper_prob=0.02)

    # # Add Poisson noise
    noisy_image_poisson = add_noise(image, noise_type="poisson")

    # Display a slice of the original and inpainted image
    slice_index = image.shape[2] // 2
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(image[:, :, slice_index], cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.title('noisy_image_gaussian Image')
    plt.imshow(noisy_image_gaussian[:, :, slice_index], cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('noisy_image_salt_pepper Image')
    plt.imshow(noisy_image_salt_pepper[:, :, slice_index], cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('noisy_image_poisson Image')
    plt.imshow(noisy_image_poisson[:, :, slice_index], cmap='gray')
    plt.colorbar()
    plt.show()
'''
Function for create painting
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import random

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, x):
    return a + x * (b - a)

def grad(hash, x, y, z):
    h = hash & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

def perlin(x, y, z, perm):
    xi, yi, zi = x.astype(int) & 255, y.astype(int) & 255, z.astype(int) & 255
    xf, yf, zf = x - xi, y - yi, z - zi
    u, v, w = fade(xf), fade(yf), fade(zf)

    aaa = perm[perm[perm[xi] + yi] + zi]
    aba = perm[perm[perm[xi] + yi + 1] + zi]
    aab = perm[perm[perm[xi] + yi] + zi + 1]
    abb = perm[perm[perm[xi] + yi + 1] + zi + 1]
    baa = perm[perm[perm[xi + 1] + yi] + zi]
    bba = perm[perm[perm[xi + 1] + yi + 1] + zi]
    bab = perm[perm[perm[xi + 1] + yi] + zi + 1]
    bbb = perm[perm[perm[xi + 1] + yi + 1] + zi + 1]

    x1 = lerp(grad(aaa, xf, yf, zf), grad(baa, xf - 1, yf, zf), u)
    x2 = lerp(grad(aba, xf, yf - 1, zf), grad(bba, xf - 1, yf - 1, zf), u)
    y1 = lerp(x1, x2, v)

    x1 = lerp(grad(aab, xf, yf, zf - 1), grad(bab, xf - 1, yf, zf - 1), u)
    x2 = lerp(grad(abb, xf, yf - 1, zf - 1), grad(bbb, xf - 1, yf - 1, zf - 1), u)
    y2 = lerp(x1, x2, w)

    return (y1 + y2) / 2

def generate_perlin_noise_3d(shape, scale):
    perm = np.arange(256)
    np.random.shuffle(perm)
    perm = np.stack([perm, perm]).flatten()

    # Generate coordinates with the desired scale
    lin = [np.linspace(0, s, s // scale, endpoint=False) for s in shape]
    x, y, z = np.meshgrid(*lin, indexing='ij')

    noise = perlin(x, y, z, perm)
    noise = zoom(noise, scale, order=1)  # Interpolate to match the desired shape
    return noise

def create_inpainting_mask(image_shape, scale=10, threshold=0):
    perlin_noise = generate_perlin_noise_3d(image_shape, scale)
    mask = perlin_noise > threshold
    return mask

def apply_inpainting_mask(image, mask):
    inpainted_image = np.copy(image)
    inpainted_image[mask] = 0  # You can choose a different inpainting strategy here
    return inpainted_image

def img_2_painting(image, scales = [8, 16],thresholds = (0.1, 0.4), **kwargs):
    scale = random.choice(scales)
    threshold = random.uniform(thresholds[0], thresholds[1])
    image_shape = image.shape
    
    # Create a random binary mask from Perlin noise
    mask = create_inpainting_mask(image_shape, scale=scale, threshold=threshold)

    # Apply the mask to the image
    inpainted_image = apply_inpainting_mask(image, mask)
    
    return inpainted_image

if __name__ == "__main__":  
    # Example usage
    image_shape = (128, 128, 128)
    image = np.random.rand(*image_shape)  # Replace with actual 3D medical image

    config = {'scales': [8, 16, 32], 'thresholds' : (0.1, 0.3)}

    inpainted_image = img_2_painting(image, **config)

    # Display a slice of the original and inpainted image
    slice_index = image_shape[2] // 2
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image[:, :, slice_index], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Inpainted Image')
    plt.imshow(inpainted_image[:, :, slice_index], cmap='gray')
    plt.show()


'''
Funtion for create bias field

'''
import numpy as np
from typing import Sequence, Tuple, Union
def rand_bias_field(
    img: Union[np.ndarray, 'torch.Tensor'],
    degree: int = 3,
    coeff_range: Tuple[float, float] = (-0.5, 0.5),
    dtype: Union[np.dtype, str] = np.float32,
    prob: float = 1.0,
    random_state: int = None,
    **kwargs,
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Random bias field augmentation for MR images as a function.

    Args:
        img: Input image.
        degree: Degree of freedom of the polynomials. The value should be no less than 1.
            Defaults to 3.
        coeff_range: Range of the random coefficients. Defaults to (0.0, 0.1).
        dtype: Output data type, if None, same as input image. Defaults to float32.
        prob: Probability to do random bias field.
        random_state: Seed for the random number generator.

    Returns:
        The image with the bias field applied.
    """
    def _generate_random_field(spatial_shape: Sequence[int], degree: int, coeff: Sequence[float], coeff_range: Tuple[float, float]):
        """
        Products of polynomials as bias field estimations
        """
        rank = len(spatial_shape)
        coeff_mat = np.zeros((degree + 1,) * rank)
#         coeff_mat = np.random.uniform(*coeff_range, size=coeff_mat.shape)
        coords = [np.linspace(-1.0, 1.0, dim, dtype=np.float32) for dim in spatial_shape]
        if rank == 2:
            coeff_mat[np.tril_indices(degree + 1)] = coeff
            return np.polynomial.legendre.leggrid2d(coords[0], coords[1], coeff_mat)
        if rank == 3:
            pts = [[0, 0, 0]]
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    for k in range(degree + 1 - i - j):
                        pts.append([i, j, k])
            if len(pts) > 1:
                pts = pts[1:]
            np_pts = np.stack(pts)
            coeff_mat[np_pts[:, 0], np_pts[:, 1], np_pts[:, 2]] = coeff
            return np.polynomial.legendre.leggrid3d(coords[0], coords[1], coords[2], coeff_mat)
        raise NotImplementedError("only supports 2D or 3D fields")

    if random_state is not None:
        np.random.seed(random_state)
    
    do_transform = np.random.rand() < prob
    if not do_transform:
        return img

    img = np.asarray(img, dtype=dtype)
    spatial_shape = img.shape[1:]
    n_coeff = int(np.prod([(degree + k) / k for k in range(1, len(spatial_shape) + 1)]))
    coeff = np.random.uniform(*coeff_range, n_coeff).tolist()
    
    num_channels, *spatial_shape = img.shape

    bias_fields = np.stack(
        [_generate_random_field(spatial_shape=spatial_shape, degree=degree, coeff=coeff, coeff_range = coeff_range) for _ in range(num_channels)],
        axis=0,
    )
    
    return_ = img * np.exp(bias_fields)
    # norm
    return_ = (return_-return_.min())/(return_.max()-return_.min())
    return return_