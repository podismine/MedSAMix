import torch
from tqdm import tqdm
import numpy as np

from evaluation.UniSeg_SAM import *

from evaluation.dataloader import MetaDataset_Multi_Extended_ordered_SAM
from evaluation.config import data_loading_config_meta

class image_loader_3D(object):

    def __init__(self, image=None, label=None):
        self.image = image
        self.label = label

        self.valid_indices = [i for i in range(label.shape[-1]) 
                              if torch.any(label[:, :, :, i] > 0)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        '''
        test_image: torch.Size([1, 3, 128, 128]) dtype=torch.uint8   
        label_slice: [1, 1, 128, 128] numpy
        '''
        slice_idx = self.valid_indices[idx]

        image_slice = self.image[:, :, :, slice_idx]  # shape: [1, H, W]
        image_slice = image_slice.repeat(3, 1, 1)  # shape: [ 3, H, W]
        
        image_slice = image_slice*255
        image_slice = image_slice.to(torch.uint8) 

        label_slice = self.label[:, :, :, slice_idx].numpy()  # shape: [1, H, W]

        return image_slice[None], label_slice[None], slice_idx
    
def eval_3d(predictor, task_name, data_root, is_train):
    task_names = list(data_loading_config_meta.keys())
    if task_name not in task_names:
        raise KeyError(f"{task_name} not found")
    # task_name = task_names[task_idx]
    data_loading_config = data_loading_config_meta[task_name]
    print('All task:',task_names)
    print('Length :',len(task_names))
    print('The current task is:', task_name)

    dataset_train = MetaDataset_Multi_Extended_ordered_SAM(
            dataset_dir = data_root, 
            skip_resize = True,
            data_loading_config = data_loading_config,
            train_or_val = 'val' if is_train is False else 'train',
            group_size = 1,
        train_or_val_split_rate = 0.8,
            )
    dataset_train.task_idx = 0 # Only Use the First Task

    print("======> Load SAM" )
    print(f"find data size: {len(dataset_train)}")#;exit()
    # Run inference
    Dice_medsam = []
    for i in tqdm(range(len(dataset_train))):
        item = dataset_train[i]
        dataloader_single = image_loader_3D(image = item['image'],
                                            label = item['label'])
        
        for j in range(len(dataloader_single)):
            test_image, test_mask, idx = dataloader_single.__getitem__(j)
            
            bbox = np.array(compute_bounding_box(test_mask[0,0]))
            predictor.set_image(test_image[0,:].permute(1,2,0).numpy())
            masks, scores, _, high_res_masks = predictor.predict(
                box=bbox, 
                mask_input =  None,
                multimask_output=False,
                attn_sim=None,  # Target-guided Attention
                target_embedding=None  # Target-semantic Prompting
            )

            dice_medsam = compute_dice_coefficient(masks[0,:], test_mask[0,0,:])
            Dice_medsam.append(dice_medsam)
            if j%20==0:
                print("dice_medsam:", round(dice_medsam,3),f"Subject: {i}; Slice: {idx}")
    print('Mean Dice:',np.mean(Dice_medsam))
    return np.mean(Dice_medsam)