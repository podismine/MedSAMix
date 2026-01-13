import torch

import os
import cv2
from tqdm import tqdm
import numpy as np

from evaluation.UniSeg_SAM import *

import cv2

class image_loader_2D(object):

    def __init__(self, inf_folder, 
                 Thres=None, 
                ):
        # Inference
        inf_image_paths = sorted(os.listdir(inf_folder))
        inf_image_paths = sorted([os.path.join(inf_folder,i) for i in inf_image_paths if '.png' in i])
        self.inf_image_paths = inf_image_paths
        print('Total length of test set:',len(inf_image_paths))
        
        # Save variable
        self.Thres = Thres

    def __len__(self):
        # return 10
        return int(len(self.inf_image_paths)* 1.0)
    
    def __getitem__(self, idx, is_train = True):
        test_image_path = self.inf_image_paths[idx]
        test_mask_path = test_image_path.replace('/image/','/mask/')
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(test_mask_path)
        test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2RGB)
        
        test_mask_thres = np.zeros((1,1,test_mask.shape[0],test_mask.shape[1]))
        for thres in self.Thres:
            test_mask_thres+= (test_mask[:,:,0]==thres).astype('float')[None,None,:,:]
        test_mask = test_mask_thres>0
        
    
        test_image = test_image.transpose(2,0,1)[None,:]
        test_image = torch.tensor(test_image)
        return test_image, test_mask, test_image_path
    
def eval_fundus(predictor, task_name, data_root, is_train):
    #thres cup: 0 disk: 0 128
    domain_type = task_name.split("_")[1].lower()
    task_thres = task_name.split("_")[2].lower()

    if task_thres == 'cup':
        thres = [0]
    elif task_thres == 'disk':
        thres = [0,128]
    else:
        raise KeyError("wrong task name for fundus")
    if is_train is True:
        data_path = os.path.join(data_root,f"Fundus/D{domain_type[1:]}/train/ROIs/image")
    else:
        data_path = os.path.join(data_root,f"Fundus/D{domain_type[1:]}/test/ROIs/image")

    Dice_medsam = []

    dataloader = image_loader_2D(data_path, thres)

    for i in tqdm(range(len(dataloader))):
        test_image, test_mask, test_image_path = dataloader.__getitem__(i, is_train = is_train)
        
        # MedSAM
        bbox = np.array(augment_bounding_box(test_mask))
        # PerSAM
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

    return np.mean(Dice_medsam)
