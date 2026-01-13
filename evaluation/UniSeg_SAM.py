import torch
from torch.nn import functional as F

import numpy as np
def get_structured_data_new(data, data_s, target_s, target_index=slice(0,1), size = 128):
    
    '''
    Shape of the data: [Batchsize, C, H, W]
    Shape of the data_s: list(torch.tensor([Batchsize, C, H, W]))
    Shape of the target_s: list(torch.tensor([Batchsize, C, H, W]))
    target_index: Select the channel index when there are more than one channel
    size: size of input image
    '''
    
    data_s = [F.interpolate(i, size=(size,size), mode="bilinear") for i in data_s]
    target_s = [F.interpolate(i, size=(size,size), mode="nearest") for i in target_s]
    
    T = target_index
    B,_,_,_ = data.shape
    images = data[:,T,:,:]
    support_images = torch.stack(data_s*B,dim=1)[:,:,T,:,:]
    support_labels = torch.stack(target_s*B,dim=1)
    
    # normalized support_images
    value,_ =support_images.min(axis=3,keepdim=True)
    min_,_ = value.min(axis=4,keepdim=True)
    value,_ =support_images.max(axis=3,keepdim=True)
    max_,_ = value.max(axis=4,keepdim=True)
    support_images = (support_images-min_)/(max_-min_)
    
    # normalized images
    value,_ =images.min(axis=2,keepdim=True)
    min_,_ = value.min(axis=3,keepdim=True)
    value,_ =images.max(axis=2,keepdim=True)
    max_,_ = value.max(axis=3,keepdim=True)
    images = (images-min_)/(max_-min_)
    
    images = F.interpolate(images, size=(size,size), mode="bilinear")
    return images, support_images, support_labels
def get_structured_data(data, data_s, target_s, target_index=0, size = 128):
    
    '''
    Shape of the data: [Batchsize, C, H, W]
    Shape of the data_s: list(torch.tensor([Batchsize, C, H, W]))
    Shape of the target_s: list(torch.tensor([Batchsize, C, H, W]))
    target_index: Select the channel index when there are more than one channel
    size: size of input image
    '''
    
    data_s = [F.interpolate(i, size=(size,size), mode="bilinear") for i in data_s]
    target_s = [F.interpolate(i, size=(size,size), mode="nearest") for i in target_s]
    
    T = target_index
    B,_,_,_ = data.shape
    images = data[:,T:T+1,:,:]
    support_images = torch.stack(data_s*B,dim=1)[:,:,T:T+1,:,:]
    support_labels = torch.stack(target_s*B,dim=1)
    
    # normalized support_images
    value,_ =support_images.min(axis=3,keepdim=True)
    min_,_ = value.min(axis=4,keepdim=True)
    value,_ =support_images.max(axis=3,keepdim=True)
    max_,_ = value.max(axis=4,keepdim=True)
    support_images = (support_images-min_)/(max_-min_)
    
    # normalized images
    value,_ =images.min(axis=2,keepdim=True)
    min_,_ = value.min(axis=3,keepdim=True)
    value,_ =images.max(axis=2,keepdim=True)
    max_,_ = value.max(axis=3,keepdim=True)
    images = (images-min_)/(max_-min_)
    
    images = F.interpolate(images, size=(size,size), mode="bilinear")
    return images, support_images, support_labels

'''
Compute the Embedding
'''
# from segment_anything import sam_model_registry,SamPredictor,SamPredictor_PerSAM
import numpy as np
import torch
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Target feature extraction
def compute_target_feat(ref_feat, ref_mask, verbose = False):
    target_feat = ref_feat[ref_mask > 0]
    if verbose:
        print('target_feat',target_feat.shape,type(target_feat))
    target_embedding = target_feat.mean(0).unsqueeze(0)
    if verbose:
        print('target_embedding',target_embedding.shape,type(target_embedding))
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    if verbose:
        print('target_feat',target_feat.shape,type(target_feat))
    target_embedding = target_embedding.unsqueeze(0)
    if verbose:
        print('target_embedding',target_embedding.shape,type(target_embedding))
    return target_embedding, target_feat

def compute_feature_embedding(ref_image, ref_mask, predictor):
    # Image features encoding
    ref_mask = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)
    # interpolate mask
    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]
    target_embedding,target_feat = compute_target_feat(ref_feat, ref_mask, verbose = False)
    background_embedding,background_feat = compute_target_feat(ref_feat, -1*ref_mask, verbose = False)
    return target_embedding,target_feat,background_embedding,background_feat

'''
Compute the Confidence map for both images and support_images
'''
def feat_cos_sim(test_feat, target_feat,verbose = False):
    # Cosine similarity
    C, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    if verbose:
        print('test_feat',test_feat.shape,type(test_feat))
        
    test_feat = test_feat.reshape(C, h * w)
    if verbose:
        print('test_feat',test_feat.shape,type(test_feat))
    sim = target_feat @ test_feat
    if verbose:
        print('sim',sim.shape,type(sim))

    sim = sim.reshape(1, 1, h, w)
    if verbose:
        print('sim',sim.shape,type(sim))
    return sim
def compute_L2_distance(test_feat, target_embedding):
    # compute the distance of two prototype
    distance = test_feat - target_embedding.permute(2,1,0)
    return torch.norm(distance,dim = 0)[None,None,:,:]

def postprocess_sim(sim, predictor, verbose = False):
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    if verbose:
        print('sim',sim.shape,type(sim))
    sim = predictor.model.postprocess_masks(
                    sim,
                    input_size=predictor.input_size,
                    original_size=predictor.original_size).squeeze()
    if verbose:
        print('sim',sim.shape,type(sim))
    return sim

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label
from skimage.morphology import remove_small_objects
def process_mask(mask, shrink_factor=0.8, min_component_size=100):
    # Step 1: Erosion to make it 20% smaller
    eroded_mask = binary_erosion(mask, structure=np.ones((3, 3))).astype(np.uint8)
    step=1
    while mask.sum()*shrink_factor<eroded_mask.sum()*1.0:
        eroded_mask = binary_erosion(eroded_mask, structure=np.ones((3, 3))).astype(np.uint8)
        step+=1
    # Step 2: Connected components analysis
    labeled_mask, num_components = label(eroded_mask)
    # Find the largest connected component
    component_sizes = np.bincount(labeled_mask.flatten())[1:]
    largest_component_label = np.argmax(component_sizes) + 1

    # Keep only the largest component
    largest_component_mask = (labeled_mask == largest_component_label).astype(np.uint8)

    # Step 3: Dilate back to the original size
    dilated_mask = largest_component_mask
    for _ in range(step):
        dilated_mask = binary_dilation(dilated_mask, structure=np.ones((3, 3))).astype(np.uint8)
    return dilated_mask

def compute_Cmap_single(test_feat, target_feat, background_feat,predictor,target_embedding,background_embedding):
    '''
    test_feat [256, 64, 64]
    target_feat [1, 256]
    background_feat [1, 256]
    sim_final [1,3,ori,ori]
    '''
    sim = feat_cos_sim(test_feat, target_feat)
    sim_background = feat_cos_sim(test_feat, background_feat)
    sim = postprocess_sim(sim,predictor)
    sim_background = postprocess_sim(sim_background,predictor)
    
    sim_final = sim.cpu()-sim_background.cpu()
    sim_final = sim_final[None,None,:,:].repeat(1,3,1,1)
    
#     dis = compute_L2_distance(test_feat, target_embedding)
#     dis_background = compute_L2_distance(test_feat, background_embedding)
#     dis = postprocess_sim(dis, predictor)
#     dis_background = postprocess_sim(dis_background,predictor)  
#     sim_final = dis_background.cpu()-dis.cpu()
#     sim_final = sim_final[None,None,:,:].repeat(1,3,1,1)
    return sim_final



# compute the confidence map for test
def Image2ConfidenceMap(test_image,predictor,target_feat,background_feat,target_embedding,background_embedding,verbose = False):
    '''
    test_image[1,3,ori_size,ori_size]
    target_feat [1, 256]
    background_feat [1, 256]
    sim_final [1,3,ori_size,ori_size]
    '''
    # Image feature encoding
    predictor.set_image(test_image[0,:].permute(1,2,0).numpy())
    test_feat = predictor.features.squeeze()
    if verbose:
        print('test_feat',test_feat.shape,type(test_feat))
    # Compute the confidence map
    confidence_map = compute_Cmap_single(test_feat, target_feat, background_feat,predictor,target_embedding,background_embedding)
    return confidence_map

'''
Inference
'''

def compute_dice_coefficient(predicted_mask, groundtruth_mask):
    intersection = np.sum(predicted_mask * groundtruth_mask)
    union = np.sum(predicted_mask) + np.sum(groundtruth_mask)
    
    dice_coefficient = (2.0 * intersection) / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    
    return dice_coefficient
def mask_fusion(high_res_masks, soft_pred,shape_,gamma):
    high_res_masks = high_res_masks/high_res_masks.std()
    soft_pred = (soft_pred-0.5)/soft_pred.std()
    soft_pred = F.interpolate(soft_pred, size=shape_, mode="bilinear")
    mask_final = (1-gamma)*high_res_masks[0]+gamma*soft_pred[0,0]
    return mask_final.cpu().detach().numpy()


'''
Process mask from UniverSeg
'''

def norm_(tmp):
    tmp = tmp.float()
    max_ = tmp.max()
    min_ = tmp.min()
    tmp = (tmp-min_)/(max_-min_)
    return tmp
def compute_bounding_box(segmentation_mask):
    '''
    input size: (H,W)
    '''
    # Find non-zero indices
    non_zero_indices = np.nonzero(segmentation_mask)

    # Compute bounding box
    min_y, min_x = np.min(non_zero_indices, axis=1)
    max_y, max_x = np.max(non_zero_indices, axis=1)

    return min_x, min_y, max_x, max_y


def augment_bounding_box(mask, jitter=True, shift=True, scale=True):
    '''
    mask: numpy array of shape (1,1,H,W)
    returns: augmented bbox (min_x, min_y, max_x, max_y)
    '''
    _, _, H, W = mask.shape
    seg = mask[0,0]

    # Step 1: Compute the base bounding box
    min_x, min_y, max_x, max_y = compute_bounding_box(seg)
    width = max_x - min_x
    height = max_y - min_y

    # Step 2: Jittering
    if jitter:
        jitter_range_x = width * 0.05
        jitter_range_y = height * 0.05
        min_x += int(np.random.uniform(-jitter_range_x*0.4, jitter_range_x))
        max_x += int(np.random.uniform(-jitter_range_x*0.4, jitter_range_x))
        min_y += int(np.random.uniform(-jitter_range_y*0.4, jitter_range_y))
        max_y += int(np.random.uniform(-jitter_range_y*0.4, jitter_range_y))

    # Step 3: Shifting
    if shift:
        shift_x = int(np.random.uniform(-width * 0.05, width * 0.05))
        shift_y = int(np.random.uniform(-height * 0.05, height * 0.05))
        min_x += shift_x
        max_x += shift_x
        min_y += shift_y
        max_y += shift_y

    # Step 4: Scaling (with 0.3 probability)
    if scale and np.random.rand() < 0.3:
        scale_x = np.random.uniform(0.95, 1.2)
        scale_y = np.random.uniform(0.95, 1.2)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        new_w = width * scale_x
        new_h = height * scale_y
        min_x = int(cx - new_w / 2)
        max_x = int(cx + new_w / 2)
        min_y = int(cy - new_h / 2)
        max_y = int(cy + new_h / 2)

    # Clamp to image boundaries
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(W - 1, max_x)
    max_y = min(H - 1, max_y)

    return min_x, min_y, max_x, max_y


def process_mask(mask, shrink_factor=0.9, min_component_size=100):
    # Step 1: Erosion to make it 20% smaller
    eroded_mask = binary_erosion(mask, structure=np.ones((3, 3))).astype(np.uint8)
    step=1
    while mask.sum()*shrink_factor<eroded_mask.sum()*1.0:
        eroded_mask = binary_erosion(eroded_mask, structure=np.ones((3, 3))).astype(np.uint8)
        step+=1
    # Step 2: Connected components analysis
    labeled_mask, num_components = label(eroded_mask)
    # Find the largest connected component
    component_sizes = np.bincount(labeled_mask.flatten())[1:]
    largest_component_label = np.argmax(component_sizes) + 1

    # Keep only the largest component
    largest_component_mask = (labeled_mask == largest_component_label).astype(np.uint8)

    # Step 3: Dilate back to the original size
    dilated_mask = largest_component_mask
    for _ in range(step):
        dilated_mask = binary_dilation(dilated_mask, structure=np.ones((3, 3))).astype(np.uint8)
    return dilated_mask

def process_mask_iter(mask,min_component_size=100,shrink_factor=0.9):
    # Step -1: Erosion to make it 20% smaller
    eroded_mask = binary_erosion(mask, structure=np.ones((3, 3))).astype(np.uint8)
    step=1
    while mask.sum()*shrink_factor<eroded_mask.sum()*1.0:
        eroded_mask = binary_erosion(eroded_mask, structure=np.ones((3, 3))).astype(np.uint8)
        step+=1
        
    dilated_mask = eroded_mask
    # Dilate back to the original size
    for _ in range(step):
        dilated_mask = binary_dilation(dilated_mask, structure=np.ones((3, 3))).astype(np.uint8)
    mask = dilated_mask
    
    # Step 1: Connected components analysis
    labeled_mask, num_components = label(mask)
    # Find the largest connected component
    component_sizes = np.bincount(labeled_mask.flatten())[1:]
    largest_component_label = np.argmax(component_sizes) + 1

    # Keep only the largest component
    largest_component_mask = (labeled_mask == largest_component_label).astype(np.uint8)
    
    # compute other mask
    all_large_component = np.where(component_sizes>0.05*component_sizes.max())[0]+1
    other_large_component = [i for i in all_large_component if i!= largest_component_label]
    if len(other_large_component) >0:
        other_mask = []
        for i in other_large_component:
            other_mask.append((labeled_mask == i).astype(np.uint8))
        return largest_component_mask, other_mask
    else:
        return largest_component_mask
    
    
# Logistic regression

# test_image_list, test_mask_list, predictor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def compute_logistic_regression(test_image_list, test_mask_list, predictor):
    Positive_feats = torch.zeros(0,256)
    Negative_feats = torch.zeros(0,256)
    for ref_image,ref_mask in zip(test_image_list, test_mask_list):
        ref_image = ref_image[0,:].permute(1,2,0).numpy()
        ref_mask = ref_mask[0,:].permute(1,2,0).repeat(1,1,3).numpy().astype('uint8')*255
        # Image features encoding
        ref_mask = predictor.set_image(ref_image, ref_mask)
        ref_feat = predictor.features.squeeze().permute(1, 2, 0)
        # interpolate mask
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="nearest")
        ref_mask = ref_mask.squeeze()[0]

        # obtain features
        index_tmp = ref_mask.view(-1)
        Positive_feat = ref_feat.view(-1,256)[index_tmp>0,:]
        Negative_feat = ref_feat.view(-1,256)[index_tmp<=0,:]

        Negative_feats = torch.cat([Negative_feats,Negative_feat.cpu()])
        Positive_feats = torch.cat([Positive_feats,Positive_feat.cpu()])
    
    # Generate sample data
    positive_samples = Positive_feats  # Assuming positive samples
    negative_samples = Negative_feats  # Assuming negative samples

    # Create labels (1 for positive, 0 for negative)
    positive_labels = torch.ones(Positive_feats.shape[0], 1)
    negative_labels = torch.zeros(Negative_feats.shape[0], 1)

    # Concatenate positive and negative samples and labels
    X = torch.cat([positive_samples, negative_samples], dim=0)
    y = torch.cat([positive_labels, negative_labels], dim=0).squeeze().numpy()
    # Down sample
    if len(X)>30000:
        random_indices = sorted(np.random.choice(len(X), size=30000, replace=False))
        X,y = X[random_indices],y[random_indices]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y, test_size=0.2, random_state=42)

    # Initialize logistic regression model
    model_LG = LogisticRegression(class_weight = 'balanced',solver = 'saga')

    # Train the model
    model_LG.fit(X_train, y_train)
    return model_LG

# compute the confidence map for test
def Image2ConfidenceMap_LG(test_image,predictor,model_LG,verbose = False):
    '''
    test_image[1,3,ori_size,ori_size]
    target_feat [1, 256]
    background_feat [1, 256]
    sim_final [1,3,ori_size,ori_size]
    '''
    # Image feature encoding
    predictor.set_image(test_image[0,:].permute(1,2,0).numpy())
    test_feat = predictor.features.squeeze().reshape(256,-1).T
    
    # Compute the confidence map
    confidence_map = model_LG.predict_proba(test_feat.cpu().numpy())[:,1]
    confidence_map = torch.tensor(confidence_map.reshape(64,64)[None, None, :])
    confidence_map = F.interpolate(confidence_map, size=test_image.shape[2:], mode="bilinear")
    if verbose:
        print('test_feat',test_feat.shape,type(test_feat))
        print('confidence_map',confidence_map.shape,type(confidence_map))
    # Compute the confidence map
    return confidence_map
