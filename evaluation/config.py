# config.py

import argparse
def parse_tuple(string):
    try:
        return tuple(map(int, string.strip('()').split(',')))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: {string}. Expected format: (x,y,z).")


def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameters for deep learning training")
    
    # SAM
    parser.add_argument('--sam_type', type=str, default='vit_b', help="sam_type")
    parser.add_argument('--sam_ckpt', type=str, default="../sam_vit_b_01ec64.pth", help="sam_ckpt")
    parser.add_argument('--data_dir_2D', type=str, default='Domain2/test/ROIs/image/', help='...')
    parser.add_argument('--json_name', type=str, default='Fundus_Domain2', help='...')
    parser.add_argument('--Thres', nargs='+', type=int, default=[0,128])
    
    # args for logging
    parser.add_argument('--validation_first', type=int, default=1)
    
    # load model
    parser.add_argument('--model_name', type=str, default='neuralizer', help="model")
    
    # args for training and evaluation
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decline_patience', type=int, default=10)
    parser.add_argument('--nb_inner_channels', nargs='+', type=int, default=[48, 96, 192, 384])
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--gradient_clip_val', type=float, default=2.5)
    parser.add_argument('--loss_seg', type=str, default='case_weight', help="case_weight or dice_sigmoid or dice_a01")
    parser.add_argument('--task_idx', type=int, default=0, help='Choose the task index.')
    parser.add_argument('--context_task_idx', type=int, default=None, help='Choose the task index for context.')
    parser.add_argument('--ell', type=int, default=3, help='Mini-context size.')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--note', type=str, default=' ')
    
    # args for fine-tunning
    
    # args for DDPM
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='linear')
    parser.add_argument('--min_snr_loss_weight', type=bool, default=False)
    
    
    # args for GAN
    parser.add_argument('--lambda_adv', type=float, default=0.05)
    parser.add_argument('--gan_inner_channels', nargs='+', type=int, default=[16, 32, 64, 128, 256])
    
    # data augmentation 
    parser.add_argument('--flip_prob', type=float, default=0.05, help='Prob. of fliping the seg mask.')
    parser.add_argument('--sobel_prob', type=float, default=0.05, help='Prob. of apply edge detection to the seg mask.')
    parser.add_argument('--gin_prob', type=float, default=0.05, help = 'Prob. of apply GIN augmentation to the images.')
    parser.add_argument('--save_train_images',type=bool, default=False)
    parser.add_argument('--train_aug', type=int, default=1)
    
    
    # dataloader
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--context_size', type=int, default=3, help='Context size for training')
    parser.add_argument('--workers', type=int, default=8, help='Workers of dataloaders.')
    parser.add_argument('--len_divid', type=int, default=50, help='...')
    parser.add_argument('--data_dir', type=str, default='Brain_ICL_2mm/', help='...')
    parser.add_argument('--skip_resize', type=bool, default=False, help='Skip the resizing of data')
    parser.add_argument('--random_context_size', type=bool, default=False, help='Use random context size')
    parser.add_argument('--train_or_val_split_rate', type=float, default=0.1, help='...')
    
    
    
    # load
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint') # './lightning_logs/version_37/checkpoints'
    parser.add_argument('--soft_checkpoint_path', type=str, default=None, help='Path to the soft checkpoint') # './lightning_logs/version_37/checkpoints'
    parser.add_argument('--checkpoint_index', type=int, default=-1, help='Index of checkpoint. (If you have multiples checkpoints.)')
    
    # Devices
    parser.add_argument('--device', type=str, default="cuda:0", help='Device')
    parser.add_argument('--train_gpus', type=parse_tuple, default='(1,4,6,7)')
    
    # Other Neuralizer hparameters
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--accumulate_grad_batches', type=int, default=8)
    parser.add_argument('--amp_backend', type=str, default='native')
    parser.add_argument('--auto_lr_find', type=bool, default=False)
    parser.add_argument('--auto_scale_batch_size', type=bool, default=False)
    parser.add_argument('--auto_select_gpus', type=bool, default=False)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--checkpoint_callback', default=None)
    parser.add_argument('--default_root_dir', default=None)
    parser.add_argument('--detect_anomaly', type=bool, default=False)
    parser.add_argument('--deterministic', default=None)
    parser.add_argument('--devices', default='auto')
    parser.add_argument('--enable_checkpointing', type=bool, default=True)
    parser.add_argument('--enable_model_summary', type=bool, default=True)
    parser.add_argument('--enable_progress_bar', type=bool, default=False)
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--gradient_clip_algorithm', default=None)
    parser.add_argument('--ipus', default=None)
    parser.add_argument('--limit_predict_batches', default=None)
    parser.add_argument('--limit_test_batches', default=None)
    parser.add_argument('--limit_train_batches', default=None)
    parser.add_argument('--limit_val_batches', default=None)
    parser.add_argument('--log_every_n_steps', type=int, default=500)
    parser.add_argument('--logger', type=bool, default=True)
    
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--max_time', default=None)
    parser.add_argument('--min_epochs', default=None)
    parser.add_argument('--min_steps', default=None)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--num_processes', default=None)
    parser.add_argument('--num_sanity_val_steps', type=int, default=2)
    parser.add_argument('--overfit_batches', type=float, default=0.0)
    parser.add_argument('--plugins', default=None)
    parser.add_argument('--process_position', type=int, default=0)
    parser.add_argument('--profiler', default=None)
    parser.add_argument('--reload_dataloaders_every_n_epochs', type=int, default=0)
    parser.add_argument('--replace_sampler_ddp', type=bool, default=True)
    parser.add_argument('--resume_from_checkpoint', default=None)
    parser.add_argument('--stochastic_weight_avg', type=bool, default=False)
    parser.add_argument('--strategy', default=None)
    parser.add_argument('--sync_batchnorm', type=bool, default=False)
    parser.add_argument('--terminate_on_nan', default=None)
    parser.add_argument('--tpu_cores', default=None)
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser.add_argument('--val_check_interval', default=None)
    parser.add_argument('--weights_save_path', default=None)
    parser.add_argument('--weights_summary', default='top')
    parser.add_argument('--accumulate_grad_batches_rampup_epochs', type=int, default=0)
    parser.add_argument('--context_informed_uncertainty_ok', type=int, default=1)
    parser.add_argument('--data_downsample_factor', type=int, default=1)
    parser.add_argument('--data_num_workers', type=int, default=16)
    parser.add_argument('--data_slice_only', type=bool, default=False)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--flush_logs_every_n_steps', default=None)
    parser.add_argument('--init_weights_from', default=None)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--log_gpu_memory', default=None)
    parser.add_argument('--max_context_size', type=int, default=32)
    parser.add_argument('--min_context_size', type=int, default=1)
    parser.add_argument('--model', default='pairwise_conv_avg_model')
    parser.add_argument('--move_metrics_to_cpu', type=bool, default=False)
    parser.add_argument('--multiple_trainloader_mode', default='max_size_cycle')
    parser.add_argument('--nb_conv_layers_per_stage', type=int, default=2)
    parser.add_argument('--nb_levels', type=int, default=4)
    parser.add_argument('--noeval', type=bool, default=False)
    parser.add_argument('--not_trained_datasets', default=None)
    parser.add_argument('--not_trained_modalities', default=None)
    parser.add_argument('--not_trained_seg_classes', default=None)
    parser.add_argument('--not_trained_tasks', default=None)
    parser.add_argument('--only_dataset', default=None)
    parser.add_argument('--only_modality', default=None)
    parser.add_argument('--only_seg_classes', default=None)
    parser.add_argument('--only_task', default=None)
    parser.add_argument('--progress_bar_refresh_rate', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--sample_every_n_hours', type=int, default=6)
    parser.add_argument('--spatial_augmentation_mode', default='context_aligned')
    parser.add_argument('--tags', nargs='+', default=['all_data'])
    parser.add_argument('--train_unique_query_cnt', type=int, default=-1)
    parser.add_argument('--train_unique_query_seed', default=None)
    parser.add_argument('--weight_decay', type=float, default=0)

    return parser


"""
Configuration of dataloader
task abbreviation:
    Seg: Segmentation
    ModTran: Modality Transfer
    SupRes: Super Resolution
    Bias: Bias Field Denoising
    Denoi: Denoising
    Inp: Inpainting
    2D23D: 2D to 3D image
"""

data_loading_config_meta = {
# 0 Tumor T1
    'Tumor_T1':[
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':1,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2,3]},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':2,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2,3]},'sample rate':1, 'weight': 1},
    ],
# 1 Tumor T2, FLAIR
    'Tumor_T2':[
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2,3]},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2,3]},'sample rate':1, 'weight': 1},
    ],
# 2 vascular
    'Topcow':[
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},
    ],
# 3 Cerebral Cortex
    'Cerebral_Cortex':[
    {'name':'Dataset613_OASIS_Freesurfer-35', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [2,21]},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,42]},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,42]},'sample rate':1, 'weight': 1},
    ],
# 4 Hippo
    'Hippo':[
    {'name':'Dataset613_OASIS_Freesurfer-35', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [14,30]},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [17,53]},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [17,53]},'sample rate':1, 'weight': 1},
    ],
# 5 Thalamus
    'Thalamus':[
    {'name':'Dataset613_OASIS_Freesurfer-35', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [7,26]},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,49]},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,49]},'sample rate':1, 'weight': 1},
    ],
            
# 6 Lateral Ventricle
    'Lateral_Ventricle':[
    {'name':'Dataset613_OASIS_Freesurfer-35', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,22]},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,43]},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,43]},'sample rate':1, 'weight': 1},
    ],
            
# 7 Putamen
    'Putamen':[
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [12,51]},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [12,51]},'sample rate':1, 'weight': 1},
    ],
            
# 8 Amygdala
    'Amygdala':[
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [18,54]},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [18,54]},'sample rate':1, 'weight': 1},
    ],
            
## 9 Modality_transform
    'Modality_transform':[
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':0,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    ],
        
## 10 Bias remove
    'Bias_remove':[
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset602_ADNI', 'input':2,'output':2,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset620_adhd_t1', 'input':1,'output':1,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':2,'output':2,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':3,'output':3,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':1,'output':1,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':2,'output':2,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    ],
            
## 11 Gaussian noise remove
    'Gaussian_noise':[
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset620_adhd_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':2,'output':2, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    ],
        
## 12 salt & pepper noise remove
    'salt_pepper_noise':[
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset620_adhd_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':2,'output':2, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':2,'output':2, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    ],
            
## 13 2D to 3D
    '2Dto3D':[
    {'name':'Dataset626_gsp_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64},'sample rate':1,'weight': 1},

    {'name':'Dataset635_ukb_t1_2000', 'input':3,'output':3, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64},'sample rate':1,'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':3,'output':3, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64},'sample rate':1,'weight': 1},
    ],
## 14 Inpainting
    'Inpainting':[
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset620_adhd_t1', 'input':3,'output':3, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':1,'output':1, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':2,'output':2, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':3,'output':3, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    ],
            
    
## 15 Superresolution
    'Superresolution':[
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},    
    {'name':'Dataset620_adhd_t1', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':1,'output':1, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':2,'output':2, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':3,'output':3, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    ],
          
    
## 16 Skull Stripping
    'Skull_Stripping':[
    {'name':'Dataset620_adhd_t1', 'input':1,'output':2,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':1,'output':2,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':1,'output':2,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':1,'output':2,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':1,'output':2,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':1,'output':2,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    ],
        
## 17 Real denoising
    'Real_denoising':[
    {'name':'Dataset620_adhd_t1', 'input':2,'output':3,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':2,'output':3,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':2,'output':3,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    ],
    
## Segmentation
# 18 Cerebral Cortex
    'Cerebral_Cortex_heldout':[
    {'name':'Dataset725_fcon1000_t1_freesurfer', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,42]},'sample rate':1, 'weight': 1},   
    {'name':'Dataset825_fcon1000_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,42]},'sample rate':1, 'weight': 1},
    ],
# 19 Hippo
    'Hippo_heldout':[
    {'name':'Dataset725_fcon1000_t1_freesurfer', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [17,53]},'sample rate':1, 'weight': 1},
    {'name':'Dataset825_fcon1000_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [17,53]},'sample rate':1, 'weight': 1},
    ],
# 20 Thalamus
    'Thalamus_heldout':[
    {'name':'Dataset825_fcon1000_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,49]},'sample rate':1, 'weight': 1},
    {'name':'Dataset725_fcon1000_t1_freesurfer', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,49]},'sample rate':1, 'weight': 1},
    ],
            
# 21 Lateral Ventricle
    'Lateral_Ventricle_heldout':[
    {'name':'Dataset725_fcon1000_t1_freesurfer', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,43]},'sample rate':1, 'weight': 1},
    {'name':'Dataset825_fcon1000_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,43]},'sample rate':1, 'weight': 1},
    ],
            
# 22 Putamen
    'Putamen_heldout':[
    {'name':'Dataset725_fcon1000_t1_freesurfer', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [12,51]},'sample rate':1, 'weight': 1},
    {'name':'Dataset825_fcon1000_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [12,51]},'sample rate':1, 'weight': 1},
    ],
            
# 23 Amygdala
    'Amygdala_heldout':[
    {'name':'Dataset725_fcon1000_t1_freesurfer', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [18,54]},'sample rate':1, 'weight': 1},
    {'name':'Dataset825_fcon1000_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [18,54]},'sample rate':1, 'weight': 1},
    ],
        
## Generation
## 24 Bias remove
    'Bias_remove_heldout':[
    {'name':'Dataset623_ccnp_t1', 'input':1,'output':1,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset625_fcon1000_t1', 'input':1,'output':1,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset631_ppmi_t1', 'input':2,'output':2,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},

    ],
            
## 25 Gaussian noise remove
    'Gaussian_noise_heldout':[
    {'name':'Dataset623_ccnp_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    {'name':'Dataset625_fcon1000_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.1},'sample rate':1, 'weight': 1},
    {'name':'Dataset631_ppmi_t1', 'input':2,'output':2, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25},'sample rate':1, 'weight': 1},
    ],
        
## 26 salt & pepper noise remove
    'salt_pepper_noise_heldout':[
    {'name':'Dataset623_ccnp_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset625_fcon1000_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    {'name':'Dataset631_ppmi_t1', 'input':2,'output':2, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    ],

            
## 27 2D to 3D
    '2Dto3D_heldout':[
    {'name':'Dataset623_ccnp_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64},'sample rate':1,'weight': 1},
    {'name':'Dataset631_ppmi_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64},'sample rate':1,'weight': 1},
    {'name':'Dataset625_fcon1000_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64},'sample rate':1,'weight': 1},
    ],
        
## 28 Inpainting
    'Inpainting_heldout_heldout':[
    {'name':'Dataset623_ccnp_t1', 'input':2,'output':2, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset631_ppmi_t1', 'input':2,'output':2, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset625_fcon1000_t1', 'input':1,'output':1, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    ],
            
    
## 29 Superresolution
    'Superresolution_heldout':[
    {'name':'Dataset623_ccnp_t1', 'input':2,'output':2, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset631_ppmi_t1', 'input':2,'output':2, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset625_fcon1000_t1', 'input':1,'output':1, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    ],
          
    
## 30 Skull Stripping
    'Skull_Stripping_heldout':[
    {'name':'Dataset623_ccnp_t1', 'input':1,'output':2,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset625_fcon1000_t1', 'input':1,'output':2,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset631_ppmi_t1', 'input':1,'output':2,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    ],
    
## 31 Modality transform
    'Modality_transform_heldout':[
    {'name':'Dataset606_WMH_Seg_Challenge', 'input':1,'output':0,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset606_WMH_Seg_Challenge', 'input':0,'output':1,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset606_WMH_Seg_Challenge', 'input':1,'output':0,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset606_WMH_Seg_Challenge', 'input':0,'output':1,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset606_WMH_Seg_Challenge', 'input':1,'output':0,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset606_WMH_Seg_Challenge', 'input':0,'output':1,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    ],
    
## 32 Synthetic data Seg
    'Synthetic_Seg_heldout':[{'name':'Dataset967_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,52]},'sample rate':0.1, 'weight': 1} for i in range(10)]+\
    [{'name':'Dataset967_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [5,53]},'sample rate':0.1, 'weight': 1} for i in range(10)]+\
    [{'name':'Dataset967_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,57]},'sample rate':0.1, 'weight': 1} for i in range(10)],

## 33 Synthetic data Modality transform
    'Synthetic_Mod_heldout':[{'name': 'Dataset967_syn_multimodal_brain', 'input': j, 'output': i, 'task': 'ModTran', 'task_config': [], 'sample rate': 0.03, 'weight': 1} for i in range(2) for j in range(8,10)],
    
## 34 FLARE22_seg_1
    'FLARE22_seg_1':[{'name':'Dataset670_FLARE22_LabeledCase50', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1}],
## 35 FLARE22_seg_2
    'FLARE22_seg_2':[{'name':'Dataset670_FLARE22_LabeledCase50', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [2]},'sample rate':1, 'weight': 1}],
## 36 FLARE22_seg_13
    'FLARE22_seg_13':[{'name':'Dataset670_FLARE22_LabeledCase50', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [13]},'sample rate':1, 'weight': 1}],
    
## 37 EPISURG
    'EPISURG':[{'name':'Dataset669_EPISURG', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1}],
    
## 38 Synthetic abdomen data Seg
    'Synthetic_abdomen_Seg':[{'name':'Dataset968_syn_multimodal_abdominal', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':0.1, 'weight': 1} for i in range(10)]+\
    [{'name':'Dataset968_syn_multimodal_abdominal', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [2,3]},'sample rate':0.1, 'weight': 1} for i in range(10)]+\
    [{'name':'Dataset968_syn_multimodal_abdominal', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [5]},'sample rate':0.1, 'weight': 1} for i in range(10)],

## 39 FLARE22_seg_1
    'FLARE22_resample_seg_1':[{'name':'Dataset671_FLARE22_LabeledCase50', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1}],
## 40 FLARE22_seg_2
    'FLARE22_resample_seg_2':[{'name':'Dataset671_FLARE22_LabeledCase50', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [2]},'sample rate':1, 'weight': 1}],
## 41 FLARE22_seg_13
    'FLARE22_resample_seg_13':[{'name':'Dataset671_FLARE22_LabeledCase50', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [13]},'sample rate':1, 'weight': 1}],
## 42 motion correction
    'motion_correction':[
        {'name':'Dataset623_ccnp_t1', 'input':7,'output':3,'task':'ModTran','task_config':{},'sample rate':1, 'weight': 1},
        {'name':'Dataset625_fcon1000_t1', 'input':7,'output':4,'task':'ModTran','task_config':{},'sample rate':1, 'weight': 1},
        {'name':'Dataset631_ppmi_t1', 'input':7,'output':1,'task':'ModTran','task_config':{},'sample rate':1, 'weight': 1},
    ],
    
## 43 undersample reconstruction
    'undersample_reconstruction':[
        {'name':'Dataset623_ccnp_t1', 'input':8,'output':3,'task':'ModTran','task_config':{},'sample rate':1, 'weight': 1},
        {'name':'Dataset625_fcon1000_t1', 'input':8,'output':4,'task':'ModTran','task_config':{},'sample rate':1, 'weight': 1},
        {'name':'Dataset631_ppmi_t1', 'input':8,'output':1,'task':'ModTran','task_config':{},'sample rate':1, 'weight': 1},
    ],
## 44  Nasal class 1,2
    'NasalSeg_1_2':[
         {'name':'Dataset675_NasalSeg', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2]},'sample rate':1, 'weight': 1},
    ],
    
## 45  Nasal class 3,4
    'NasalSeg_3_4':[
         {'name':'Dataset675_NasalSeg', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,4]},'sample rate':1, 'weight': 1},
    ],
    
## 46  Nasal class 5
    'NasalSeg_5':[
         {'name':'Dataset675_NasalSeg', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [5]},'sample rate':1, 'weight': 1},
    ],
    
## 47  PROMISE12 prostate seg.
    'PROMISE12':[
         {'name':'Dataset674_PROMISE12', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},
    ],
    
## 48  Rat lung seg.
    'Rat_4':[
         {'name':'Dataset672_Rat', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4]},'sample rate':1, 'weight': 1},
    ],
    
## 49  Rat lung seg.
    'Rat_12':[
         {'name':'Dataset672_Rat', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [12]},'sample rate':1, 'weight': 1},
    ],
    
## 50  Rat lung seg.
    'Rat_1':[
         {'name':'Dataset672_Rat', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},
    ],
    
## 51  Heart
    'MSD_Heart':[
             {'name':'Dataset901_MSD_Heart', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},
    ],

## 52  Spleen
    'FLARE22_resample_seg_3':[{'name':'Dataset671_FLARE22_LabeledCase50', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3]},'sample rate':1, 'weight': 1}],
## 53  Pancreas
    'FLARE22_resample_seg_4':[{'name':'Dataset671_FLARE22_LabeledCase50', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4]},'sample rate':1, 'weight': 1}],

## 54 Pet Seg
    'Pet_Seg':[
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},   
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [2]},'sample rate':1, 'weight': 1},   
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3]},'sample rate':1, 'weight': 1},   
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4]},'sample rate':1, 'weight': 1},   
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [5]},'sample rate':1, 'weight': 1},   
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [6]},'sample rate':1, 'weight': 1},   
    ],
    
## 55 Pet Seg Cerebral_Cortex
    'Pet_Seg_1':[
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},   
    ],
## 56 Pet Seg Lateral_Ventricle
    'Pet_Seg_2':[
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [2]},'sample rate':1, 'weight': 1},   
    ],
## 57 Pet Seg Putamen
    'Pet_Seg_3':[
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3]},'sample rate':1, 'weight': 1},   
    ],
## 58 Pet Seg Thalamus_Proper
    'Pet_Seg_4':[
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4]},'sample rate':1, 'weight': 1},   
    ],
## 59 Pet Seg Hippocampus
    'Pet_Seg_5':[
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [5]},'sample rate':1, 'weight': 1},   
    ],
## 60 Pet Seg Amygdala
    'Pet_Seg_6':[
    {'name':'Dataset602_ADNI', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [6]},'sample rate':1, 'weight': 1},   
    ],
    
## 61 RAOS_CT_128: liver
    'RAOS_CT_128_1':[
    {'name':'Dataset683_RAOS_Real_CancerImages_S1_128', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},   
    ],
    
## 62 RAOS_CT_128: kidney
    'RAOS_CT_128_3_4':[  
    {'name':'Dataset683_RAOS_Real_CancerImages_S1_128', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,4]},'sample rate':1, 'weight': 1},   
    ],
    
## 63 RAOS_CT_128: stomach
    'RAOS_CT_128_5':[
    {'name':'Dataset683_RAOS_Real_CancerImages_S1_128', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [5]},'sample rate':1, 'weight': 1},   
    ],
    
## 64 Dataset680_AMOS22_CT_128: liver
    'Dataset680_AMOS22_CT_128_6':[
    {'name':'Dataset680_AMOS22_CT_128', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [6]},'sample rate':1, 'weight': 1},           ],
    
## 65 Dataset680_AMOS22_CT_128: kidney
    'Dataset680_AMOS22_CT_128_2_3':[
    {'name':'Dataset680_AMOS22_CT_128', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [2,3]},'sample rate':1, 'weight': 1},   
    ],

    
## 66 Dataset680_AMOS22_CT_128: stomach
    'Dataset680_AMOS22_CT_128_7':[
    {'name':'Dataset680_AMOS22_CT_128', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [7]},'sample rate':1, 'weight': 1},   
    ],
    
}


'''
from config import data_loading_config_meta

task_names = list(data_loading_config_meta.keys())
task_name = task_names[args.task_idx]
data_loading_config = data_loading_config_meta[task_name]
print('All task:',task_names)
print('Length :',len(task_names))
print('The current task is:', task_name)

'''

data_loading_config_meta_CrossMod = {

## 0 Bias remove T1
    'T1_Bias':[
            {'name':'Dataset620_adhd_t1', 'input':1,'output':1, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    ],
## 1 Bias remove T2
    'T2_Bias':[
            {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    ],
    
## 2 gaussian CT
    'CT_Bias':[
            {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    ],
    
## 3 gaussian T1
    'T1_gaussian':[
            {'name':'Dataset620_adhd_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.10},'sample rate':1, 'weight': 1},
    ],
## 4 gaussian T2
    'T2_gaussian':[
            {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.10},'sample rate':1, 'weight': 1},
    ],
    
## 5 gaussian CT
    'CT_gaussian':[
            {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.10}, 'sample rate':1, 'weight': 1},
    ],
    
    
## 6 salt & pepper noise remove T1
    'T1_saltpepper':[
            {'name':'Dataset620_adhd_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    ],
## 7 salt & pepper noise remove T2
    'T2_saltpepper':[
            {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04},'sample rate':1, 'weight': 1},
    ],
    
## 8 salt & pepper noise remove CT
    'CT_saltpepper':[
            {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04}, 'sample rate':1, 'weight': 1},
    ],
    
## 9 Inpainting T1
    'T1_Inpain':[
            {'name':'Dataset620_adhd_t1', 'input':1,'output':1, 'task':'Denoi', 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    ],
## 10 Inpainting T2
    'T2_Inpain':[
            {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Denoi', 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    ],
    
## 11 Inpainting CT
    'CT_Inpain':[
            {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Denoi', 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)}, 'sample rate':1, 'weight': 1},
    ],
    
## 12 Superresolution T1
    'T1_Supres':[
            {'name':'Dataset620_adhd_t1', 'input':1,'output':1, 'task':'Denoi', 'task':'SupRes','task_config':{'factor':2}, 'sample rate':1, 'weight': 1},
    ],
## 13 Superresolution T2
    'T2_Supres':[
            {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Denoi', 'task':'SupRes','task_config':{'factor':2}, 'sample rate':1, 'weight': 1},
    ],
    
## 14 Superresolution CT
    'CT_Supres':[
            {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Denoi', 'task':'SupRes','task_config':{'factor':2},  'sample rate':1, 'weight': 1},
    ],
    
}


data_loading_config = [
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':1,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2,3], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2,3], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':2,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2,3], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2,3], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
#     {'name':'Dataset607_ICH_CT', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [254]},'sample rate':1, 'weight': 1},
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':3, 'weight': 1},
#     {'name':'Dataset603_Task007_VALDO', 'input':1,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},
#     {'name':'Dataset604_Task008_VALDO', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},
#     {'name':'Dataset605_Task009_VALDO', 'input':1,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},
    {'name':'Dataset606_WMH_Seg_Challenge', 'input':1,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1,2], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':3, 'weight': 1},
#     {'name':'Dataset609_ISLES2022', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},
    {'name':'Dataset610_ATLAS', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    
    {'name':'Dataset612_OASIS_Freesurfer-04', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset612_OASIS_Freesurfer-04', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [2], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset612_OASIS_Freesurfer-04', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    
    {'name':'Dataset613_OASIS_Freesurfer-35', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [2,21], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset613_OASIS_Freesurfer-35', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,22], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset613_OASIS_Freesurfer-35', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [7,26], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset613_OASIS_Freesurfer-35', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [14,30], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset613_OASIS_Freesurfer-35', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    
    {'name':'Dataset727_hab_t1_freesurfer', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    {'name':'Dataset729_nimh_t1_freesurfer', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,42], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,43], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [12,51], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,49], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [17,53], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [18,54], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset827_hab_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,42], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,43], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [12,51], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,49], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [17,53], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [18,54], 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset829_nimh_t1_freesurfer_raw', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':2, 'weight': 1},
    
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset636_ukb_t1_2000', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random', 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    

#     {'name':'Dataset616_Canada_Hippo25', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [1]},'sample rate':1, 'weight': 1},

## Modality transform
    {'name':'Dataset602_ADNI', 'input':2,'output':0,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset602_ADNI', 'input':0,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
#     {'name':'Dataset606_WMH_Seg_Challenge', 'input':1,'output':0,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
#     {'name':'Dataset606_WMH_Seg_Challenge', 'input':0,'output':1,'task':'ModTran','task_config':[],'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':0,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':1,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':1,'output':3,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':1,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':0,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':1,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':2,'output':1,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':3,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':2,'output':3,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':1,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':1,'output':0,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    

## Abdominal
# {'name':'Dataset670_FLARE22_LabeledCase50', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True},'sample rate':3, 'weight': 1},
# {'name':'Dataset670_FLARE22_LabeledCase50', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':3, 'weight': 1},
# {'name':'Dataset670_FLARE22_LabeledCase50', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':3, 'weight': 1},

## Bias remove
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset602_ADNI', 'input':2,'output':2,'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1},'sample rate':1, 'weight': 1},
    {'name':'Dataset620_adhd_t1', 'input':1,'output':1,'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':2,'output':2,'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':3,'output':3,'task':'Bias','task_config':{'coeff_range' :(0.3, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':1,'output':1,'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':2,'output':2,'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'Bias','task_config':{'coeff_range' :(-0.5, 0.5) ,'prob' :1, 'task_aug':{'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    
#     {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1},
#     {'name':'Dataset620_adhd_t1', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1},
#     {'name':'Dataset621_adni_t1', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1},
#     {'name':'Dataset622_camcan_t1', 'input':0,'output':0,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1},
#     {'name':'Dataset621_adni_t1', 'input':2,'output':2,'task':'Bias','task_config':{'coeff_range' :(0.3,0.5) ,'prob' :1},'sample rate':1},

## Gaussian noise remove
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset620_adhd_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':2,'output':2, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"gaussian" , 'std':0.25, 'random':True},'sample rate':1, 'weight': 1},


## salt & pepper noise remove
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1} },'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1} },'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset620_adhd_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':2,'output':2, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':1,'output':1, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':2,'output':2, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'Denoi', 'task_config': {'noise_type' :"salt_pepper" , 'salt_prob':0.04, 'pepper_prob':0.04, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':1, 'weight': 1},
    
## 2D to 3D
    {'name':'Dataset626_gsp_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':3,'output':3, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':3,'output':3, 'task':'2D23D', 'task_config': {'axis': 2, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
      {'name':'Dataset626_gsp_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 1, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 1, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':3,'output':3, 'task':'2D23D', 'task_config': {'axis': 1, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':3,'output':3, 'task':'2D23D', 'task_config': {'axis': 1, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
      {'name':'Dataset626_gsp_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 0, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':0,'output':0, 'task':'2D23D', 'task_config': {'axis': 0, 'slice_index' : 64, 'random':True, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1,'Inp':0.1}},'sample rate':0.2,'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':3,'output':3, 'task':'2D23D', 'task_config': {'axis': 0, 'slice_index' : 64, 'random':True},'sample rate':0.2,'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':3,'output':3, 'task':'2D23D', 'task_config': {'axis': 0, 'slice_index' : 64, 'random':True},'sample rate':0.2,'weight': 1},


    
# Inpainting
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset620_adhd_t1', 'input':3,'output':3, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':1,'output':1, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':2,'output':2, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':3,'output':3, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4), 'task_aug':{'Bias':0.1,'Denoi': 0.2,'SupRes':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'Inp','task_config':{'scales': [8, 16], 'thresholds' : (0.1, 0.4)},'sample rate':1, 'weight': 1},
    
    
## Superresolution
    {'name':'Dataset601_TopCow_MRA', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset608_CAS2023', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset609_ISLES2022', 'input':1,'output':1, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset611_IXI', 'input':3,'output':3, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset615_ICH_unlabeled', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},    
    {'name':'Dataset620_adhd_t1', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':1,'output':1, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':2,'output':2, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':3,'output':3, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset635_ukb_t1_2000', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset637_ukb_t2_2000', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2, 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':0,'output':0, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    {'name':'Dataset668_BraTS2021_MR_1251', 'input':3,'output':3, 'task':'SupRes','task_config':{'factor':2},'sample rate':1, 'weight': 1},
    
    
## Skull Stripping
    {'name':'Dataset620_adhd_t1', 'input':1,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':1,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':1,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':1,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':1,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':1,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':1,'output':2,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},

## Real denoising
    {'name':'Dataset620_adhd_t1', 'input':2,'output':3,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset621_adni_t1', 'input':2,'output':3,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset624_cmi_t1', 'input':2,'output':3,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset626_gsp_t1', 'input':2,'output':3,'task':'ModTran','task_config':{ 'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset627_hab_t1', 'input':2,'output':3,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset629_nimh_t1', 'input':2,'output':3,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},
    {'name':'Dataset630_oasis_t1', 'input':2,'output':3,'task':'ModTran','task_config':{'task_aug':{'Bias':0.1,'Denoi': 0.2,'Inp':0.1}},'sample rate':1, 'weight': 1},

## Registration
#     {'name':'Dataset620_adhd_t1', 'input':1,'output':4,'task':'ModTran','task_config':[],'sample rate':1},
    
# # Synthetic tasks
        ]+\
[{'name':f'Dataset{960+i}_syn_type3', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(0,4)]+\
[{'name':f'Dataset{900+i}_syn_type1', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.05, 'weight': 1} for i in range(0,30)]+\
[{'name':f'Dataset{900+i}_syn_type2', 'input':0,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.05, 'weight': 1} for i in range(30,60)]+\
[{'name':'Dataset964_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,51]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset964_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,52]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset964_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [5,53]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset964_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,57]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset964_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [11,58]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset964_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name': 'Dataset964_syn_multimodal_brain', 'input': j, 'output': i, 'task': 'ModTran', 'task_config': [], 'sample rate': 0.02, 'weight': 1} for i in range(10) for j in range(10)]+\
[{'name':'Dataset965_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,51]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset965_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,52]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset965_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [5,53]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset965_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,57]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset965_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [11,58]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset965_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name': 'Dataset965_syn_multimodal_brain', 'input': j, 'output': i, 'task': 'ModTran', 'task_config': [], 'sample rate': 0.02, 'weight': 1} for i in range(10) for j in range(10)]+\
[{'name':'Dataset966_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [3,51]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset966_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [4,52]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset966_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [5,53]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset966_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [10,57]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset966_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': [11,58]},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name':'Dataset966_syn_multimodal_brain', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.05, 'weight': 1} for i in range(10)]+\
[{'name': 'Dataset966_syn_multimodal_brain', 'input': j, 'output': i, 'task': 'ModTran', 'task_config': [], 'sample rate': 0.02, 'weight': 1} for i in range(10) for j in range(10)]
