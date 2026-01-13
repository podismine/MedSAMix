import gc
import json
import time
import os
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
import copy
import numpy as np
import torch
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
)
from smac import (
    MultiFidelityFacade,
    Scenario,
    RunHistory,
)
from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity

from evaluation.evaluate_main import evaluate_main
from utils import logger
from .base_strategy import MergeStrategy
from .merge_utils import SAMLayerWiseMerger, save_layer_config
from per_segment_anything import sam_model_registry as per_sam_model_registry

SUPPORTED_METHOD_PARAM_MAPS = {
    "linear": ["weights"],
    "task_arithmetic": ["scaling_coefficient"],
    "ties": ["param_value_mask_rate", "scaling_coefficient"],
    "slerp": ["slerp_t"]
}

COMPONENT_TYPES = ["attention", "mlp", "norm"]

class ViTLfsMerge(MergeStrategy):
    """ViT Large Fidelity Search Merge Strategy with component-level optimization"""
    
    def __init__(self, config):
        super().__init__(config)
        self.models = self.config["models"]
        self.merging_method = self.config["merging_method"]
        self.load_run_history = self.config.get("load_run_history", None)
        self.random_init_points = self.config.get("random_init_points", 0)
        self.base_model = self.config["base_model"]
        self.layer_granularity = self.config["layer_granularity"]
        self.component_granularity = self.config.get("component_granularity", "unified")  # "unified" or "separate"
        self.n_trials = self.config["n_trials"]
        self.min_budget = self.config.get("min_budget", 20)
        self.max_budget = self.config.get("max_budget", 1000)
        self.total_budget = self.config.get("total_budget")
        self.eta = self.config.get("eta", 3)
        self.n_workers = self.config.get("n_workers", 4)
        
        self.output_path = self.config.get("output_path", None)
        
        # Evaluation setup
        self.evaluate_tasks = [task['task'] for task in self.config.get('evaluation', {}).get('tasks', [])]
        
        # evaluator_key = 'ondisk_evaluate'
        # self.evaluator_class = evaluator_classes[evaluator_key]
        # self.evaluator_instance = self.evaluator_class(self.config)
        
        # Initialize SAM merger
        self.sam_base_model = per_sam_model_registry['vit_b'](checkpoint=self.base_model)
        self.sam_merger = SAMLayerWiseMerger(self.sam_base_model)
        
        # SAM Architecture Analysis
        self.sam_architecture = self._analyze_sam_architecture()
        
        # Vision Encoder (ViT) layers - this is what we primarily merge with granularity
        self.vision_encoder_layers = self.sam_merger.total_layers  # ViT transformer layers
        
        # Handle layer granularity ONLY for Vision Encoder
        if self.layer_granularity > 0:
            self.num_groups = max(1, self.vision_encoder_layers // self.layer_granularity)
            # Allow for remainder layers in the last group
            self.layer_groups = self._create_layer_groups()
        else:
            # Treat entire vision encoder as one group
            self.num_groups = 1
            self.layer_groups = [(0, self.vision_encoder_layers)]
        
        # Define Prompt Encoder and Mask Decoder groups (unified mode only)
        self.prompt_encoder_groups = self._define_prompt_encoder_groups()
        self.mask_decoder_groups = self._define_mask_decoder_groups()
        
        logger.info(f"Initialized ViT LFS Merge:")
        logger.info(f"  SAM Vision Encoder: {self.vision_encoder_layers} transformer layers")
        logger.info(f"  Vision Encoder groups: {self.num_groups} (granularity: {self.layer_granularity})")
        logger.info(f"  Prompt Encoder groups: {len(self.prompt_encoder_groups)} groups")
        logger.info(f"  Mask Decoder groups: {len(self.mask_decoder_groups)} groups")
        logger.info(f"  Total new parameter groups: {len(self.prompt_encoder_groups) + len(self.mask_decoder_groups)}")
        logger.info(f"  Component granularity: {self.component_granularity}")
    
    def _define_prompt_encoder_groups(self) -> List[Dict[str, Any]]:
        """Define prompt encoder groups - 4 separate groups."""
        return [
            {
                'name': 'point_embeddings',
                'components': ['point_embeddings.0', 'point_embeddings.1', 'point_embeddings.2', 'point_embeddings.3'],
                'description': '4个点嵌入层'
            },
            {
                'name': 'not_a_point_embed',
                'components': ['not_a_point_embed'],
                'description': '背景点嵌入'
            },
            {
                'name': 'mask_downscaling',
                'components': ['mask_downscaling.0', 'mask_downscaling.1', 'mask_downscaling.3', 
                              'mask_downscaling.4', 'mask_downscaling.6'],
                'description': '7个卷积下采样层'
            },
            {
                'name': 'no_mask_embed',
                'components': ['no_mask_embed'],
                'description': '无掩码嵌入'
            }
        ]
    
    def _define_mask_decoder_groups(self) -> List[Dict[str, Any]]:
        """Define mask decoder groups - 6 separate groups."""
        return [
            {
                'name': 'transformer_layers',
                'components': [
                    # Layer 0 components
                    'transformer.layers.0.self_attn.q_proj', 'transformer.layers.0.self_attn.k_proj', 'transformer.layers.0.self_attn.v_proj', 'transformer.layers.0.self_attn.out_proj',
                    'transformer.layers.0.cross_attn_token_to_image.q_proj', 'transformer.layers.0.cross_attn_token_to_image.k_proj', 'transformer.layers.0.cross_attn_token_to_image.v_proj', 'transformer.layers.0.cross_attn_token_to_image.out_proj',
                    'transformer.layers.0.cross_attn_image_to_token.q_proj', 'transformer.layers.0.cross_attn_image_to_token.k_proj', 'transformer.layers.0.cross_attn_image_to_token.v_proj', 'transformer.layers.0.cross_attn_image_to_token.out_proj',
                    'transformer.layers.0.mlp.lin1', 'transformer.layers.0.mlp.lin2',
                    'transformer.layers.0.norm1', 'transformer.layers.0.norm2', 'transformer.layers.0.norm3', 'transformer.layers.0.norm4',
                    # Layer 1 components  
                    'transformer.layers.1.self_attn.q_proj', 'transformer.layers.1.self_attn.k_proj', 'transformer.layers.1.self_attn.v_proj', 'transformer.layers.1.self_attn.out_proj',
                    'transformer.layers.1.cross_attn_token_to_image.q_proj', 'transformer.layers.1.cross_attn_token_to_image.k_proj', 'transformer.layers.1.cross_attn_token_to_image.v_proj', 'transformer.layers.1.cross_attn_token_to_image.out_proj',
                    'transformer.layers.1.cross_attn_image_to_token.q_proj', 'transformer.layers.1.cross_attn_image_to_token.k_proj', 'transformer.layers.1.cross_attn_image_to_token.v_proj', 'transformer.layers.1.cross_attn_image_to_token.out_proj',
                    'transformer.layers.1.mlp.lin1', 'transformer.layers.1.mlp.lin2',
                    'transformer.layers.1.norm1', 'transformer.layers.1.norm2', 'transformer.layers.1.norm3', 'transformer.layers.1.norm4'
                ],
                'description': '2个Transformer层（包含attention, MLP, norm）'
            },
            {
                'name': 'final_attention',
                'components': ['transformer.final_attn_token_to_image.q_proj', 'transformer.final_attn_token_to_image.k_proj', 
                              'transformer.final_attn_token_to_image.v_proj', 'transformer.final_attn_token_to_image.out_proj',
                              'transformer.norm_final_attn'],
                'description': '最终注意力层'
            },
            {
                'name': 'output_tokens',
                'components': ['iou_token', 'mask_tokens'],
                'description': '输出token（IoU和mask tokens）'
            },
            {
                'name': 'output_upscaling',
                'components': ['output_upscaling.0', 'output_upscaling.1', 'output_upscaling.3'],
                'description': '4个上采样层'
            },
            {
                'name': 'output_hypernetworks',
                'components': [
                    'output_hypernetworks_mlps.0.layers.0', 'output_hypernetworks_mlps.0.layers.1', 'output_hypernetworks_mlps.0.layers.2',
                    'output_hypernetworks_mlps.1.layers.0', 'output_hypernetworks_mlps.1.layers.1', 'output_hypernetworks_mlps.1.layers.2',
                    'output_hypernetworks_mlps.2.layers.0', 'output_hypernetworks_mlps.2.layers.1', 'output_hypernetworks_mlps.2.layers.2',
                    'output_hypernetworks_mlps.3.layers.0', 'output_hypernetworks_mlps.3.layers.1', 'output_hypernetworks_mlps.3.layers.2'
                ],
                'description': '4个超网络MLP'
            },
            {
                'name': 'iou_prediction_head',
                'components': ['iou_prediction_head.layers.0', 'iou_prediction_head.layers.1', 'iou_prediction_head.layers.2'],
                'description': '3层IoU预测头'
            }
        ]
    
    def _analyze_sam_architecture(self) -> Dict[str, Any]:
        """Analyze SAM's three-component architecture"""
        architecture = {
            'image_encoder': {
                'type': 'ViT (Vision Transformer)',
                'layers': self.sam_merger.total_layers,
                'components_per_layer': ['attention', 'mlp', 'norm'],
                'granularity_applicable': True,  # Can be split into groups
                'description': 'Main component for layer-wise merging optimization',
                'special_parameters': [
                    'pos_embed',      # Positional embeddings for image patches
                    'patch_embed',    # Patch embedding projection layer
                    'neck',           # Additional conv layers specific to SAM
                ]
            },
            'prompt_encoder': {
                'type': 'Lightweight encoder',
                'purpose': 'Encode points, boxes, masks, text prompts',
                'groups': 4,  # Fixed: 4 groups
                'description': 'point_embeddings, not_a_point_embed, mask_downscaling, no_mask_embed'
            },
            'mask_decoder': {
                'type': 'Transformer decoder',
                'purpose': 'Generate segmentation masks',
                'groups': 6,  # Fixed: 6 groups  
                'description': 'transformer_layers, final_attention, output_tokens, output_upscaling, output_hypernetworks, iou_prediction_head'
            }
        }
        return architecture
    
    def _create_layer_groups(self) -> List[Tuple[int, int]]:
        """Create layer groups handling remainder layers"""
        groups = []
        remaining_layers = self.vision_encoder_layers
        current_start = 0
        
        while remaining_layers > 0:
            if remaining_layers >= self.layer_granularity:
                # Full group
                group_size = self.layer_granularity
            else:
                # Remainder group - include all remaining layers
                group_size = remaining_layers
            
            groups.append((current_start, current_start + group_size))
            current_start += group_size
            remaining_layers -= group_size
        
        logger.info(f"Created layer groups: {groups}")
        return groups
    
    def get_slerp_component_config(self, component_id, component_type, config):
        """Generate merging configuration for slerp method for a specific component."""
        # Get t-values for each model
        t_values = {}
        for model in [self.base_model] + self.models:
            t_key = f"model_{model}_{component_id}_method_slerp_param_slerp_t"
            t_values[model] = config.get(t_key, 0)
        
        # Pick top 2 models based on t-values
        sorted_models = sorted(t_values, key=t_values.get, reverse=True)
        model1, model2 = sorted_models[:2]
        
        sources = [
            {"model": model1},
            {"model": model2}
        ]
        
        # Calculate interpolation weight
        t1, t2 = t_values[model1], t_values[model2]
        weights = np.exp([t1, t2]) / np.sum(np.exp([t1, t2]))
        slerp_t = weights[1]  # weight for model2
        
        return {
            "method": "slerp",
            "params": {"slerp_t": slerp_t}
        }, sources

    def get_initial_params(self):
        """Generate initial parameter configurations with simple and clear patterns."""
        init_trials = []
        methods_list = list(SUPPORTED_METHOD_PARAM_MAPS.keys())
        
        # Define scaling coefficients for task_arithmetic and ties methods
        merge_scales = [0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
        
        # All models including base model
        all_models = [self.base_model] + self.models
        SPECIAL_COMPONENTS = ['pos_embed', 'patch_embed', 'neck']
        
        def create_config_for_method(target_method, **method_kwargs):
            """Create a configuration for a specific method."""
            config = {}
            
            # 1. Set layer selection methods for Vision Encoder
            for group_index in range(self.num_groups):
                if self.component_granularity == "unified":
                    config[f"layer_selection_method_{group_index}"] = target_method
                else:
                    for component in COMPONENT_TYPES:
                        config[f"layer_selection_method_{group_index}_component_{component}"] = target_method
            
            # 2. Set special component methods (Vision Encoder special parameters)
            for special_component in SPECIAL_COMPONENTS:
                config[f"special_{special_component}_method"] = target_method
            
            # 3. Set prompt encoder group methods
            for group_idx, group_info in enumerate(self.prompt_encoder_groups):
                config[f"prompt_encoder_group_{group_idx}_method"] = target_method
            
            # 4. Set mask decoder group methods
            for group_idx, group_info in enumerate(self.mask_decoder_groups):
                config[f"mask_decoder_group_{group_idx}_method"] = target_method
            
            # 5. Set method-specific parameters
            self._set_method_parameters(config, target_method, all_models, **method_kwargs)
            
            return config
        
        # Generate initial configurations
        
        # 1. Linear configurations: Each model as individual starting point
        for target_model in all_models:
            config = create_config_for_method("linear", target_model=target_model)
            init_trials.append(config)
        
        # 2. Slerp configurations: Different t values
        for t in [0.3, 0.5, 0.7]:
            config = create_config_for_method("slerp", t=t)
            init_trials.append(config)
        
        # 3. Task arithmetic configurations with different scaling coefficients
        for scale in merge_scales:
            config = create_config_for_method("task_arithmetic", scaling_coefficient=scale)
            init_trials.append(config)
        
        # 4. TIES configurations with different scaling coefficients
        for scale in merge_scales:
            config = create_config_for_method("ties", scaling_coefficient=scale)
            init_trials.append(config)
        
        return init_trials
    
    def _set_method_parameters(self, config, target_method, all_models, **method_kwargs):
        """Set method-specific parameters for all components."""
        
        # Helper function to set parameters for a component
        def set_component_params(component_id):
            if target_method == "linear":
                target_model = method_kwargs.get('target_model', self.base_model)
                for model in all_models:
                    param_key = f"model_{model}_{component_id}_method_linear_param_weights"
                    config[param_key] = 1.0 if model == target_model else 0.0
                    
            elif target_method == "slerp":
                t_value = method_kwargs.get('t', 0.5)
                for model in all_models:
                    param_key = f"model_{model}_{component_id}_method_slerp_param_slerp_t"
                    config[param_key] = t_value
                    
            elif target_method in ["task_arithmetic", "ties"]:
                scaling_coeff = method_kwargs.get('scaling_coefficient', 0.8)
                method_config = self.merging_method[target_method]
                
                for method_param in SUPPORTED_METHOD_PARAM_MAPS[target_method]:
                    param_key = f"{component_id}_method_{target_method}_param_{method_param}"
                    
                    if method_param == "scaling_coefficient":
                        config[param_key] = scaling_coeff
                    elif method_param == "param_value_mask_rate":
                        config[param_key] = 0.2
                    else:
                        min_value = method_config[method_param]['min']
                        max_value = method_config[method_param]['max']
                        config[param_key] = (min_value + max_value) / 2.0
        
        # Set parameters for Vision Encoder layers
        for group_index in range(self.num_groups):
            if self.component_granularity == "unified":
                component_id = f"layer_{group_index}"
                set_component_params(component_id)
            else:
                for component in COMPONENT_TYPES:
                    component_id = f"layer_{group_index}_component_{component}"
                    set_component_params(component_id)
        
        # Set parameters for Vision Encoder special components
        SPECIAL_COMPONENTS = ['pos_embed', 'patch_embed', 'neck']
        for special_component in SPECIAL_COMPONENTS:
            component_id = f"special_{special_component}"
            set_component_params(component_id)
        
        # Set parameters for Prompt Encoder groups
        for group_idx, group_info in enumerate(self.prompt_encoder_groups):
            component_id = f"prompt_encoder_group_{group_idx}"
            set_component_params(component_id)
        
        # Set parameters for Mask Decoder groups
        for group_idx, group_info in enumerate(self.mask_decoder_groups):
            component_id = f"mask_decoder_group_{group_idx}"
            set_component_params(component_id)

    def generate_layer_config(self, config):
        """Generate SAM layer-wise configuration from optimization parameters."""
        layer_configs = []
        
        # Process each layer group for Vision Encoder
        for group_idx, (start_layer, end_layer) in enumerate(self.layer_groups):
            group_layer_count = end_layer - start_layer
            
            # Create layer configuration for this group
            group_config = {
                'sources': [{'model': cur_model} for cur_model in self.models],
                'merging_method': {}
            }
            
            if self.component_granularity == "unified":
                # Unified configuration for all components
                selected_method = config.get(f"layer_selection_method_{group_idx}", "linear")
                component_id = f"layer_{group_idx}"
                
                if selected_method == "slerp":
                    component_config, sources = self.get_slerp_component_config(component_id, "unified", config)
                    group_config['sources'] = sources
                else:
                    component_config = self._get_component_config(selected_method, component_id, config)
                    if selected_method in ["linear"]:
                        group_config['sources'] = [{'model': model} for model in [self.base_model] + self.models]
                
                group_config['merging_method']['unified'] = component_config
                
            else:
                # Component-specific configuration
                for component in COMPONENT_TYPES:
                    selected_method = config.get(f"layer_selection_method_{group_idx}_component_{component}", "linear")
                    component_id = f"layer_{group_idx}_component_{component}"
                    
                    if selected_method == "slerp":
                        component_config, sources = self.get_slerp_component_config(component_id, component, config)
                        group_config['sources'] = sources
                    else:
                        component_config = self._get_component_config(selected_method, component_id, config)
                        if selected_method in ["linear"]:
                            group_config['sources'] = [{'model': model} for model in [self.base_model] + self.models]
                    
                    group_config['merging_method'][component] = component_config
            
            # Replicate group configuration for all layers in this group
            for layer_idx in range(group_layer_count):
                layer_configs.append(copy.deepcopy(group_config))
        
        # Ensure we have configuration for all vision encoder layers
        while len(layer_configs) < self.vision_encoder_layers:
            layer_configs.append(copy.deepcopy(layer_configs[-1]))
        
        layer_configs = layer_configs[:self.vision_encoder_layers]
        
        # Generate special parameters configuration for Vision Encoder
        SPECIAL_COMPONENTS = ['pos_embed', 'patch_embed', 'neck']
        sam_special_components_config = {}
        
        for special_component in SPECIAL_COMPONENTS:
            selected_method = config.get(f"special_{special_component}_method", "linear")
            component_id = f"special_{special_component}"
            
            sam_special_components_config[special_component] = {
                'method': selected_method,
                'sources': [{'model': model} for model in self.models]
            }
            
            if selected_method == "slerp":
                component_config, sources = self.get_slerp_component_config(component_id, special_component, config)
                sam_special_components_config[special_component]['sources'] = sources
                sam_special_components_config[special_component]['params'] = component_config['params']
            else:
                component_config = self._get_component_config(selected_method, component_id, config)
                sam_special_components_config[special_component]['params'] = component_config['params']
                if selected_method in ["linear"]:
                    sam_special_components_config[special_component]['sources'] = [{'model': model} for model in [self.base_model] + self.models]
        
        # Generate Prompt Encoder group configurations
        prompt_encoder_config = {}
        for group_idx, group_info in enumerate(self.prompt_encoder_groups):
            selected_method = config.get(f"prompt_encoder_group_{group_idx}_method", "linear")
            component_id = f"prompt_encoder_group_{group_idx}"
            
            prompt_encoder_config[group_info['name']] = {
                'method': selected_method,
                'sources': [{'model': model} for model in self.models],
                'components': group_info['components']
            }
            
            if selected_method == "slerp":
                component_config, sources = self.get_slerp_component_config(component_id, group_info['name'], config)
                prompt_encoder_config[group_info['name']]['sources'] = sources
                prompt_encoder_config[group_info['name']]['params'] = component_config['params']
            else:
                component_config = self._get_component_config(selected_method, component_id, config)
                prompt_encoder_config[group_info['name']]['params'] = component_config['params']
                if selected_method in ["linear"]:
                    prompt_encoder_config[group_info['name']]['sources'] = [{'model': model} for model in [self.base_model] + self.models]
        
        # Generate Mask Decoder group configurations
        mask_decoder_config = {}
        for group_idx, group_info in enumerate(self.mask_decoder_groups):
            selected_method = config.get(f"mask_decoder_group_{group_idx}_method", "linear")
            component_id = f"mask_decoder_group_{group_idx}"
            
            mask_decoder_config[group_info['name']] = {
                'method': selected_method,
                'sources': [{'model': model} for model in self.models],
                'components': group_info['components']
            }
            
            if selected_method == "slerp":
                component_config, sources = self.get_slerp_component_config(component_id, group_info['name'], config)
                mask_decoder_config[group_info['name']]['sources'] = sources
                mask_decoder_config[group_info['name']]['params'] = component_config['params']
            else:
                component_config = self._get_component_config(selected_method, component_id, config)
                mask_decoder_config[group_info['name']]['params'] = component_config['params']
                if selected_method in ["linear"]:
                    mask_decoder_config[group_info['name']]['sources'] = [{'model': model} for model in [self.base_model] + self.models]
        
        # Create complete SAM configuration
        sam_config = {
            'base_model': self.base_model,
            'layers': layer_configs,
            'special_parameters': sam_special_components_config,
            'prompt_encoder': prompt_encoder_config,
            'mask_decoder': mask_decoder_config
        }
        
        return sam_config
    
    def _get_component_config(self, selected_method, component_id, config):
        """Get component configuration for a specific method."""
        method_params = {}
        
        for param_name in SUPPORTED_METHOD_PARAM_MAPS[selected_method]:
            if selected_method in ["linear", "slerp"]:
                # Collect weights for each model (including base model)
                weights = []
                for model in [self.base_model] + self.models:
                    param_key = f"model_{model}_{component_id}_method_{selected_method}_param_{param_name}"
                    weight = config.get(param_key, 0)
                    weights.append(weight)
                method_params[param_name] = weights
            else:
                # Single parameter value
                param_key = f"{component_id}_method_{selected_method}_param_{param_name}"
                method_params[param_name] = config.get(param_key, 0)
        
        return {
            "method": selected_method,
            "params": method_params
        }

    def get_config_space(self):
        """Define the configuration space for hyperparameter optimization."""
        cs = ConfigurationSpace()
        methods_list = list(SUPPORTED_METHOD_PARAM_MAPS.keys())
        config_list = []
        conditions_list = []
        
        # Define special component types
        SPECIAL_COMPONENTS = ['pos_embed', 'patch_embed', 'neck']
        
        # 1. Add selection methods for Vision Encoder layer groups
        method_hyperparams = {}
        for group_index in range(self.num_groups):
            if self.component_granularity == "unified":
                method_param = Categorical(
                    f"layer_selection_method_{group_index}",
                    items=methods_list
                )
                config_list.append(method_param)
                method_hyperparams[f"layer_selection_method_{group_index}"] = method_param
            else:
                for component in COMPONENT_TYPES:
                    method_param = Categorical(
                        f"layer_selection_method_{group_index}_component_{component}",
                        items=methods_list
                    )
                    config_list.append(method_param)
                    method_hyperparams[f"layer_selection_method_{group_index}_component_{component}"] = method_param
        
        # 2. Add parameters for Vision Encoder layers with conditions
        self._add_component_parameters(
            config_list, conditions_list, method_hyperparams, 
            "layer", range(self.num_groups), COMPONENT_TYPES if self.component_granularity != "unified" else ["unified"]
        )
        
        # 3. Add special component parameters for Vision Encoder
        special_method_hyperparams = {}
        for special_component in SPECIAL_COMPONENTS:
            method_param = Categorical(
                f"special_{special_component}_method",
                items=methods_list
            )
            config_list.append(method_param)
            special_method_hyperparams[special_component] = method_param
        
        self._add_special_component_parameters(
            config_list, conditions_list, special_method_hyperparams, SPECIAL_COMPONENTS
        )
        
        # 4. Add Prompt Encoder group parameters
        prompt_encoder_method_hyperparams = {}
        for group_idx, group_info in enumerate(self.prompt_encoder_groups):
            method_param = Categorical(f"prompt_encoder_group_{group_idx}_method", items=methods_list)
            config_list.append(method_param)
            prompt_encoder_method_hyperparams[f"group_{group_idx}"] = method_param
            
            self._add_unified_component_parameters(
                config_list, conditions_list, method_param, f"prompt_encoder_group_{group_idx}"
            )
        
        # 5. Add Mask Decoder group parameters
        mask_decoder_method_hyperparams = {}
        for group_idx, group_info in enumerate(self.mask_decoder_groups):
            method_param = Categorical(f"mask_decoder_group_{group_idx}_method", items=methods_list)
            config_list.append(method_param)
            mask_decoder_method_hyperparams[f"group_{group_idx}"] = method_param
            
            self._add_unified_component_parameters(
                config_list, conditions_list, method_param, f"mask_decoder_group_{group_idx}"
            )
        
        # Add all hyperparameters to configuration space
        cs.add_hyperparameters(config_list)
        
        # Add all conditions to configuration space
        for condition in conditions_list:
            cs.add_condition(condition)
        
        return cs
    
    def _add_component_parameters(self, config_list, conditions_list, method_hyperparams, 
                                 component_prefix, indices, components):
        """Add parameters for components with conditions."""
        all_models = [self.base_model] + list(self.models)
        
        for index in indices:
            for component in components:
                if component == "unified":
                    method_selector = method_hyperparams[f"{component_prefix}_selection_method_{index}"]
                    component_id = f"{component_prefix}_{index}"
                else:
                    method_selector = method_hyperparams[f"{component_prefix}_selection_method_{index}_component_{component}"]
                    component_id = f"{component_prefix}_{index}_component_{component}"
                
                for method in SUPPORTED_METHOD_PARAM_MAPS.keys():
                    method_config = self.merging_method[method]
                    
                    for method_param in SUPPORTED_METHOD_PARAM_MAPS[method]:
                        min_value = method_config[method_param]['min']
                        max_value = method_config[method_param]['max']
                        
                        if method in ["linear", "slerp"]:
                            # Model-specific parameters (including base model)
                            for model in all_models:
                                param_name = f"model_{model}_{component_id}_method_{method}_param_{method_param}"
                                param = Float(param_name, (min_value, max_value))
                                config_list.append(param)
                                
                                # Add condition: parameter is active only when this method is selected
                                condition = EqualsCondition(param, method_selector, method)
                                conditions_list.append(condition)
                        else:
                            # Component-specific parameters
                            param_name = f"{component_id}_method_{method}_param_{method_param}"
                            param = Float(param_name, (min_value, max_value))
                            config_list.append(param)
                            
                            # Add condition: parameter is active only when this method is selected
                            condition = EqualsCondition(param, method_selector, method)
                            conditions_list.append(condition)
    
    def _add_special_component_parameters(self, config_list, conditions_list, method_hyperparams, components):
        """Add parameters for special components with conditions."""
        all_models = [self.base_model] + list(self.models)
        
        for component in components:
            method_selector = method_hyperparams[component]
            
            for method in SUPPORTED_METHOD_PARAM_MAPS.keys():
                method_config = self.merging_method[method]
                
                for method_param_name in SUPPORTED_METHOD_PARAM_MAPS[method]:
                    min_value = method_config[method_param_name]['min']
                    max_value = method_config[method_param_name]['max']
                    
                    if method in ["linear", "slerp"]:
                        # Model-specific weights (including base model)
                        for model in all_models:
                            param_name = f"model_{model}_special_{component}_method_{method}_param_{method_param_name}"
                            param = Float(param_name, (min_value, max_value))
                            config_list.append(param)
                            
                            # Add condition: parameter is active only when this method is selected
                            condition = EqualsCondition(param, method_selector, method)
                            conditions_list.append(condition)
                    else:
                        # Single parameter value for other methods
                        param_name = f"special_{component}_method_{method}_param_{method_param_name}"
                        param = Float(param_name, (min_value, max_value))
                        config_list.append(param)
                        
                        # Add condition: parameter is active only when this method is selected
                        condition = EqualsCondition(param, method_selector, method)
                        conditions_list.append(condition)
    
    def _add_unified_component_parameters(self, config_list, conditions_list, method_selector, component_id):
        """Add parameters for unified components (prompt_encoder, mask_decoder) with conditions."""
        all_models = [self.base_model] + list(self.models)
        
        for method in SUPPORTED_METHOD_PARAM_MAPS.keys():
            method_config = self.merging_method[method]
            
            for method_param_name in SUPPORTED_METHOD_PARAM_MAPS[method]:
                min_value = method_config[method_param_name]['min']
                max_value = method_config[method_param_name]['max']
                
                if method in ["linear", "slerp"]:
                    # Model-specific parameters (including base model)
                    for model in all_models:
                        param_name = f"model_{model}_{component_id}_method_{method}_param_{method_param_name}"
                        param = Float(param_name, (min_value, max_value))
                        config_list.append(param)
                        
                        # Add condition: parameter is active only when this method is selected
                        condition = EqualsCondition(param, method_selector, method)
                        conditions_list.append(condition)
                else:
                    # Single parameter value for other methods
                    param_name = f"{component_id}_method_{method}_param_{method_param_name}"
                    param = Float(param_name, (min_value, max_value))
                    config_list.append(param)
                    
                    # Add condition: parameter is active only when this method is selected
                    condition = EqualsCondition(param, method_selector, method)
                    conditions_list.append(condition)

    def objective(self, config, seed, budget):
        """Objective function for hyperparameter optimization."""
        logger.info(f"Start evaluating, current budget is {budget}")
        budget = int(budget)
        result = {}
        
        # Generate SAM layer-wise configuration
        sam_config = self.generate_layer_config(config)
        logger.info(f"Generated SAM config for {len(sam_config['layers'])} layers")
        
        try:
            # Load models for merging
            model_paths = {}
            for model_name in self.models:
                model_paths[model_name] = model_name  # Assuming model names are paths
            model_paths[self.base_model] = self.base_model  # Base model uses empty path
            
            # Perform merge using SAM layer-wise merger
            merged_model = self.sam_merger.merge_models_layer_wise(
                layer_config=sam_config,
                model_paths=model_paths,
                debug_mode=False
            )
            
            # Save merged model temporarily for evaluation
            temp_model_path = os.path.join(self.output_path, f"temp_merged_model_{seed}_{budget}"+".pth")
            #merged_model.save_pretrained(temp_model_path)
            torch.save(merged_model.state_dict(),temp_model_path)
            # Todo： evaluate the merged model
            eval_result = {}
            #self.evaluator_instance.evaluate(temp_model_path, budget)
            res = evaluate_main(merged_model,self.config)
            for key,val in res.items():
                # TODO to check some invalid values
                eval_result[key] = val
            
            # Calculate error rate (1 - score)
            for cur_task in self.evaluate_tasks:
                result[cur_task] = 1 - eval_result[cur_task]#['score']
            
            # Clean up temporary files
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            
            # Clean up resources
            del merged_model
            gc.collect()
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            logger.error(traceback.format_exc())
            # Return worst score on error
            for cur_task in self.evaluate_tasks:
                result[cur_task] = 1
            raise ValueError("objective evaluation failed")
        
        return result[self.evaluate_tasks[0]]
    
    def optimize(self):
        """Run hyperparameter optimization to find optimal merging parameters."""
        configspace = self.get_config_space()
        
        # Set up GPU cluster for parallel evaluation
        logger.info(",".join(map(str, range(min(self.n_workers, torch.cuda.device_count())))))
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(min(self.n_workers, torch.cuda.device_count())))),
            threads_per_worker=1,
            memory_limit="90GB",
            device_memory_limit=0.9
        )
        client = Client(cluster)
        logger.info(f"Client: {client}")
        
        if self.n_trials == 0:
            self.n_trials = get_n_trials_for_hyperband_multifidelity(
                total_budget=self.total_budget,
                min_budget=self.min_budget,
                max_budget=self.max_budget,
                eta=self.eta,
                print_summary=True,
            )
        
        # Scenario object specifying the optimization "environment"
        scenario = Scenario(
            configspace,
            output_directory=Path(self.output_path),
            deterministic=True,
            n_trials=self.n_trials,
            min_budget=self.min_budget,
            max_budget=self.max_budget
        )
        
        intensifier = MultiFidelityFacade.get_intensifier(scenario=scenario, eta=self.eta)
        
        if self.load_run_history is not None:
            runhistory = RunHistory()
            runhistory.update_from_json(self.load_run_history, configspace)
            initial_design = MultiFidelityFacade.get_initial_design(
                scenario,
                n_configs=0,
                additional_configs=None,
            )
            smac = MultiFidelityFacade(
                scenario,
                self.objective,
                overwrite=False,
                intensifier=intensifier,
                logging_level=0,
                initial_design=initial_design,
                dask_client=client
            )
            for (trial_key, trial_value) in runhistory.items():
                trial_info = TrialInfo(
                    config=runhistory.get_config(trial_key.config_id),
                    instance=trial_key.instance,
                    seed=trial_key.seed,
                    budget=trial_key.budget
                )
                smac.tell(trial_info, trial_value)
        else:
            init_trials = self.get_initial_params()
            configurations = [Configuration(configspace, trial) for trial in init_trials]
            initial_design = MultiFidelityFacade.get_initial_design(
                scenario,
                n_configs=self.random_init_points,
                additional_configs=configurations,
            )
            
            smac = MultiFidelityFacade(
                scenario,
                self.objective,
                overwrite=False,
                intensifier=intensifier,
                initial_design=initial_design,
                logging_level=0,
                dask_client=client
            )
        
        incumbent = smac.optimize()
        
        # Calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        logger.info(f"Incumbent cost: {incumbent_cost}")
        
        # Save the best configuration
        best_sam_config = self.generate_layer_config(incumbent)
        save_layer_config(best_sam_config, os.path.join(self.output_path, "best_sam_config.yaml"))
        
        return incumbent, incumbent_cost
    
    def merge(self):
        """Main merge function that runs optimization."""
        return self.optimize()
    
    def eval_config(self, config, config_id=0,prefix =''):
        """Evaluate a specific configuration."""
        result = {}
        configspace = self.get_config_space()
        config = Configuration(configspace, config)
        sam_config = self.generate_layer_config(config)
        logger.info(f"Evaluating SAM config with {len(sam_config['layers'])} layers")
        logger.info(f"Prompt encoder groups: {list(sam_config['prompt_encoder'].keys())}")
        logger.info(f"Mask decoder groups: {list(sam_config['mask_decoder'].keys())}")
        
        try:
            # Load models for merging
            model_paths = {}
            for model_name in self.models:
                model_paths[model_name] = model_name
            
            model_paths[self.base_model] = self.base_model
            
            # Perform merge
            merged_model = self.sam_merger.merge_models_layer_wise(
                layer_config=sam_config,
                model_paths=model_paths,
                debug_mode=True
            )
            
            # Save and evaluate
            temp_model_path = os.path.join(self.output_path, f"eval_model_{config_id}.pth")
            torch.save(merged_model.state_dict(), temp_model_path)
            
            eval_result = {}
            res = evaluate_main(merged_model, self.config, is_train=False)
            try:
                with open("results/merge_res.txt", 'a') as f:
                    for key, val in res.items():
                        eval_result[key] = val
                        f.write(f"{prefix}_{key}: mean dice: {val:.3f} \n")
            except:
                pass
            
            # Clean up
            del merged_model
            gc.collect()
            
        except Exception as e:
            logger.error(f"Config evaluation failed: {str(e)}")
            logger.error(traceback.format_exc())
            result['score'] = 0
        
        return result

if __name__ == "__main__":
    pass