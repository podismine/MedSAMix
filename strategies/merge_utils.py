from transformers import SamModel
from typing import List, Dict, Any, Callable, Optional
import copy
from enum import Enum
import yaml
import re
from utils import logger
from per_segment_anything import sam_model_registry as per_sam_model_registry
from methods import merging_methods_dict

class ComponentType(Enum):
    """ViT layer component types"""
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"

class SAMLayerWiseMerger:
    """SAM Model Layer-wise Merger with guaranteed layer coverage and consistent model size"""
    
    def __init__(self, base_model: SamModel):
        """
        Initialize the merger
        
        Args:
            base_model: Base SAM model
        """
        self.base_model = base_model
        self.device = next(base_model.parameters()).device
        self.merge_instances_cache = {}
        
        # Analyze model structure layer by layer
        self.layer_structure = self._analyze_layer_structure()
        logger.info("Analyzed layer structure:")
        logger.info(self.layer_structure)
        
        self.total_layers = len(self.layer_structure['image_encoder']['layers'])
        
        # Define component mappings for prompt_encoder and mask_decoder
        self.prompt_encoder_mapping = self._define_prompt_encoder_mapping()
        self.mask_decoder_mapping = self._define_mask_decoder_mapping()
        
    def _define_prompt_encoder_mapping(self) -> Dict[str, List[str]]:
        """Define mapping from component groups to actual parameter names for prompt_encoder"""
        return {
            'point_embeddings': [
                'prompt_encoder.point_embeddings.0.weight',
                'prompt_encoder.point_embeddings.1.weight', 
                'prompt_encoder.point_embeddings.2.weight',
                'prompt_encoder.point_embeddings.3.weight'
            ],
            'not_a_point_embed': [
                'prompt_encoder.not_a_point_embed.weight'
            ],
            'mask_downscaling': [
                'prompt_encoder.mask_downscaling.0.weight',
                'prompt_encoder.mask_downscaling.0.bias',
                'prompt_encoder.mask_downscaling.1.weight', 
                'prompt_encoder.mask_downscaling.1.bias',
                'prompt_encoder.mask_downscaling.3.weight',
                'prompt_encoder.mask_downscaling.3.bias',
                'prompt_encoder.mask_downscaling.4.weight',
                'prompt_encoder.mask_downscaling.4.bias',
                'prompt_encoder.mask_downscaling.6.weight',
                'prompt_encoder.mask_downscaling.6.bias'
            ],
            'no_mask_embed': [
                'prompt_encoder.no_mask_embed.weight'
            ]
        }
    
    def _define_mask_decoder_mapping(self) -> Dict[str, List[str]]:
        """Define mapping from component groups to actual parameter names for mask_decoder"""
        return {
            'transformer_layers': [
                # Layer 0 components
                'mask_decoder.transformer.layers.0.self_attn.q_proj.weight',
                'mask_decoder.transformer.layers.0.self_attn.q_proj.bias',
                'mask_decoder.transformer.layers.0.self_attn.k_proj.weight',
                'mask_decoder.transformer.layers.0.self_attn.k_proj.bias',
                'mask_decoder.transformer.layers.0.self_attn.v_proj.weight',
                'mask_decoder.transformer.layers.0.self_attn.v_proj.bias',
                'mask_decoder.transformer.layers.0.self_attn.out_proj.weight',
                'mask_decoder.transformer.layers.0.self_attn.out_proj.bias',
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj.weight',
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj.bias',
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.k_proj.weight',
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.k_proj.bias',
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj.weight',
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj.bias',
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.out_proj.weight',
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.out_proj.bias',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj.weight',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj.bias',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.k_proj.weight',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.k_proj.bias',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.v_proj.weight',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.v_proj.bias',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.out_proj.weight',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.out_proj.bias',
                'mask_decoder.transformer.layers.0.mlp.lin1.weight',
                'mask_decoder.transformer.layers.0.mlp.lin1.bias',
                'mask_decoder.transformer.layers.0.mlp.lin2.weight',
                'mask_decoder.transformer.layers.0.mlp.lin2.bias',
                'mask_decoder.transformer.layers.0.norm1.weight',
                'mask_decoder.transformer.layers.0.norm1.bias',
                'mask_decoder.transformer.layers.0.norm2.weight',
                'mask_decoder.transformer.layers.0.norm2.bias',
                'mask_decoder.transformer.layers.0.norm3.weight',
                'mask_decoder.transformer.layers.0.norm3.bias',
                'mask_decoder.transformer.layers.0.norm4.weight',
                'mask_decoder.transformer.layers.0.norm4.bias',
                # Layer 1 components
                'mask_decoder.transformer.layers.1.self_attn.q_proj.weight',
                'mask_decoder.transformer.layers.1.self_attn.q_proj.bias',
                'mask_decoder.transformer.layers.1.self_attn.k_proj.weight',
                'mask_decoder.transformer.layers.1.self_attn.k_proj.bias',
                'mask_decoder.transformer.layers.1.self_attn.v_proj.weight',
                'mask_decoder.transformer.layers.1.self_attn.v_proj.bias',
                'mask_decoder.transformer.layers.1.self_attn.out_proj.weight',
                'mask_decoder.transformer.layers.1.self_attn.out_proj.bias',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.q_proj.weight',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.q_proj.bias',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.k_proj.weight',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.k_proj.bias',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.v_proj.weight',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.v_proj.bias',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.out_proj.weight',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.out_proj.bias',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.q_proj.weight',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.q_proj.bias',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.k_proj.weight',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.k_proj.bias',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.v_proj.weight',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.v_proj.bias',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.out_proj.weight',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.out_proj.bias',
                'mask_decoder.transformer.layers.1.mlp.lin1.weight',
                'mask_decoder.transformer.layers.1.mlp.lin1.bias',
                'mask_decoder.transformer.layers.1.mlp.lin2.weight',
                'mask_decoder.transformer.layers.1.mlp.lin2.bias',
                'mask_decoder.transformer.layers.1.norm1.weight',
                'mask_decoder.transformer.layers.1.norm1.bias',
                'mask_decoder.transformer.layers.1.norm2.weight',
                'mask_decoder.transformer.layers.1.norm2.bias',
                'mask_decoder.transformer.layers.1.norm3.weight',
                'mask_decoder.transformer.layers.1.norm3.bias',
                'mask_decoder.transformer.layers.1.norm4.weight',
                'mask_decoder.transformer.layers.1.norm4.bias'
            ],
            'final_attention': [
                'mask_decoder.transformer.final_attn_token_to_image.q_proj.weight',
                'mask_decoder.transformer.final_attn_token_to_image.q_proj.bias',
                'mask_decoder.transformer.final_attn_token_to_image.k_proj.weight',
                'mask_decoder.transformer.final_attn_token_to_image.k_proj.bias',
                'mask_decoder.transformer.final_attn_token_to_image.v_proj.weight',
                'mask_decoder.transformer.final_attn_token_to_image.v_proj.bias',
                'mask_decoder.transformer.final_attn_token_to_image.out_proj.weight',
                'mask_decoder.transformer.final_attn_token_to_image.out_proj.bias',
                'mask_decoder.transformer.norm_final_attn.weight',
                'mask_decoder.transformer.norm_final_attn.bias'
            ],
            'output_tokens': [
                'mask_decoder.iou_token.weight',
                'mask_decoder.mask_tokens.weight'
            ],
            'output_upscaling': [
                'mask_decoder.output_upscaling.0.weight',
                'mask_decoder.output_upscaling.0.bias',
                'mask_decoder.output_upscaling.1.weight',
                'mask_decoder.output_upscaling.1.bias',
                'mask_decoder.output_upscaling.3.weight',
                'mask_decoder.output_upscaling.3.bias'
            ],
            'output_hypernetworks': [
                'mask_decoder.output_hypernetworks_mlps.0.layers.0.weight',
                'mask_decoder.output_hypernetworks_mlps.0.layers.0.bias',
                'mask_decoder.output_hypernetworks_mlps.0.layers.1.weight',
                'mask_decoder.output_hypernetworks_mlps.0.layers.1.bias',
                'mask_decoder.output_hypernetworks_mlps.0.layers.2.weight',
                'mask_decoder.output_hypernetworks_mlps.0.layers.2.bias',
                'mask_decoder.output_hypernetworks_mlps.1.layers.0.weight',
                'mask_decoder.output_hypernetworks_mlps.1.layers.0.bias',
                'mask_decoder.output_hypernetworks_mlps.1.layers.1.weight',
                'mask_decoder.output_hypernetworks_mlps.1.layers.1.bias',
                'mask_decoder.output_hypernetworks_mlps.1.layers.2.weight',
                'mask_decoder.output_hypernetworks_mlps.1.layers.2.bias',
                'mask_decoder.output_hypernetworks_mlps.2.layers.0.weight',
                'mask_decoder.output_hypernetworks_mlps.2.layers.0.bias',
                'mask_decoder.output_hypernetworks_mlps.2.layers.1.weight',
                'mask_decoder.output_hypernetworks_mlps.2.layers.1.bias',
                'mask_decoder.output_hypernetworks_mlps.2.layers.2.weight',
                'mask_decoder.output_hypernetworks_mlps.2.layers.2.bias',
                'mask_decoder.output_hypernetworks_mlps.3.layers.0.weight',
                'mask_decoder.output_hypernetworks_mlps.3.layers.0.bias',
                'mask_decoder.output_hypernetworks_mlps.3.layers.1.weight',
                'mask_decoder.output_hypernetworks_mlps.3.layers.1.bias',
                'mask_decoder.output_hypernetworks_mlps.3.layers.2.weight',
                'mask_decoder.output_hypernetworks_mlps.3.layers.2.bias'
            ],
            'iou_prediction_head': [
                'mask_decoder.iou_prediction_head.layers.0.weight',
                'mask_decoder.iou_prediction_head.layers.0.bias',
                'mask_decoder.iou_prediction_head.layers.1.weight',
                'mask_decoder.iou_prediction_head.layers.1.bias',
                'mask_decoder.iou_prediction_head.layers.2.weight',
                'mask_decoder.iou_prediction_head.layers.2.bias'
            ]
        }
        
    def _get_merge_instance(self, merging_method: str):
        """Get or create merge instance for the specified method"""
        if merging_method not in self.merge_instances_cache:
            if merging_method not in merging_methods_dict:
                raise ValueError(f"Unsupported merge method: {merging_method}")
            merging_class = merging_methods_dict[merging_method]
            self.merge_instances_cache[merging_method] = merging_class()
        return self.merge_instances_cache[merging_method]
    
    def _analyze_layer_structure(self) -> Dict[str, Any]:
        """Analyze layer structure to ensure complete coverage"""
        structure = {
            'image_encoder': {
                'layers': {},  # {layer_idx: {attention: [...], mlp: [...], norm: [...]}}
                'special_params': []  # pos_embed, cls_token, etc.
            },
            'prompt_encoder': [],
            'mask_decoder': []
        }
        
        # Analyze Vision Encoder layers
        for name, param in self.base_model.named_parameters():
            if 'image_encoder' in name and 'blocks.' in name:
                # Extract layer index
                layer_match = re.search(r'blocks\.(\d+)', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    
                    if layer_idx not in structure['image_encoder']['layers']:
                        structure['image_encoder']['layers'][layer_idx] = {
                            'attention': [],
                            'mlp': [],
                            'norm': []
                        }
                    
                    # Categorize parameters by component
                    if 'attn' in name:
                        structure['image_encoder']['layers'][layer_idx]['attention'].append(name)
                    elif 'mlp' in name:
                        structure['image_encoder']['layers'][layer_idx]['mlp'].append(name)
                    elif 'norm' in name:
                        structure['image_encoder']['layers'][layer_idx]['norm'].append(name)
            
            elif 'image_encoder' in name and 'blocks.' not in name:
                structure['image_encoder']['special_params'].append(name)
            elif 'prompt_encoder' in name:
                structure['prompt_encoder'].append(name)
            elif 'mask_decoder' in name:
                structure['mask_decoder'].append(name)
        
        return structure
    
    def validate_layer_config(self, config: Dict[str, Any]) -> bool:
        """Validate that all layers are covered in configuration"""
        layer_configs = config.get('layers', [])
        
        # Check if we have configuration for all layers
        if len(layer_configs) != self.total_layers:
            print(f"Error: Config has {len(layer_configs)} layers, model has {self.total_layers} layers")
            return False
        
        # Validate each layer configuration
        for layer_idx, layer_config in enumerate(layer_configs):
            if 'sources' not in layer_config:
                print(f"Error: Layer {layer_idx} missing 'sources'")
                return False
            
            # Check merging method configuration
            if 'merging_method' not in layer_config:
                print(f"Error: Layer {layer_idx} missing 'merging_method'")
                return False
            
            merging_config = layer_config['merging_method']
            
            # Validate component-specific or unified configuration
            if 'unified' in merging_config:
                # Single configuration for all components
                self._validate_method_params(merging_config['unified'], layer_idx)
            else:
                # Component-specific configurations
                required_components = ['attention', 'mlp', 'norm']
                for component in required_components:
                    if component not in merging_config:
                        print(f"Error: Layer {layer_idx} missing '{component}' configuration")
                        return False
                    self._validate_method_params(merging_config[component], layer_idx, component)
        
        return True
    
    def _validate_method_params(self, method_config: Dict[str, Any], layer_idx: int, component: str = "unified") -> bool:
        """Validate method parameters"""
        method_name = method_config.get('method', '')
        method_params = method_config.get('params', {})
        
        if method_name == "task_arithmetic":
            scaling_coeff = method_params.get('scaling_coefficient', 1.0)
            if not (0.0 <= scaling_coeff <= 1.0):
                print(f"Warning: Layer {layer_idx} {component} scaling_coefficient out of range")
                return False
        elif method_name == "slerp":
            slerp_t = method_params.get('slerp_t', 0.5)
            if not (0.0 <= slerp_t <= 1.0):
                print(f"Warning: Layer {layer_idx} {component} slerp_t out of range")
                return False
        elif method_name == "ties":
            mask_rate = method_params.get('param_value_mask_rate', 0.8)
            scaling_coeff = method_params.get('scaling_coefficient', 1.0)
            if not (0.0 <= mask_rate < 0.99) or not (0.0 <= scaling_coeff <= 1.0):
                print(f"Warning: Layer {layer_idx} {component} TIES parameters out of range")
                return False
        elif method_name == "linear":
            weights = method_params.get('weights', [])
            if not all(0.0 <= w <= 1.0 for w in weights):
                print(f"Warning: Layer {layer_idx} {component} weights out of range")
                return False
        
        return True
    
    def merge_models_layer_wise(
        self,
        layer_config: Dict[str, Any],
        model_paths: Dict[str, str],
        progress_callback: Optional[Callable[[int, str, float], None]] = None,
        debug_mode: bool = False
    ) -> SamModel:
        """
        Merge models with layer-wise configuration ensuring complete coverage
        
        Args:
            layer_config: Layer-wise configuration
            model_paths: Dictionary mapping model names to paths
            progress_callback: Progress callback (layer_idx, component, progress)
            debug_mode: Enable debug mode
            
        Returns:
            Merged model with guaranteed consistent size
        """
        # Validate configuration
        if not self.validate_layer_config(layer_config):
            raise ValueError("Layer configuration validation failed")
        print("Layer configuration validated successfully")
        
        # Load models
        models = {}
        base_model_name = layer_config.get('base_model', '')
        for model_name, model_path in model_paths.items():
            if model_name == base_model_name or not model_path:
                models[model_name] = self.base_model
            else:
                models[model_name] = per_sam_model_registry['vit_b'](checkpoint=model_path)
        
        # Create merged model
        merged_model = copy.deepcopy(self.base_model)
        merged_state_dict = merged_model.state_dict()
        
        print(f"Starting layer-wise merge: {self.total_layers} layers")
        
        # Process each layer
        layer_configs = layer_config['layers']
        for layer_idx in range(self.total_layers):
            if layer_idx >= len(layer_configs):
                print(f"Warning: No configuration for layer {layer_idx}, using base model")
                continue
            
            layer_merge_config = layer_configs[layer_idx]
            sources = layer_merge_config['sources']
            merging_method_config = layer_merge_config['merging_method']
            
            logger.info(f"\nProcessing Layer {layer_idx}")
            
            # Get layer parameters
            layer_params = self.layer_structure['image_encoder']['layers'].get(layer_idx, {})
            
            # Process each component
            components = ['attention', 'mlp', 'norm']
            for component in components:
                component_params = layer_params.get(component, [])
                if not component_params:
                    continue
                
                # Determine merge configuration for this component
                if 'unified' in merging_method_config:
                    # Use unified configuration for all components
                    component_config = merging_method_config['unified']
                else:
                    # Use component-specific configuration
                    component_config = merging_method_config.get(component, {})
                
                if not component_config:
                    print(f"Warning: No configuration for layer {layer_idx} component {component}")
                    continue
                
                method_name = component_config['method']
                method_params = component_config['params']
                
                if debug_mode:
                    print(f"  {component.upper()}: {method_name} with {len(component_params)} parameters, sources {sources}")
                
                # Merge each parameter in this component
                for param_name in component_params:
                    try:
                        # Get base tensor
                        base_tensor = self.base_model.state_dict()[param_name]
                        
                        # Collect tensors from source models
                        tensors_to_merge = []
                        for source in sources:
                            model_name = source['model']
                            if model_name in models:
                                tensor = models[model_name].state_dict()[param_name]
                                tensors_to_merge.append(tensor)
                        
                        if not tensors_to_merge:
                            continue
                        
                        logger.info(f"merge using {method_name} {method_params}. for param {param_name} with {len(tensors_to_merge)} tensor")
                        # Merge using your existing method
                        merge_instance = self._get_merge_instance(method_name)
                        merged_tensor = merge_instance.merge_tensor(
                            base_tensor,
                            tensors_to_merge,
                            method_params
                        )
                        logger.info(f"Merged {param_name} using {method_name} method")
                        logger.info(f"current param name: {param_name}")
                        
                        # Update merged model - ensure same size
                        if merged_tensor.shape != base_tensor.shape:
                            print(f"Error: Size mismatch for {param_name}: {merged_tensor.shape} vs {base_tensor.shape}")
                            continue
                        
                        merged_state_dict[param_name].copy_(merged_tensor)
                        
                    except Exception as e:
                        print(f"Failed to merge {param_name}: {str(e)}")
                        if debug_mode:
                            import traceback
                            traceback.print_exc()
            
            # Progress callback
            if progress_callback:
                progress = (layer_idx + 1) / self.total_layers
                progress_callback(layer_idx, "layer_complete", progress)
        
        # Merge special parameters (pos_embed, patch_embed, neck)
        self._merge_special_parameters_v1(merged_model, models, layer_config, debug_mode)
        
        # Merge other components (prompt_encoder, mask_decoder) with new group-based approach
        self._merge_other_components_v2(merged_model, models, layer_config, debug_mode)
        
        print("Layer-wise merging completed with guaranteed size consistency")
        return merged_model

    def _merge_special_parameters_v1(self, merged_model: SamModel, models: Dict[str, SamModel],
                                config: Dict[str, Any], debug_mode: bool):
        """Merge special parameters like pos_embed, cls_token, patch_embed, neck"""
        special_configs = config.get('special_parameters', {})
        if not special_configs:
            return
        
        # Define mapping of special components to actual parameter names
        SPECIAL_COMPONENT_MAPPING = {
            'pos_embed': ['image_encoder.pos_embed'],
            'patch_embed': ['image_encoder.patch_embed.proj.bias', 'image_encoder.patch_embed.proj.weight'],
            'neck': ['image_encoder.neck.0.weight', 'image_encoder.neck.1.weight', 
                    'image_encoder.neck.1.bias', 'image_encoder.neck.2.weight',
                    'image_encoder.neck.3.weight', 'image_encoder.neck.3.bias']
        }
        
        # Get all special parameters from layer structure
        all_special_params = self.layer_structure['image_encoder']['special_params']
        
        # Process each special component
        for special_component, component_config in special_configs.items():
            if special_component not in SPECIAL_COMPONENT_MAPPING:
                if debug_mode:
                    print(f"Warning: Unknown special component {special_component}")
                continue
                
            method_name = component_config.get('method', 'linear')
            method_params = component_config.get('params', {'weights': [1.0]})
            sources = component_config.get('sources', [])
            logger.info(f"\nProcessing Special Component: {special_component} using {method_name} method source {sources}")
            # Get parameter names for this component
            param_names = SPECIAL_COMPONENT_MAPPING[special_component]
            
            # Process each parameter in this component
            for param_name in param_names:
                # Check if this parameter exists in the special params list
                if param_name not in all_special_params:
                    if debug_mode:
                        print(f"Parameter {param_name} not found in special params, skipping")
                    continue
                    
                try:
                    # Get base tensor
                    base_tensor = self.base_model.state_dict()[param_name]
                    tensors_to_merge = []
                    
                    # Collect tensors from source models
                    for source in sources:
                        model_name = source['model']
                        if model_name in models:
                            tensor = models[model_name].state_dict()[param_name]
                            tensors_to_merge.append(tensor)
                    
                    if tensors_to_merge:
                        modified_params = method_params
                        
                        # Get merge instance and merge tensors
                        merge_instance = self._get_merge_instance(method_name)
                        merged_tensor = merge_instance.merge_tensor(
                            base_tensor, tensors_to_merge, modified_params
                        )
                        
                        # Update the merged model
                        merged_model.state_dict()[param_name].copy_(merged_tensor)
                        
                        if debug_mode:
                            print(f"Merged special parameter: {param_name} using {method_name} method for component {special_component}")
                            if method_name in ['linear', 'slerp']:
                                weights_info = method_params.get('weights', [])
                                print(f"  Weights used: {weights_info}")
                    else:
                        if debug_mode:
                            print(f"No valid tensors to merge for parameter: {param_name}")
                            
                except Exception as e:
                    print(f"Failed to merge special parameter {param_name} in component {special_component}: {str(e)}")
                    if debug_mode:
                        import traceback
                        traceback.print_exc()
                    
            if debug_mode:
                print(f"Completed processing special component: {special_component}")

    def _merge_other_components_v2(self, merged_model: SamModel, models: Dict[str, SamModel],
                                  config: Dict[str, Any], debug_mode: bool):
        """Merge prompt_encoder and mask_decoder components using new group-based configuration"""
        
        # Process prompt_encoder groups
        prompt_encoder_config = config.get('prompt_encoder', {})
        if prompt_encoder_config:
            logger.info(f"\nProcessing Prompt Encoder with {len(prompt_encoder_config)} groups")
            
            for group_name, group_config in prompt_encoder_config.items():
                if group_name not in self.prompt_encoder_mapping:
                    if debug_mode:
                        print(f"Warning: Unknown prompt encoder group {group_name}")
                    continue
                
                method_name = group_config.get('method', 'linear')
                method_params = group_config.get('params', {'weights': [1.0]})
                sources = group_config.get('sources', [])
                
                # Get parameter names for this group
                param_names = self.prompt_encoder_mapping[group_name]
                
                logger.info(f"  Processing group '{group_name}' with {len(param_names)} parameters using {method_name}, source {sources}")
                
                # Process each parameter in this group
                for param_name in param_names:
                    # Check if this parameter exists in the model
                    if param_name not in self.base_model.state_dict():
                        if debug_mode:
                            print(f"Parameter {param_name} not found in model, skipping")
                        continue
                        
                    try:
                        # Get base tensor
                        base_tensor = self.base_model.state_dict()[param_name]
                        tensors_to_merge = []
                        
                        # Collect tensors from source models
                        for source in sources:
                            model_name = source['model']
                            if model_name in models:
                                tensor = models[model_name].state_dict()[param_name]
                                tensors_to_merge.append(tensor)
                        
                        if tensors_to_merge:
                            # Get merge instance and merge tensors
                            merge_instance = self._get_merge_instance(method_name)
                            merged_tensor = merge_instance.merge_tensor(
                                base_tensor, tensors_to_merge, method_params
                            )
                            
                            # Update the merged model
                            merged_model.state_dict()[param_name].copy_(merged_tensor)
                            
                            logger.info(f"    Merged parameter: {param_name}")
                        else:
                            logger.info(f"    No valid tensors to merge for parameter: {param_name}")
                                
                    except Exception as e:
                        print(f"Failed to merge prompt_encoder parameter {param_name} in group {group_name}: {str(e)}")
                        if debug_mode:
                            import traceback
                            traceback.print_exc()
                
                logger.info(f"  Completed processing prompt encoder group: {group_name}")
        
        # Process mask_decoder groups
        mask_decoder_config = config.get('mask_decoder', {})
        if mask_decoder_config:
            logger.info(f"\nProcessing Mask Decoder with {len(mask_decoder_config)} groups")
            
            for group_name, group_config in mask_decoder_config.items():
                if group_name not in self.mask_decoder_mapping:
                    logger.info(f"Warning: Unknown mask decoder group {group_name}")
                    continue
                
                method_name = group_config.get('method', 'linear')
                method_params = group_config.get('params', {'weights': [1.0]})
                sources = group_config.get('sources', [])
                
                # Get parameter names for this group
                param_names = self.mask_decoder_mapping[group_name]
                
                logger.info(f"  Processing group '{group_name}' with {len(param_names)} parameters using {method_name}")
                
                # Process each parameter in this group
                for param_name in param_names:
                    # Check if this parameter exists in the model
                    if param_name not in self.base_model.state_dict():
                        logger.info(f"Parameter {param_name} not found in model, skipping")
                        continue
                        
                    try:
                        # Get base tensor
                        base_tensor = self.base_model.state_dict()[param_name]
                        tensors_to_merge = []
                        
                        # Collect tensors from source models
                        for source in sources:
                            model_name = source['model']
                            if model_name in models:
                                tensor = models[model_name].state_dict()[param_name]
                                tensors_to_merge.append(tensor)
                        
                        if tensors_to_merge:
                            # Get merge instance and merge tensors
                            merge_instance = self._get_merge_instance(method_name)
                            merged_tensor = merge_instance.merge_tensor(
                                base_tensor, tensors_to_merge, method_params
                            )
                            
                            # Update the merged model
                            merged_model.state_dict()[param_name].copy_(merged_tensor)
                            
                            logger.info(f"    Merged parameter: {param_name}")
                        else:
                            if debug_mode:
                                print(f"    No valid tensors to merge for parameter: {param_name}")
                                
                    except Exception as e:
                        logger.info(f"Failed to merge mask_decoder parameter {param_name} in group {group_name}: {str(e)}")
                        if debug_mode:
                            import traceback
                            traceback.print_exc()
                
                logger.info(f"  Completed processing mask decoder group: {group_name}")

    def _merge_other_components(self, merged_model: SamModel, models: Dict[str, SamModel],
                              config: Dict[str, Any], debug_mode: bool):
        """Legacy method - kept for backward compatibility"""
        for component_name in ['prompt_encoder', 'mask_decoder']:
            component_config = config.get(component_name, {})
            if not component_config:
                continue
            
            method_name = component_config.get('method', 'linear')
            method_params = component_config.get('params', {'weights': [1.0]})
            sources = component_config.get('sources', [])
            
            component_params = self.layer_structure[component_name]
            
            for param_name in component_params:
                try:
                    base_tensor = self.base_model.state_dict()[param_name]
                    tensors_to_merge = []
                    
                    for source in sources:
                        model_name = source['model']
                        if model_name in models:
                            tensor = models[model_name].state_dict()[param_name]
                            tensors_to_merge.append(tensor)
                    
                    if tensors_to_merge:
                        merge_instance = self._get_merge_instance(method_name)
                        merged_tensor = merge_instance.merge_tensor(
                            base_tensor, tensors_to_merge, method_params
                        )
                        merged_model.state_dict()[param_name].copy_(merged_tensor)
                        
                        if debug_mode:
                            print(f"Merged {component_name} parameter: {param_name}")
                            
                except Exception as e:
                    print(f"Failed to merge {component_name} parameter {param_name}: {str(e)}")

def save_layer_config(config: Dict[str, Any], config_path: str) -> None:
    """Save layer-wise configuration to YAML file"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Layer configuration saved to: {config_path}")

def load_layer_config(config_path: str) -> Dict[str, Any]:
    """Load layer-wise configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"Layer configuration loaded from: {config_path}")
    return config

def validate_model_size_consistency(original_model: SamModel, merged_model: SamModel) -> bool:
    """Validate that merged model has same size as original"""
    orig_state = original_model.state_dict()
    merged_state = merged_model.state_dict()
    
    if set(orig_state.keys()) != set(merged_state.keys()):
        print("Error: Parameter names don't match")
        return False
    
    for key in orig_state.keys():
        if orig_state[key].shape != merged_state[key].shape:
            print(f"Error: Shape mismatch for {key}: {orig_state[key].shape} vs {merged_state[key].shape}")
            return False
    
    print("✓ Model size consistency validated")
    return True

# Usage examples
if __name__ == "__main__":
    pass