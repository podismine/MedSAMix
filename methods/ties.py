from collections import OrderedDict
import copy

import torch
import torch.nn as nn

from utils import TaskVector
from .base_method import MergeMethod


class TiesMerging(MergeMethod):
    def task_vector_param_dict_to_single_vector(self, task_vector):
        """
        convert parameter dictionary in task vector to a single vector
        """
        task_vector_param_dict = copy.deepcopy(
            task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(
            sorted(task_vector_param_dict.items()))

        # Tensor, shape (num_total_params, )
        return nn.utils.parameters_to_vector(
            [
                param.flatten() 
                for param in sorted_task_vector_param_dict.values()
            ]
        )

    def single_vector_to_task_vector_param_dict(
        self,
        single_vector,
        task_vector
    ):
        """
        convert a single vector to parameter dictionary in task vector
        """
        task_vector_param_dict = copy.deepcopy(
            task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(
            sorted(task_vector_param_dict.items()))

        nn.utils.vector_to_parameters(
            single_vector, sorted_task_vector_param_dict.values())

        return sorted_task_vector_param_dict

    def mask_smallest_magnitude_param_values(
        self, 
        flattened_models_to_merge_param, 
        param_value_mask_rate=0.8
    ):
        """
        mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
        """
        # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
        num_mask_params = int(
            flattened_models_to_merge_param.shape[1] * param_value_mask_rate
        )
        try:
            # find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(
                k=num_mask_params, dim=1, keepdim=True
            )
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            mask = flattened_models_to_merge_param.abs() >= kth_values
        except:
            mask = torch.ones_like(flattened_models_to_merge_param).bool()

        return flattened_models_to_merge_param * mask

    def get_param_signs(self, flattened_models_to_merge_param):
        """
        get the signs for each parameter in flattened_models_to_merge_param, 
        computed over individual models that need to be merged
        """
        # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        param_signs = torch.sign(
            flattened_models_to_merge_param.sum(dim=0)
        )
        # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
        majority_sign = torch.sign(param_signs.sum(dim=0))
        param_signs[param_signs == 0] = majority_sign
        return param_signs

    def disjoint_merge(
        self, 
        flattened_models_to_merge_param, 
        param_signs
    ):
        """
        disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, 
        and calculates the averaged parameters.
        """
        # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
        param_to_preserve_mask = (
            (param_signs.unsqueeze(dim=0) > 0) & 
            (flattened_models_to_merge_param > 0)
        ) | (
            (param_signs.unsqueeze(dim=0) < 0) & 
            (flattened_models_to_merge_param < 0)
        )
        # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
        param_to_preserve = (
            flattened_models_to_merge_param * param_to_preserve_mask
        )

        # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
        num_models_param_preserved = (
            param_to_preserve != 0
        ).sum(dim=0).float()
        # Tensor, shape (num_total_params, ), the averaged flattened parameters
        merged_flattened_param = torch.sum(
            param_to_preserve, dim=0
        ) / torch.clamp(num_models_param_preserved, min=1.0)

        return merged_flattened_param
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        scaling_coefficient = method_params["scaling_coefficient"]
        param_value_mask_rate = method_params["param_value_mask_rate"]
        base_tensor_dict = {tensor_name: base_tensor}
        
        models_to_merge_task_vectors = [
            TaskVector(
                task_vector_param_dict={
                    tensor_name: merging_tensor.to("cpu") - base_tensor.to("cpu")
                }
            )
            for merging_tensor in tensors_to_merge
        ]
        
        flattened_models_to_merge_param = [
            self.task_vector_param_dict_to_single_vector(
                task_vector=task_vector
            )
            for task_vector in models_to_merge_task_vectors
        ]
        # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        flattened_models_to_merge_param = torch.vstack(
            flattened_models_to_merge_param
        )
        
        with torch.no_grad():
            # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
            flattened_models_to_merge_param = self.mask_smallest_magnitude_param_values(
                flattened_models_to_merge_param=flattened_models_to_merge_param, 
                param_value_mask_rate=param_value_mask_rate
            )
            
            # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
            param_signs = self.get_param_signs(
                flattened_models_to_merge_param=flattened_models_to_merge_param
            )
            # Tensor, shape (num_total_params, ), disjoint merge
            merged_flattened_param = self.disjoint_merge(
                flattened_models_to_merge_param=flattened_models_to_merge_param, 
                param_signs=param_signs
            )
            # merged parameter dictionary
            merged_task_vector_param_dict = self.single_vector_to_task_vector_param_dict(
                single_vector=merged_flattened_param, 
                task_vector=models_to_merge_task_vectors[0]
            )
            merged_task_vector = TaskVector(
                task_vector_param_dict=merged_task_vector_param_dict
            )
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_base_tensor(
                base_tensor=base_tensor_dict, 
                scaling_coefficient=scaling_coefficient
            )
        return merged_params[tensor_name]