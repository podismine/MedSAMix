import torch

from utils import TaskVector 
from .base_method import MergeMethod


class TaskArithmetic(MergeMethod):
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        scaling_coefficient = method_params["scaling_coefficient"]
        assert isinstance(scaling_coefficient, float), \
            "wrong type of scaling_coefficient, should be float!"
        base_tensor_dict = {tensor_name: base_tensor}
        models_to_merge_task_vectors = [
            TaskVector(
                task_vector_param_dict={
                    tensor_name: merging_tensor.to("cpu") - base_tensor.to("cpu")
                }
            )
            for merging_tensor in tensors_to_merge
        ]
        with torch.no_grad():
            # sum up the task vectors
            merged_task_vector = models_to_merge_task_vectors[0] + \
                models_to_merge_task_vectors[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + \
                    models_to_merge_task_vectors[index]
            
            merged_params = merged_task_vector.combine_with_base_tensor(
                base_tensor=base_tensor_dict,
                scaling_coefficient=scaling_coefficient
            )
        return merged_params[tensor_name]