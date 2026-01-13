from utils import logger
from .base_method import MergeMethod


class LinearMerging(MergeMethod):
   
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        # Ensure weights are provided and normalized
        weights = method_params["weights"]
        normalize = True #method_params["normalize"]
        if weights is None:
            weights = [1.0] * len(tensors_to_merge)  # Default to equal weights
        if normalize:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]  # Normalize weights
        logger.info(f"linear merge: current merge weights is {weights}")
        merged_tensor = sum(
            weight * tensor.to("cpu") 
            for weight, tensor in zip(weights, tensors_to_merge)
        )
        return merged_tensor
        