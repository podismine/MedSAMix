import torch

from .base_merge_method import MergeMethod


class AverageMerging(MergeMethod):
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        merged_tensor = torch.stack(
            [
                merging_tensor 
                for merging_tensor in tensors_to_merge
            ],
            dim=0
        ).mean(dim=0)
        return merged_tensor
        
    