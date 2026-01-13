import os
from abc import ABC, abstractmethod



CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE', os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers"))


class MergeMethod(ABC):
    def __init__(self):
        pass  
    
    @abstractmethod
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        pass
