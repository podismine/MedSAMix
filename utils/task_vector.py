import torch

from .utils import get_param_names_to_merge


class TaskVector:
    def __init__(
        self,
        pretrained_model=None,
        finetuned_model=None,
        exclude_param_names_regex=None,
        task_vector_param_dict=None
    ):
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {
                param_name: param_value 
                for param_name, param_value 
                in pretrained_model.named_parameters()
            }
            finetuned_param_dict = {
                param_name: param_value
                for param_name, param_value
                in finetuned_model.named_parameters()
            }
            param_names_to_merge = get_param_names_to_merge(
                input_param_names=list(pretrained_param_dict.keys()), 
                exclude_param_names_regex=exclude_param_names_regex
            )
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = (
                        finetuned_param_dict[param_name] -
                        pretrained_param_dict[param_name]
                    )

    def __add__(self, other):
        assert isinstance(other, TaskVector), \
            "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), \
                    f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = (
                    self.task_vector_param_dict[param_name] + 
                    other.task_vector_param_dict[param_name]
                )
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        return self.__add__(other)

    def combine_with_pretrained_model(
        self,
        pretrained_model,
        scaling_coefficient=1.0
    ):
        pretrained_param_dict = {
            param_name: param_value 
            for param_name, param_value 
            in pretrained_model.named_parameters()
        }
        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = (
                    pretrained_param_dict[param_name] + 
                    scaling_coefficient * self.task_vector_param_dict[param_name]
                )
        return merged_params

    def combine_with_base_tensor(
        self, 
        base_tensor, 
        scaling_coefficient=1.0
    ):
        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = (
                    base_tensor[param_name] + 
                    scaling_coefficient * self.task_vector_param_dict[param_name]
                )
        return merged_params
        

