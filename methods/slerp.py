import torch

from .base_method import MergeMethod


class SlerpMerging(MergeMethod):
    def normalize(self, input_tensor, eps=1e-10):
        """
        normalize the input tensor
        """
        norm_input_tensor = torch.norm(input_tensor)
        if norm_input_tensor > eps:
            input_tensor = input_tensor / norm_input_tensor
        return input_tensor

    def lerp(self, v0, v1, slerp_t=0.5):
        """
        linear interpolation
        """
        return (1 - slerp_t) * v0 + slerp_t * v1
        
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        dot_threshold = 0.9995 #method_params["dot_threshold"]
        slerp_t = method_params["slerp_t"]    
        assert len(tensors_to_merge) == 2, "slerp merging expects exactly two models!"
        v0, v1 = tensors_to_merge[0], tensors_to_merge[1]

        # Tensor, copy the vectors for reusing
        v0_copy = v0.clone().detach()
        v1_copy = v1.clone().detach()

        # Tensor, normalize the input tensors
        v0 = self.normalize(input_tensor=v0)
        v1 = self.normalize(input_tensor=v1)

        # dot product with the normalized vectors
        dot = torch.sum(v0 * v1)

        # if absolute value of dot product larger than dot_threshold (almost 1), vectors are colinear, so use lerp
        if torch.abs(dot) > dot_threshold:
            merged_tensor = self.lerp(v0=v0_copy, v1=v1_copy, slerp_t=slerp_t)
            return merged_tensor

        # slerp(p, q, t) = \frac{sin((1-t) \theta) \cdot p + sin(t \theta) \cdot q}{sin(\theta)}
        # calculate initial angle between v0 and v1
        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0)

        # angle for hyperparameter slerp_t
        theta_t = theta_0 * slerp_t
        sin_theta_t = torch.sin(theta_t)

        # finish the slerp algorithm
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        merged_tensor = s0 * v0_copy + s1 * v1_copy
        return merged_tensor