from .base_method import MergeMethod


class PassthroughMerging(MergeMethod):
    def merge_tensor(
        self, 
        base_tensor, 
        tensors_to_merge, 
        method_params, 
        mask_merging=None,
        tensor_name="default"
    ):
        if len(tensors_to_merge) != 1:
            raise RuntimeError("Passthrough merge expects exactly one tensor.")

        merging_tensor = tensors_to_merge[0]
        scale = method_params.get("scale", None)
        if scale is not None:
            merging_tensor = merging_tensor * scale

        return merging_tensor
