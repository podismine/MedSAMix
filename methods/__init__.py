from .linear import LinearMerging
from .slerp import SlerpMerging
from .task_arithmetic import TaskArithmetic
from .ties import TiesMerging
from .passthrough import PassthroughMerging

merging_methods_dict = {
    "linear": LinearMerging,
    "slerp": SlerpMerging,
    "task_arithmetic": TaskArithmetic,
    "ties": TiesMerging,
    "passthrough": PassthroughMerging,
}


__all__ = ["merging_methods_dict"]