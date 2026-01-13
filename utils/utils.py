import re
import os
import json
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["DATA_SEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_param_names_to_merge(input_param_names, exclude_param_names_regex):
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([
            re.match(exclude_pattern, param_name)
            for exclude_pattern
            in exclude_param_names_regex
        ])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge