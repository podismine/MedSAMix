from loguru import logger
from datetime import datetime
import sys


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"output_{timestamp}.log"
logger.remove()  
logger.add(sys.stdout, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}")
logger.add(log_filename, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}")

def load_and_validate_config(config_file_path):
    config_path = Path(config_file_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    try:
        with config_path.open('r') as file:
            print("File opened successfully")
            config = yaml.load(file, Loader=ConfigLoader)
            print(f"Loaded config: {config}")
    except Exception as e:
        print(f"Error while loading YAML: {e}")
        raise

    if config is None:
        raise ValueError(f"Failed to load configuration from {config_file_path}")
    
    validate_config(config)
    selected_strategy = config.get('strategy')
    
    if not selected_strategy:
        raise ValueError("No strategy defined in the configuration file.")
    
    global_params = config.get('global_params', {})
    strategy_params = config.get('strategies', {}).get(selected_strategy, {})
    strategy_params.update({"strategy": selected_strategy})
    merged_config = {**global_params, **strategy_params}
    return merged_config


def validate_config(config):
    required_keys = ['strategy']
    valid_strategies = ['normal_models', 'normal_slices', 'evo_ps', 'evo_dfs', 'evo_dfs_fo', 'random_perturb', 'evo_dfs_dg', 'evo_dfs_scales', 'evo_dfs_fo_mo', 'evo_ps_mo', 'evo_dfs_fo_4prune', 'evo_dfs_fo_depth_4prune', 'evo_dfs_fo_depth_mo_4prune', 'evo_dfs_fix_order_mo_depth4prune_v1', 'evo_dfs_fix_order_mo_depth4prune_v2', 'evo_dfs_fix_order_mo_depth4prune_v3', 'evo_dfs_fix_order_mo_depth4prune_v0', 'evo_dfs_fix_order_mo_depth4prune_v5', 'evo_dfs_fix_order_mo_depth4prune_v7', 'evo_dfs_fix_order_mo_depth4prune_v8', 'evo_dfs_fix_order_mo_depth4prune_v9', 'evo_dfs_fix_order_mo_depth4prune_v11','evo_dfs_fix_order_mo_depth4prune_v527','evo_dfs_fix_order_mo_depth4prune_v528','evo_ps_evo', 'evo_dfs_fo_evo']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    strategy = config['strategy']
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy '{strategy}', must be one of {valid_strategies}")

    if strategy == 'normal_models':
        normal_models_required_keys = ['models']
        for key in normal_models_required_keys:
            if key not in config['strategies'][strategy]:
                raise ValueError(f"For 'normal_models' strategy, missing required key: {key}")

    elif strategy == 'normal_slices':
        normal_slices_required_keys = ['slices']
        for key in normal_slices_required_keys:
            if key not in config['strategies'][strategy]:
                raise ValueError(f"For 'normal_slices' strategy, missing required key: {key}")

    else:
        pass

    print("Config is valid!")
