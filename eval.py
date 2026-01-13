import argparse
from utils import logger,seed_everything, load_config
import os
def get_merge_strategy(strategy_name, config):
    from strategies import strategy_classes
    strategy_class = strategy_classes.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return strategy_class(config)  


def main(config):
    logger.info("start merging")
    print(config)
    print(config.get("random_seed"))
    seed_everything(config.get("random_seed"))
    selected_strategy = config.get('strategy')

    merge_strategy_instance = get_merge_strategy(selected_strategy, config)
    merge_strategy_instance.merge()
    
def eval(config):
    logger.info("start merging")
    print(config)
    print(config.get("random_seed"))
    seed_everything(config.get("random_seed"))
    selected_strategy = config.get('strategy')

    merge_strategy_instance = get_merge_strategy(selected_strategy, config)

    import json
    import glob
    path = config['output_path']
    has_folder = [f for f in glob.glob(os.path.join(path, "*")) if os.path.isdir(f)]
    latest_folder = max(has_folder, key=os.path.getmtime)
    print(f"reading from latest_folder: {latest_folder}")

    intensifier = glob.glob(f"{latest_folder}/0/intensifier.json")
    with open(intensifier[0], 'r') as f:
        best_id = json.load(f)['incumbent_ids'][0]
        print(f"reading from intensifier: {best_id}")

    json_file = glob.glob(f"{latest_folder}/0/runhistory.json")
    with open(json_file[0], 'r') as f:
        run_config = json.load(f)['configs'][str(best_id)]

    merge_strategy_instance.eval_config(run_config)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Strategy Application")
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = load_config(args.config) 
    # main(config)
    eval(config)
