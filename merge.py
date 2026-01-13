import argparse
from utils import logger,seed_everything, load_config

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Strategy Application")
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = load_config(args.config) 
    main(config)
    # eval(config)
