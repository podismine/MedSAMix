import os
import yaml
import importlib
from pathlib import Path
import importlib.util
import sys



class ConfigLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self._base = Path(stream.name).parent

    def include(self, node):
        file_name = self.construct_scalar(node)
        file_path = self._base.joinpath(file_name)

        with file_path.open("rt") as fh:
            return yaml.load(fh, ConfigLoader)

    def function(self, node):
        value = self.construct_scalar(node)
     
        try:
            module_name, func_name = value.split('.')
            module_path = self._base.joinpath(f"{module_name}.py")
            
            if not module_path.exists():
                raise ImportError(f"can not found: {module_path}")

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            func = getattr(module, func_name)
            
            if not callable(func):
                raise AttributeError(f"{func_name} is not a function")
                
            return func

        except (ImportError, AttributeError) as e:
            raise yaml.constructor.ConstructorError(
                None, None,
                f"Cannot load function {value}: {str(e)}",
                node.start_mark
            )
        except Exception as e:
            raise yaml.constructor.ConstructorError(
                None, None,
                f"{value}: {str(e)}",
                node.start_mark
            )
            
ConfigLoader.add_constructor('!include', ConfigLoader.include)
ConfigLoader.add_constructor('!function', ConfigLoader.function)


def load_config(config_file_path):
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
    
    # validate_config(config)
    # selected_strategy = config.get('strategy')
    
    # if not selected_strategy:
    #     raise ValueError("No strategy defined in the configuration file.")
    
    global_params = config.get('global_params', {})
    # strategy_params = config.get('strategies', {}).get(selected_strategy, {})
    # strategy_params.update({"strategy": selected_strategy})
    # merged_config = {**global_params, **strategy_params}
    return config