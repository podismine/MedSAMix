from abc import ABC, abstractmethod


class MergeStrategy(ABC):
    def __init__(self, config):
        self.config = config
        
    @abstractmethod
    def merge(self):
        pass
        
