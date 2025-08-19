from abc import ABC, abstractmethod

class BaseFeature(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    @abstractmethod
    def calculate(self):
        pass
    
    def get_feature(self, start, end, pairs):
        output = self.calculate(start, end, pairs, *self.args, **self.kwargs)
        return output