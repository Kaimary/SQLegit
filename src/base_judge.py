from abc import ABC, abstractmethod


class BaseJudge(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def set(self):
        pass

    @abstractmethod
    def run(self):
        pass