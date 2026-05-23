from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """
    Base class for all models.

    The reason why we need a model layer within this framework (instead of directly using standard ML libraries in training/evaluation scripts)
    is thatwe need a layer to abstract away financial data fetching and ML data transformation on top of it.

    """
    def __init__(self):
        pass

    @abstractmethod
    def train(self, start_datestr: str, end_datestr: str):
        raise NotImplementedError

    @abstractmethod
    def predict(self, start_datestr: str, end_datestr: str):
        raise NotImplementedError

    @abstractmethod
    def test(self, start_datestr: str, end_datestr: str) -> dict:
        raise NotImplementedError
