from models.base_model import BaseModel


class SingleAssetModel(BaseModel):
    """
    This class is responsible for training a single asset model.
    """
    def __init__(self, symbol: str):
        super().__init__()

        self._symbol = symbol
