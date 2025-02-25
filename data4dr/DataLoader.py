from pydantic import validate_call

from .data.model import TDataName, DataModel

from .data.mnistSeries import MnistSeriesLoader
from .data.celegans import CelegansLoader
from .data.uciml import UcimlLoader
from .data.textEmbeddings import TextEmbeddingLoader


class DataLoader:
    @validate_call
    def __init__(self, data_name: TDataName = "mnist"):
        self.data_name = data_name
        self.loader = self._get_loader()

    def _get_loader(self):
        loader_mapping = {
            "mnist": lambda name: MnistSeriesLoader(name),
            "fmnist": lambda name: MnistSeriesLoader(name),
            "kmnist": lambda name: MnistSeriesLoader(name),
            "celegans": lambda name: CelegansLoader(),
            "ionosphere": lambda name: UcimlLoader(name),
            "optical_recognition": lambda name: UcimlLoader(name),
            "raisin": lambda name: UcimlLoader(name),
            "htru2": lambda name: UcimlLoader(name),
            "20ng": lambda name: TextEmbeddingLoader(name),
            "ag_news": lambda name: TextEmbeddingLoader(name),
            "amazon_polarity": lambda name: TextEmbeddingLoader(name),
            "yelp_review": lambda name: TextEmbeddingLoader(name),
        }
        loader_provider = loader_mapping.get(self.data_name)
        if loader_provider is None:
            raise ValueError(f"Invalid type: {self.data_name}")
        return loader_provider(self.data_name)

    def get_data(self) -> DataModel:
        return self.loader.get_data()
