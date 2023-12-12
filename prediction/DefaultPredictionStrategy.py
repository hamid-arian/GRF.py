from abc import ABC, abstractmethod
from typing import List, Dict
from commons.Data import Data

namespace grf:

class DefaultPredictionStrategy(ABC):

    @abstractmethod
    def prediction_length(self) -> int:
        pass

    @abstractmethod
    def predict(self, sample: int, weights_by_sample: Dict[int, float], train_data: Data, data: Data) -> List[float]:
        pass

    @abstractmethod
    def compute_variance(self, sample: int, samples_by_tree: List[List[int]],
                         weights_by_sampleID: Dict[int, float], train_data: Data, data: Data, ci_group_size: int) -> List[float]:
        pass
