from abc import ABC, abstractmethod  # noqa: D100
import numpy as np
import typing as tp
import pandas as pd


class HallucinationDetectionDataset(ABC):
    '''Abstract class for specifying the interface of the called dataset.'''

    @abstractmethod
    def process(self) -> tuple[pd.DataFrame, pd.Series, np.ndarray, tp.Optional[np.ndarray]]:
        '''Process and return LLM Answers with hallucination label.'''
        pass

    

