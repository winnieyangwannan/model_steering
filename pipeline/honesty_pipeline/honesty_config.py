
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # model_alias: str
    # model_path: str
    # n_train: int = 64
    # n_test: int = 32
    # data_category: str = "facts"
    # batch_size = 16
    # source_layer = 14
    # intervention = "direction ablation"
    # target_layer: int = None

    model_alias: str
    model_path: str
    n_train: int = 32
    n_test: int = 32
    data_category: str = "facts"
    batch_size: int = 16
    source_layer: int = 10
    intervention: str = "diff addition"
    target_layer: int = 10
    max_new_tokens: int = 100
    # for generation_trajectory
    dataset_id: list[int] = 3


    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)