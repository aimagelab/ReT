from dataclasses import dataclass
from .base_config import BaseConfig
from .settings import *
from .core_config import DefaultVal


@dataclass
class RunConfig(BaseConfig, RunSettings):
    pass


@dataclass
class ColBERTConfig(RunSettings, ResourceSettings, DocSettings, QuerySettings, TrainingSettings,
                    IndexingSettings, SearchSettings, BaseConfig, TokenizerSettings):
    checkpoint_path: str = DefaultVal(None)
