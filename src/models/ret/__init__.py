from .configuration_ret import RetConfig, RetLayerStrategy
from .modeling_ret import RetModel, RetModelOutput
from transformers import AutoConfig, AutoModel

__all__ = [
    'RetLayerStrategy',
    'RetConfig',
    'RetModel',
    'RetModelOutput'
]

AutoConfig.register(RetConfig.model_type, RetConfig)
AutoModel.register(RetConfig, RetModel)