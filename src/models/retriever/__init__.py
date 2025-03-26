from .configuration_retriever import RetrieverConfig
from .modeling_retriever import RetrieverModel
from transformers import AutoConfig, AutoModel

__all__ = [
    'RetrieverConfig',
    'RetrieverModel'
]

AutoConfig.register(RetrieverConfig.model_type, RetrieverConfig)
AutoModel.register(RetrieverConfig, RetrieverModel)