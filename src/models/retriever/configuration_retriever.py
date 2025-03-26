from typing import Dict, Optional, Union
from transformers import PretrainedConfig
from src.models.ret import RetConfig


class RetrieverConfig(PretrainedConfig):    
    model_type = 'retriever'
    is_composition = False

    def __init__(
        self,
        query_config: Optional[Union[RetConfig, Dict]] = None,
        passage_config: Optional[Union[RetConfig, Dict]] = None,
        fg_loss: bool = True,
        simmetric_loss: bool = True,
        share_query_passage_models: bool = False,
        share_text_models: bool = True,
        share_vision_models: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(query_config, RetConfig):
            query_config = query_config
        elif isinstance(query_config, dict):
            query_config = RetConfig.from_dict(query_config)
        self.query_config = query_config

        if isinstance(passage_config, RetConfig):
            passage_config = passage_config
        elif isinstance(passage_config, dict):
            passage_config = RetConfig.from_dict(passage_config)
        
        if share_query_passage_models:
            self.passage_config = self.query_config
        else:
            self.passage_config = passage_config

        self.fg_loss = fg_loss
        self.simmetric_loss = simmetric_loss
        self.share_query_passage_models = share_query_passage_models
        self.share_text_models = share_text_models
        self.share_vision_models = share_vision_models