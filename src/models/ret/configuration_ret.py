from typing import Dict, Optional, Union
from transformers import PretrainedConfig, AutoConfig
from enum import Enum

class RetLayerStrategy(str, Enum):
    CLIP_VIT_B = 'clip_vit_b'
    CLIP_VIT_L = 'clip_vit_l'
    OPENCLIP_VIT_H = 'openclip_vit_h'
    OPENCLIP_VIT_G = 'openclip_vit_g'


_RET_LAYER_STRATEGY_MAPPING = {
    RetLayerStrategy.CLIP_VIT_B: {
        'text': tuple(range(12)),
        'vision': tuple(range(12))
    },
    RetLayerStrategy.CLIP_VIT_L: {
        'text': list(range(12)),
        'vision': list(range(24))[::2]
    },
    RetLayerStrategy.OPENCLIP_VIT_H: {
        'text': (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23),
        'vision': (0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31)
    },
    RetLayerStrategy.OPENCLIP_VIT_G: {
        'text': (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 31),
        'vision': (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47)
    }
}


class RetConfig(PretrainedConfig):
    model_type = 'ret'
    is_composition = False
    
    def __init__(
        self,
        text_config: Optional[Union[PretrainedConfig, Dict]] = None,
        vision_config: Optional[Union[PretrainedConfig, Dict]] = None,
        is_text_frozen: bool = True,
        is_vision_frozen: bool = True,
        late_proj_output_size: int = 128,
        layer_strategy: str = RetLayerStrategy.CLIP_VIT_B,
        use_pooler_features: bool = False,
        num_queries: int = 32,
        hidden_size: int = 1024,
        dropout_p: float = 0.05,
        attention_dropout: float = 0.05,
        activation_fn: str = 'gelu',
        input_gate_bias_prior: float = 0.0,
        forget_gate_bias_prior: float = 0.0,
        use_tanh: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            text_config = AutoConfig.for_model(text_config.pop('model_type'), **text_config)
        self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_config = AutoConfig.for_model(vision_config.pop('model_type'), **vision_config)
        self.vision_config = vision_config    

        self.is_text_frozen = is_text_frozen
        self.is_vision_frozen = is_vision_frozen
        self.late_proj_output_size = late_proj_output_size
        self.layer_strategy = layer_strategy
        self.use_pooler_features = use_pooler_features
        
        # recurrent cell
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        self.num_attention_heads = hidden_size // 64        
        self.dropout_p = dropout_p
        self.attention_dropout = attention_dropout
        self.activation_fn = activation_fn
        self.input_gate_bias_prior = input_gate_bias_prior
        self.forget_gate_bias_prior = forget_gate_bias_prior
        self.use_tanh = use_tanh
    
    @property
    def n_rec_layers(self):
        return len(_RET_LAYER_STRATEGY_MAPPING[self.layer_strategy]['text'])
    
    @property
    def text_layer_idxs(self):
        return _RET_LAYER_STRATEGY_MAPPING[self.layer_strategy]['text']
    
    @property
    def vision_layer_idxs(self):
        return _RET_LAYER_STRATEGY_MAPPING[self.layer_strategy]['vision']
    
    @property
    def text_hidden_size(self):
        if self.text_config is None:
            return 0
        else:
            return self.text_config.hidden_size
        
    @property
    def vision_hidden_size(self):
        if self.vision_config is None:
            return 0
        else:
            return self.vision_config.hidden_size