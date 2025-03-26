from dataclasses import dataclass
import math
from .configuration_ret import RetConfig
from typing import List, Sequence, Optional, Tuple
from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor
)
from transformers.modeling_outputs import ModelOutput
import torch.nn as nn
import torch.nn.functional as F
import torch
import src.utils as utils
from PIL import Image

logger = utils.get_logger()


class RetCell(nn.Module):
    def init_queries(self):
        self.query_hidden_states = nn.Parameter(torch.FloatTensor(
            1, self.config.num_queries, self.config.hidden_size))
        nn.init.normal_(self.query_hidden_states, mean=0.0, std=0.2)
        position = torch.arange(self.config.hidden_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, self.config.hidden_size, 2) * (-math.log(10000.0) / self.config.hidden_size))
        pe = torch.zeros(1, self.config.hidden_size, self.config.hidden_size)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            'query_state_ids', pe[:, :self.config.num_queries, :])

    def __init__(self, config: RetConfig):
        super().__init__()
        self.config = config

        self.init_queries()

        self.sa = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=self.config.num_attention_heads, 
                                        dropout=self.config.attention_dropout, batch_first=True)
        self.text_xa = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=self.config.num_attention_heads,
                                             dropout=self.config.attention_dropout, batch_first=True, bias=False)
        self.vision_xa = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=self.config.num_attention_heads,
                                               dropout=self.config.attention_dropout, batch_first=True, bias=False)

        if self.config.activation_fn == 'gelu':
            act_cls = nn.GELU
        elif self.config.activation_fn == 'relu':
            act_cls = nn.ReLU
        else:
            raise NotImplementedError()

        self.mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.Dropout(self.config.dropout_p),
            act_cls(),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size),
            nn.Dropout(self.config.dropout_p)
        )

        self.text_input_gate_w = nn.Linear(
            self.config.hidden_size, self.config.hidden_size)
        self.vision_input_gate_w = nn.Linear(
            self.config.hidden_size, self.config.hidden_size)
        self.text_forget_gate_w = nn.Linear(
            self.config.hidden_size, self.config.hidden_size)
        self.vision_forget_gate_w = nn.Linear(
            self.config.hidden_size, self.config.hidden_size)

        self.xa_ln = nn.LayerNorm(self.config.hidden_size)
        self.text_ln = nn.LayerNorm(self.config.hidden_size)
        self.vision_ln = nn.LayerNorm(self.config.hidden_size)
        self.sa_ln = nn.LayerNorm(self.config.hidden_size)
        self.mlp_ln = nn.LayerNorm(self.config.hidden_size)

        self.register_buffer('input_gate_bias_prior', torch.tensor(
            self.config.input_gate_bias_prior))
        self.register_buffer('forget_gate_bias_prior', torch.tensor(
            self.config.forget_gate_bias_prior))

        if self.config.use_tanh:
            self.state_act_fn = nn.Tanh()
        else:
            self.state_act_fn = nn.Identity()

    def forward(
        self,
        query_hidden_states: Optional[torch.Tensor] = None,
        text_key_value: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        vision_key_value: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = text_key_value.size(
            0) if text_key_value is not None else vision_key_value.size(0)
        dtype = text_key_value.dtype if text_key_value is not None else vision_key_value.dtype

        if query_hidden_states is None:
            # first layer
            query_hidden_states = self.query_hidden_states.expand(N, -1, -1)

        query_hidden_states = query_hidden_states + \
            self.query_state_ids.expand(N, -1, -1)

        residual = query_hidden_states

        query_hidden_states = self.xa_ln(query_hidden_states)

        residual = residual + \
            self.sa(query_hidden_states, query_hidden_states,
                    query_hidden_states, need_weights=False)[0]

        text_key_value = self.text_ln(text_key_value)
        if text_attention_mask is not None:
            text_attention_mask = utils.get_additive_attn_mask(text_attention_mask, dtype).unsqueeze(1).expand(
                N, self.config.num_queries, -1).repeat_interleave(self.config.num_attention_heads, dim=0)
        text_hidden_states = self.text_xa(
            query_hidden_states, text_key_value, text_key_value, attn_mask=text_attention_mask, need_weights=False)[0]

        vision_key_value = self.vision_ln(vision_key_value)
        vision_hidden_states = self.vision_xa(
            query_hidden_states, vision_key_value, vision_key_value, need_weights=False)[0]

        text_input_gate = F.sigmoid(self.text_input_gate_w(
            text_hidden_states) + self.input_gate_bias_prior)
        text_forget_gate_proj = self.text_forget_gate_w(text_hidden_states)
        if text_mask is not None:
            text_mask = text_mask[:, None, None].to(dtype)
            text_input_gate = text_input_gate * text_mask
            text_forget_gate_proj = text_forget_gate_proj * text_mask

        vision_input_gate = F.sigmoid(self.vision_input_gate_w(
            vision_hidden_states) + self.input_gate_bias_prior)
        vision_forget_gate_proj = self.vision_forget_gate_w(
            vision_hidden_states)
        if image_mask is not None:
            image_mask = image_mask[:, None, None].to(dtype)
            vision_input_gate = vision_input_gate * image_mask
            vision_forget_gate_proj = vision_forget_gate_proj * image_mask

        forget_gate = F.sigmoid(
            text_forget_gate_proj + vision_forget_gate_proj + self.forget_gate_bias_prior)

        # feature fusion
        query_hidden_states = self.state_act_fn(residual) * forget_gate +\
            self.state_act_fn(text_hidden_states) * text_input_gate +\
            self.state_act_fn(vision_hidden_states) * vision_input_gate

        residual = query_hidden_states
        query_hidden_states = self.mlp(self.mlp_ln(query_hidden_states))

        return residual + query_hidden_states


@dataclass
class RetModelOutput(ModelOutput):
    text_last_hidden_state: Optional[torch.Tensor] = None
    text_pooler_output: Optional[torch.Tensor] = None
    vision_last_hidden_state: Optional[torch.Tensor] = None
    vision_pooler_output: Optional[torch.Tensor] = None
    ret_features: Optional[torch.Tensor] = None


class RetModel(PreTrainedModel):
    config_class = RetConfig

    def __init__(
        self,
        config: RetConfig,
        text_model: Optional[PreTrainedModel] = None,
        vision_model: Optional[PreTrainedModel] = None
    ):
        super().__init__(config)
        self.config = config

        if text_model is None and config.text_config is not None:
            text_model = AutoModel.from_config(config.text_config)
        self.text_model = text_model

        if vision_model is None and config.vision_config is not None:
            vision_model = AutoModel.from_config(config.vision_config)
        self.vision_model = vision_model

        if config.text_hidden_size:
            self.text_adapter = nn.ModuleList([nn.Linear(config.text_hidden_size, config.hidden_size) for _ in range(config.n_rec_layers)])
        else:
            self.text_adapter = nn.Identity()

        if config.vision_hidden_size is not None:
            self.vision_adapter = nn.ModuleList([nn.Linear(config.vision_hidden_size, config.hidden_size) for _ in range(config.n_rec_layers)])
        else:
            self.vision_adapter = nn.Identity()

        self.ret_cell = RetCell(config)

        if self.config.late_proj_output_size:
            self.late_proj = nn.Linear(config.hidden_size, config.late_proj_output_size, bias=False)
        else:
            self.late_proj = nn.Identity()            

        self.tokenizer = None
        self.image_processor = None

    def freeze_modules(self):
        if self.config.is_text_frozen:
            self.text_model.requires_grad_(False)
        else:
            self.text_model.requires_grad_(True)

        if self.config.is_vision_frozen:
            self.vision_model.requires_grad_(False)
        else:
            self.vision_model.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.is_text_frozen:
            self.text_model.eval()
        if self.config.is_vision_frozen:
            self.vision_model.eval()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> RetModelOutput:
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)

        text_hidden_states = text_outputs.hidden_states[1:]
        selected_text_hidden_states = [text_hidden_states[idx] for idx in self.config.text_layer_idxs]

        vision_hidden_states = vision_outputs.hidden_states[1:]
        selected_vision_hidden_states = [vision_hidden_states[idx] for idx in self.config.vision_layer_idxs]

        ret_features = None
        for i, (txths, vishs, txtproj, visproj) in enumerate(zip(
            selected_text_hidden_states,
            selected_vision_hidden_states,
            self.text_adapter,
            self.vision_adapter
        )):
            ret_features = self.ret_cell(
                query_hidden_states=ret_features,
                text_key_value=txtproj(txths),
                text_attention_mask=attention_mask,
                vision_key_value=visproj(vishs),
                text_mask=text_mask,
                image_mask=image_mask
            )

        if self.config.use_pooler_features:
            ret_features = ret_features.mean(dim=1)
        ret_features = self.late_proj(ret_features)

        return RetModelOutput(
            text_last_hidden_state=text_outputs.last_hidden_state,
            text_pooler_output=text_outputs.pooler_output,
            vision_last_hidden_state=vision_outputs.last_hidden_state,
            vision_pooler_output=vision_outputs.pooler_output,
            ret_features=ret_features
        )

    def init_tokenizer_and_image_processor(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_config.name_or_path)

        if self.image_processor is None:
            self.image_processor = AutoImageProcessor.from_pretrained(self.config.vision_config.name_or_path)

    def get_ret_features(
        self, 
        docs: Sequence[Tuple[str, str | None]],
        to_cpu: bool = False
    ) -> Tuple[torch.Tensor, List[int]]:
        txts, text_mask, imgs, image_mask = list(zip(*list(map(lambda x: (
            x[0], 
            1 if x[0] else 0, 
            Image.open(x[1]) if x[1] else Image.new('RGB', (336, 336), color='black'),
            1 if x[1] else 0
        ), docs))))

        with torch.inference_mode():
            input_ids, attention_mask = self.tokenizer(txts, return_tensors='pt', padding=True, truncation=True).to(self.device).values()
            pixel_values = self.image_processor(imgs, return_tensors='pt').to(dtype=self.dtype, device=self.device).pixel_values
            text_mask = torch.tensor(text_mask, dtype=self.dtype, device=self.device)
            image_mask = torch.tensor(image_mask, dtype=self.dtype, device=self.device)
            
            ret_feats = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                text_mask=text_mask,
                image_mask=image_mask
            ).ret_features
            
            ret_feats = F.normalize(ret_feats, p=2, dim=-1)
        
        if to_cpu:
            ret_feats = ret_feats.cpu()

        return ret_feats
    
    def docFromText(
        self, 
        docs: Sequence[Tuple[str, str | None]],
        to_cpu: bool = False
    ) -> Tuple[torch.Tensor, List[int]]:
        doc_feats = self.get_ret_features(docs, to_cpu)

        doc_lens = [doc_feats.size(1)] * doc_feats.size(0)
        doc_feats = doc_feats.flatten(0, 1)

        return (doc_feats, doc_lens)

    def queryFromText(
        self, 
        docs: Sequence[Tuple[str, str | None]],
        to_cpu: bool = False
    ) -> Tuple[torch.Tensor, List[int]]:
        return self.get_ret_features(docs, to_cpu)