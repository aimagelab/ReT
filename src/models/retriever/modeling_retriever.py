import os
from .configuration_retriever import RetrieverConfig
from src.models.contrastive_loss import contrastive_loss, ContrastiveLossOutput
from src.models.ret import RetModel
import src.utils as utils
from dataclasses import dataclass
from typing import Callable, Optional, List, Hashable, Union
from transformers import PreTrainedModel
import torch
import torch.nn.functional as F

logger = utils.get_logger()


@dataclass
class RetrieverModelOutput(ContrastiveLossOutput):
    query_features: Optional[torch.Tensor] = None
    passage_features: Optional[torch.Tensor] = None


class RetrieverModel(PreTrainedModel):
    config_class = RetrieverConfig

    def __init__(
        self,
        config: RetrieverConfig,
        query_model: Optional[RetModel] = None,
        passage_model: Optional[RetModel] = None
    ):
        super().__init__(config)

        if query_model is None:
            query_model = RetModel(config.query_config)
        self.query_model = query_model

        if config.share_query_passage_models:
            passage_model = query_model
        elif passage_model is None:
            passage_model = RetModel(config.passage_config)
        self.passage_model = passage_model

        self.freeze_modules()
        self.tie_backbones()

    def tie_backbones(self):
        if not self.config.share_query_passage_models:
            if self.config.share_text_models:
                self.passage_model.text_model = self.query_model.text_model
                self.config.passage_config.text_config = self.config.query_config.text_config
            if self.config.share_vision_models:
                self.passage_model.vision_model = self.query_model.vision_model
                self.config.passage_config.vision_config = self.config.query_config.vision_config

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        if self.config.share_query_passage_models:
            self.passage_model = None
        else:
            if self.config.share_text_models:
                self.passage_model.text_model = None
            if self.config.share_vision_models:
                self.passage_model.vision_model = None
        super().save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )            

    def get_query_model(self):
        return self.query_model

    def get_passage_model(self):
        if self.config.share_query_passage_models:
            return self.query_model
        return self.passage_model

    def freeze_modules(self):
        self.query_model.freeze_modules()
        self.passage_model.freeze_modules()

    def forward(
        self,
        query_pixel_values: Optional[torch.Tensor] = None,
        query_image_mask: Optional[torch.Tensor] = None,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        passage_pixel_values: Optional[torch.Tensor] = None,
        passage_input_ids: Optional[torch.Tensor] = None,
        passage_attention_mask: Optional[torch.Tensor] = None,
        passage_text_mask: Optional[torch.Tensor] = None,
        passage_image_mask: Optional[torch.Tensor] = None,
        labels: Optional[List[Hashable]] = None
    ):
        Q, P = self.get_query_model(), self.get_passage_model()

        query_outputs = Q(
            pixel_values=query_pixel_values,
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            image_mask=query_image_mask
        )
        passage_outputs = P(
            pixel_values=passage_pixel_values,
            input_ids=passage_input_ids,
            attention_mask=passage_attention_mask,
            text_mask=passage_text_mask,
            image_mask=passage_image_mask
        )

        query_features, passage_features = query_outputs.ret_features, passage_outputs.ret_features

        query_features_norm = F.normalize(query_features, p=2, dim=-1)
        passage_features_norm = F.normalize(passage_features, p=2, dim=-1)

        loss_outputs = contrastive_loss(
            query_features=query_features_norm,
            passage_features=passage_features_norm,
            label_ids=labels,
            fine_grained=self.config.fg_loss,
            simmetric=self.config.simmetric_loss
        )

        return RetrieverModelOutput(
            query_features=query_features,
            passage_features=passage_features,
            **loss_outputs
        )
