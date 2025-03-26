from .disco_clip import Gather
from .gather_utils import all_gather, all_gather_with_grad
import src.utils as utils
from dataclasses import dataclass
from typing import Hashable, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist
import pandas as pd
import numpy as np
from transformers.modeling_outputs import ModelOutput

logger = utils.get_logger()


@dataclass
class ContrastiveLossOutput(ModelOutput):
    loss: Optional[Tensor] = None
    r_at_1: Optional[float] = None


def get_logits(
    query_features: Tensor,
    passage_features: Tensor,
    query_mask: Optional[Tensor] = None,
    passage_mask: Optional[Tensor] = None,
    logit_scale: Optional[Tensor] = None,
):
    logits = torch.matmul(
        query_features, passage_features.transpose(-1, -2))
    if logit_scale is not None:
        logits = logits * logit_scale
    return logits


def get_logits_fine_grained(
    query_features: Tensor,
    passage_features: Tensor,
    query_mask: Optional[Tensor] = None,
    passage_mask: Optional[Tensor] = None,
    logit_scale: Optional[Tensor] = None,
):
    # (Nq, Np, Lp, Lq)
    logits = torch.matmul(passage_features.unsqueeze(
        0), query_features.permute(0, 2, 1).unsqueeze(1))

    if passage_mask is not None:
        # (Np, Lp) -> (Nq, Np, Lp, Lq)
        passage_mask = utils.get_additive_attn_mask(
            passage_mask, query_features.dtype)[None, :, :, None]
        logits = logits + passage_mask

    # (Nq, Np, Lq)
    logits = logits.max(-2).values

    if query_mask is not None:
        # (Nq, Lq) -> (Nq, Np, Lq)
        query_mask = query_mask[:, None, :]
        logits = logits * query_mask

    # (Nq, Np)
    if logit_scale is not None:
        return logits.sum(-1) * logit_scale
    else:
        return logits.sum(-1)


def get_labels_contrastive_loss(
    Nq: int,
    Np: int,
    dtype,
    device,
    label_ids: Sequence[Hashable]
) -> Tuple[Tensor, bool]:
    all_label_ids = [None] * dist.get_world_size()
    dist.all_gather_object(all_label_ids, label_ids)
    all_label_ids = [y for x in all_label_ids for y in x]
    unique_ids = len(set(all_label_ids))
    labels = torch.zeros((Nq, Np), dtype=dtype,
                         device=device)
    if len(all_label_ids) > unique_ids:
        logger.info(f"Found batch with {unique_ids}/{len(all_label_ids)} unique entities.")
        all_label_ids = pd.Series(all_label_ids)
        label2idx = {k: torch.from_numpy(
            np.array(grp.index)) for k, grp in all_label_ids.groupby(by=all_label_ids, sort=False)}
        for i, label_id in enumerate(label_ids):
            labels[i, label2idx[label_id]] = 1
        labels = labels / labels.sum(1, keepdim=True)
        ret = labels, True
    else:
        rank = dist.get_rank()
        rank_start = rank * Nq
        rank_end = rank_start + Nq
        labels[:, rank_start:rank_end] = labels[:,
                                                rank_start:rank_end].fill_diagonal_(1)
        ret = labels, False

    return ret


def contrastive_loss(
    query_features: Tensor,
    passage_features: Tensor,
    logit_scale: Optional[Tensor] = None,
    label_ids: Optional[Sequence[Hashable]] = [],
    query_mask: Optional[Tensor] = None,
    passage_mask: Optional[Tensor] = None,
    simmetric: Optional[bool] = True,
    fine_grained: Optional[bool] = True
) -> ContrastiveLossOutput:
    gather_fn = Gather if simmetric else all_gather_with_grad
    logits_fn = get_logits_fine_grained if fine_grained else get_logits
    rank = dist.get_rank()
    bsz = query_features.size(0)

    if passage_mask is not None:
        passage_mask = all_gather(passage_mask)
    passage_features = gather_fn(passage_features)
    Nq, Np = bsz, passage_features.size(0)

    labels, has_conflicts = get_labels_contrastive_loss(
        Nq, Np, query_features.dtype, query_features.device, label_ids)
    logits_q = logits_fn(query_features, passage_features,
                         None if query_mask is None else query_mask, passage_mask, logit_scale)
    loss_q = F.cross_entropy(logits_q, labels)

    if simmetric:
        rank_start = rank * bsz
        rank_end = rank_start + bsz
        query_features = gather_fn(query_features)
        if query_mask is not None:
            query_mask = all_gather(query_mask)
        # label matrix is symmetric
        logits_p = logits_fn(passage_features[rank_start:rank_end], query_features,
                             None if passage_mask is None else passage_mask[rank_start:rank_end], query_mask, logits_fn)
        loss_p = contrastive_loss(logits_p, labels, simmetric=False)

        loss = (loss_q + loss_p) / 2
    else:
        loss = loss_q

    with torch.no_grad():
        r_at_1 = (labels[:, logits_q.argmax(1)] != 0).any(-1)
    all_r_at_1 = None
    if rank == 0:
        all_r_at_1 = [torch.zeros_like(r_at_1)
                      for _ in range(dist.get_world_size())]
    dist.gather(r_at_1, gather_list=all_r_at_1, dst=0)
    if rank == 0:
        all_r_at_1 = torch.concat(all_r_at_1)
        r_at_1 = all_r_at_1.float().mean().item()
    else:
        r_at_1 = None

    return ContrastiveLossOutput(
        loss=loss,
        r_at_1=r_at_1
    )
