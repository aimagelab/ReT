# from LAVIS codebase: https://github.com/salesforce/LAVIS/blob/main/lavis/models/base_model.py#L202
import torch
from torch.distributed.nn.functional import all_gather as all_gather_with_grad_torch_functional


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


def all_gather_with_grad_torch(tensors):
    """
    Official version of all_gather with grads in torch.distributed.nn.functional
    """
    tensor_all = all_gather_with_grad_torch_functional(tensors)
    return torch.cat(tensor_all, dim=0)


def all_gather(x):
    output = [
        torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(output, x)
    return torch.cat(output, dim=0)  