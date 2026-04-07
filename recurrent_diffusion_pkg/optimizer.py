from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import torch.nn.functional as F

def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


# class WeightNorm(object):
#     def __init__(self, names, dim, k=None):
#         """
#         Weight normalization module

#         :param names: The list of weight names to apply weightnorm on
#         :param dim: The dimension of the weights to be normalized
#         """
#         self.names = names
#         self.dim = dim
#         self.k = k

#     def compute_weight(self, module, name):
#         g = getattr(module, name + '_g')
#         v = getattr(module, name + '_v')
#         normalized_weight =  v * (g / _norm(v, self.dim))
#         if self.k is not None:
#             # enforce each row of teh weight matrix to have only top k largest (absolute) values activate, and zero the rest
#             # flatten the second dimension of weight matrix, sort the absolute values for each row
#             TODO


#     @staticmethod
#     def apply(module, names, dim, k=None):
#         fn = WeightNorm(names, dim, k)

#         for name in names:
#             weight = getattr(module, name)

#             # remove w from parameter list
#             del module._parameters[name]

#             # add g and v as new parameters and express w as g/||v|| * v
#             module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
#             module.register_parameter(name + '_v', Parameter(weight.data))
#             setattr(module, name, fn.compute_weight(module, name))

#         # recompute weight before every forward()
#         module.register_forward_pre_hook(fn)
#         return fn

#     def remove(self, module):
#         for name in self.names:
#             weight = self.compute_weight(module, name)
#             delattr(module, name)
#             del module._parameters[name + '_g']
#             del module._parameters[name + '_v']
#             module.register_parameter(name, Parameter(weight.data))

#     def reset(self, module):
#         for name in self.names:
#             setattr(module, name, self.compute_weight(module, name))

#     def __call__(self, module, inputs):
#         # Typically, every time the module is called we need to recompute the weight. However,
#         # in the case of TrellisNet, the same weight is shared across layers, and we can save
#         # a lot of intermediate memory by just recomputing once (at the beginning of first call).
#         pass


# def weight_norm(module, names, dim=0, k=None):
#     fn = WeightNorm.apply(module, names, dim, k)
#     return module, fn


class WeightNorm(object):
    def __init__(self, names, dim, k=None, ste=True):
        self.names = names
        self.dim = dim
        self.k = k
        self.ste = ste  # straight-through estimator for topk mask

    @staticmethod
    def _row_topk_mask_abs(W_flat, k: int):
        """
        W_flat: (rows, cols)
        returns mask of same shape with 1s on top-k |.| entries per row.
        """
        rows, cols = W_flat.shape
        k = int(k)
        if k <= 0:
            return torch.zeros_like(W_flat)
        if k >= cols:
            return torch.ones_like(W_flat)

        idx = W_flat.abs().topk(k, dim=1, largest=True, sorted=False).indices  # (rows, k)
        mask = torch.zeros_like(W_flat)
        mask.scatter_(1, idx, 1.0)
        return mask

    def compute_weight(self, module, name):
        g = getattr(module, name + "_g")
        v = getattr(module, name + "_v")

        # weightnorm
        W = v * (g / _norm(v, self.dim))

        if self.k is None:
            return W
        # print(W.shape)
        # We support dim=0 (common: per-out-channel norm). If dim != 0, we
        # transpose so "rows" correspond to that dim, sparsify, then transpose back.
        permuted = False
        if self.dim is not None and self.dim != 0:
            # move `dim` to front
            dims = list(range(W.dim()))
            dims[0], dims[self.dim] = dims[self.dim], dims[0]
            W = W.permute(*dims).contiguous()
            permuted = True

        # Flatten everything but the first axis: (rows, cols)
        rows = W.shape[0]
        W_flat = W.view(rows, -1)
        # print(W_flat.shape)

        # Build top-k mask per row
        mask_flat = self._row_topk_mask_abs(W_flat, self.k)
        W_sparse_flat = W_flat * mask_flat
        W_sparse = W_sparse_flat.view_as(W)

        # Optional STE: forward uses sparse, backward gradient flows as if dense
        if self.ste:
            W_out = W + (W_sparse - W).detach()
        else:
            W_out = W_sparse

        if permuted:
            # invert permutation
            inv = [0] * W_out.dim()
            for i, d in enumerate(dims):
                inv[d] = i
            W_out = W_out.permute(*inv).contiguous()

        return W_out

    @staticmethod
    def apply(module, names, dim, k=None, ste=True):
        fn = WeightNorm(names, dim, k=k, ste=ste)
        for name in names:
            weight = getattr(module, name)
            del module._parameters[name]
            module.register_parameter(name + "_g", Parameter(_norm(weight, dim).data))
            module.register_parameter(name + "_v", Parameter(weight.data))
            setattr(module, name, fn.compute_weight(module, name))
        module.register_forward_pre_hook(fn)
        return fn

    def reset(self, module):
        for name in self.names:
            setattr(module, name, self.compute_weight(module, name))

    def __call__(self, module, inputs):
        pass


def weight_norm(module, names, dim=0, k=None, ste=True):
    fn = WeightNorm.apply(module, names, dim, k=k, ste=ste)
    return module, fn
