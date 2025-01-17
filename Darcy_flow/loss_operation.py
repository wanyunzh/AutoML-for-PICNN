import torch
import torch.nn as nn
import torch.nn.functional as F



UNARY_OPS = {
    'absolute': lambda x: torch.abs(x),
    'square': lambda x: torch.pow(x, 2),
    'identity': lambda x: x,
}


class P_OHEM1(torch.nn.Module):
    """
    Weighted Loss
    """

    def __init__(self,weight,coef=10):
        super(P_OHEM1, self).__init__()
        self.weight=weight
        self.coef=coef
    def forward(self, difference,epoch=None):
        differ = difference.detach()
        min, max = torch.min(differ .view(differ .shape[0], -1), dim=1)[0], torch.max(differ .view(differ .shape[0], -1), dim=1)[0]
        if differ .ndim == 4:
            min, max = min.reshape(differ .shape[0], 1, 1, 1).expand(differ.shape), \
                       max.reshape(differ.shape[0], 1, 1, 1).expand(differ.shape)
        elif differ .ndim == 3:
            min, max = min.reshape(differ .shape[0], 1, 1).expand(differ .shape), \
                       max.reshape(differ.shape[0], 1, 1).expand(differ.shape)
        self_weight =self.coef * (differ - min) / (max - min) + self.weight
        return self_weight

class P_OHEM2(torch.nn.Module):
    """
    Weighted Loss
    """

    def __init__(self,weight,coef=10):
        super(P_OHEM2, self).__init__()
        self.weight=weight
        self.coef=coef
    def forward(self, difference,epoch=None):
        differ = difference.detach()
        min, max = torch.min(differ .view(differ .shape[0], -1), dim=1)[0], torch.max(differ .view(differ .shape[0], -1), dim=1)[0]
        if differ .ndim == 4:
            min, max = min.reshape(differ .shape[0], 1, 1, 1).expand(differ.shape), \
                       max.reshape(differ.shape[0], 1, 1, 1).expand(differ.shape)
        elif differ .ndim == 3:
            min, max = min.reshape(differ .shape[0], 1, 1).expand(differ .shape), \
                       max.reshape(differ.shape[0], 1, 1).expand(differ.shape)
        self_weight =self.coef * (differ - min) / (max - min) + self.weight
        return self_weight

class P_OHEM3(torch.nn.Module):
    """
    Weighted Loss
    """

    def __init__(self,weight,coef=10):
        super(P_OHEM3, self).__init__()
        self.weight=weight
        self.coef=coef
    def forward(self, difference,epoch=None):
        differ = difference.detach()
        min, max = torch.min(differ .view(differ .shape[0], -1), dim=1)[0], torch.max(differ .view(differ .shape[0], -1), dim=1)[0]
        if differ .ndim == 4:
            min, max = min.reshape(differ .shape[0], 1, 1, 1).expand(differ.shape), \
                       max.reshape(differ.shape[0], 1, 1, 1).expand(differ.shape)
        elif differ .ndim == 3:
            min, max = min.reshape(differ .shape[0], 1, 1).expand(differ .shape), \
                       max.reshape(differ.shape[0], 1, 1).expand(differ.shape)
        self_weight =self.coef * (differ - min) / (max - min) + self.weight
        return self_weight

class One(torch.nn.Module):
    """
    Original Weight
    """
    def __init__(self,weight):
        super(One, self).__init__()
        self.weight = weight
    def forward(self, difference,epoch=None):
        self_weight=torch.ones_like(difference)+self.weight
        return self_weight

class Max1(torch.nn.Module):
    """
    Max residue with per-batch weight tracking
    """
    def __init__(self, weight):
        super(Max1, self).__init__()
        self.weight_dict = {}  # 使用字典存储每个 batch 的 weight
        self.initial_weight = weight

    def forward(self, difference, epoch, iteration,weight):
        # 如果当前 batch 的 weight 尚未初始化，则初始化为初始 weight
        if iteration not in self.weight_dict:
            # self.weight_dict[iteration] = torch.zeros_like(difference)
            self.weight_dict[iteration] = self.initial_weight

        # 检查是否满足更新条件
        if  (epoch % 1) == 0:
            batch_size = difference.size(0)
            add_weight = torch.zeros_like(difference)
            top_n = 2000

            for i in range(batch_size):
                # 找到当前 batch 中绝对值最大的 top_n 个元素位置
                _, top_idxs = torch.topk(torch.abs(difference[i]).view(-1), top_n)
                # 在最大值位置将元素设置为 1
                add_weight[i].view(-1)[top_idxs] = 1

            # 更新当前 batch 的 weight
            self.weight_dict[iteration] += add_weight
            return self.weight_dict[iteration]
        else:
            return self.weight_dict[iteration]

class Max2(torch.nn.Module):
    """
    Max residue with per-batch weight tracking
    """
    def __init__(self, weight):
        super(Max2, self).__init__()
        self.weight_dict = {}  # 使用字典存储每个 batch 的 weight
        self.initial_weight = weight

    def forward(self, difference, epoch, iteration,weight):
        # 如果当前 batch 的 weight 尚未初始化，则初始化为初始 weight
        if iteration not in self.weight_dict:
            # self.weight_dict[iteration] = torch.zeros_like(difference)
            self.weight_dict[iteration] = self.initial_weight

        # 检查是否满足更新条件
        if  (epoch % 1) == 0:
            batch_size = difference.size(0)
            add_weight = torch.zeros_like(difference)
            top_n = 2000

            for i in range(batch_size):
                # 找到当前 batch 中绝对值最大的 top_n 个元素位置
                _, top_idxs = torch.topk(torch.abs(difference[i]).view(-1), top_n)
                # 在最大值位置将元素设置为 1
                add_weight[i].view(-1)[top_idxs] = 1

            # 更新当前 batch 的 weight
            self.weight_dict[iteration] += add_weight
            return self.weight_dict[iteration]
        else:
            return self.weight_dict[iteration]


class Max3(torch.nn.Module):
    """
    Max residue with per-batch weight tracking
    """
    def __init__(self, weight):
        super(Max3, self).__init__()
        self.weight_dict = {}  # 使用字典存储每个 batch 的 weight
        self.initial_weight = weight

    def forward(self, difference, epoch, iteration,weight):
        # 如果当前 batch 的 weight 尚未初始化，则初始化为初始 weight
        if iteration not in self.weight_dict:
            # self.weight_dict[iteration] = torch.zeros_like(difference)
            self.weight_dict[iteration] = self.initial_weight

        # 检查是否满足更新条件
        if  (epoch % 1) == 0:
            batch_size = difference.size(0)
            add_weight = torch.zeros_like(difference)
            top_n = 2000

            for i in range(batch_size):
                # 找到当前 batch 中绝对值最大的 top_n 个元素位置
                _, top_idxs = torch.topk(torch.abs(difference[i]).view(-1), top_n)
                # 在最大值位置将元素设置为 1
                add_weight[i].view(-1)[top_idxs] = 1

            # 更新当前 batch 的 weight
            self.weight_dict[iteration] += add_weight
            return self.weight_dict[iteration]
        else:
            return self.weight_dict[iteration]

class Loss_Adaptive1(torch.nn.Module):
    """
    Max residue
    """
    def __init__(self, weight):
        super(Loss_Adaptive1, self).__init__()
        self.res_w = torch.ones_like(weight, requires_grad=True)

    def forward(self, difference, epoch):
        # if (200 < epoch < 1500) and (epoch % 20) == 0:
        if epoch>=0:
            loss = torch.mean(self.res_w * torch.square(difference))
            grads_res = torch.autograd.grad(loss, self.res_w, create_graph=True)[0]
            self.res_w = self.res_w.clone() + 1e-4 * grads_res

        return self.res_w.detach().clone()

class Loss_Adaptive2(torch.nn.Module):
    """
    Max residue
    """
    def __init__(self, weight):
        super(Loss_Adaptive2, self).__init__()
        self.res_w = torch.ones_like(weight, requires_grad=True)

    def forward(self, difference, epoch):
        # if (200 < epoch < 1500) and (epoch % 20) == 0:
        if epoch>=50:
            loss = torch.mean(self.res_w * torch.square(difference))
            grads_res = torch.autograd.grad(loss, self.res_w, create_graph=True)[0]
            self.res_w = self.res_w.clone() + 1e-3 * grads_res

        return self.res_w.detach().clone()

class Loss_Adaptive3(torch.nn.Module):
    """
    Max residue
    """
    def __init__(self, weight):
        super(Loss_Adaptive3, self).__init__()
        self.res_w = torch.ones_like(weight, requires_grad=True)

    def forward(self, difference, epoch):
        # if (200 < epoch < 1500) and (epoch % 20) == 0:
        if epoch>=50:
            loss = torch.mean(self.res_w * torch.square(difference))
            grads_res = torch.autograd.grad(loss, self.res_w, create_graph=True)[0]
            self.res_w = self.res_w.clone() + 5*1e-2 * grads_res

        return self.res_w.detach().clone()


WEIGHT_INIT={
    'one': lambda x: torch.ones_like(x),
    'zero': lambda x: torch.zeros_like(x),
}
