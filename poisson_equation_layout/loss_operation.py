# Reference: https://github.com/zhaoxiaoyu1995/PI-UNet_HSL-TFP/blob/main/layout_data/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F



UNARY_OPS = {
    0: lambda x: torch.abs(x),
    1: lambda x: torch.pow(x, 2),
    2: lambda x: x,
}

class P_OHEM(torch.nn.Module):
    """
    Weighted Loss
    """

    def __init__(self,weight,coef=10):
        super(P_OHEM, self).__init__()
        self.weight=weight
        self.coef=coef
        # self.loss_func=F.l1_loss
    def forward(self, difference,epoch=None, iteration=None):
        # differ = self.loss_fun(input, target, reduction='none').detach()
        differ = difference.detach()
        min, max = torch.min(differ .view(differ .shape[0], -1), dim=1)[0], torch.max(differ .view(differ .shape[0], -1), dim=1)[0]
        if differ .ndim == 4:
            min, max = min.reshape(differ .shape[0], 1, 1, 1).expand(differ.shape), \
                       max.reshape(differ.shape[0], 1, 1, 1).expand(differ.shape)
        elif differ .ndim == 3:
            min, max = min.reshape(differ .shape[0], 1, 1).expand(differ .shape), \
                       max.reshape(differ.shape[0], 1, 1).expand(differ.shape)
        self_weight =self.coef * (differ - min) / (max - min) + self.weight
        # return torch.mean(torch.abs(self_weight * difference))
        return self_weight

class One(torch.nn.Module):
    """
    Original Weight
    """
    def __init__(self,weight):
        super(One, self).__init__()
        self.weight = weight
    def forward(self, difference,epoch=None, iteration=None):
        self_weight=torch.ones_like(difference)+self.weight
        return self_weight


class Max(torch.nn.Module):
    """
    Max residue with per-batch weight tracking
    """
    def __init__(self, weight):
        super(Max, self).__init__()
        global global_weight_dict_1
        self.initial_weight = weight

    def forward(self, difference, epoch, iteration):
        global global_weight_dict_1
        if epoch==0:
            global_weight_dict_1={}

        # 如果当前 batch 的 weight 尚未初始化，则初始化为上一个 iteration 的 weight 或初始 weight
        if iteration not in global_weight_dict_1:
            if len(global_weight_dict_1) == 0:
                # 第一次调用，使用初始 weight
                global_weight_dict_1[iteration] = self.initial_weight.clone()
            else:
                # 使用上一个 iteration 的 weight
                last_iteration = max(global_weight_dict_1.keys())
                global_weight_dict_1[iteration] = global_weight_dict_1[last_iteration].clone()

        # 检查是否满足更新条件
        if (1 < epoch) and (epoch % 2) == 0:
            batch_size = difference.size(0)
            add_weight = torch.zeros_like(difference)
            top_n = 500

            for i in range(batch_size):
                # 找到当前 batch 中绝对值最大的 top_n 个元素位置
                _, top_idxs = torch.topk(torch.abs(difference[i]).view(-1), top_n)
                # 在最大值位置将元素设置为 1
                add_weight[i].view(-1)[top_idxs] = 1

            # 更新当前 batch 的 weight
            global_weight_dict_1[iteration] += add_weight
            return global_weight_dict_1[iteration]
        else:
            return global_weight_dict_1[iteration]

class Loss_Adaptive(torch.nn.Module):
    """
    Adaptive loss with per-batch weight tracking
    """
    def __init__(self, weight):
        super(Loss_Adaptive, self).__init__()
        self.res_w_dict = {}  # 使用字典存储每个 batch 的 res_w
        self.initial_weight = weight

    def forward(self, difference, epoch, iteration):
        # 初始化当前 batch 对应的 res_w
        if iteration not in self.res_w_dict:
            self.res_w_dict[iteration] = torch.ones_like(self.initial_weight, requires_grad=True)
        # 获取当前 batch 对应的 res_w
        res_w = self.res_w_dict[iteration]

        # 更新 res_w 权重（如果满足条件）
        # if epoch >= 50:
        if epoch >= 2:
            loss = torch.mean(res_w * torch.square(difference))
            grads_res = torch.autograd.grad(loss, res_w, create_graph=True)[0]
            # 使用 clone() 更新 res_w，避免 in-place 操作
            self.res_w_dict[iteration] = res_w.clone() + 0.1 * grads_res

        # 返回当前 batch 的 res_w（detached）
        return self.res_w_dict[iteration].detach().clone()

WEIGHT_INIT={
    'one': lambda x: torch.ones_like(x),
    'zero': lambda x: torch.zeros_like(x),
}
