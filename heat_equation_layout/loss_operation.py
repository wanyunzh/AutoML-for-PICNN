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
    def forward(self, difference,epoch=None):
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
    def forward(self, difference,epoch=None):
        self_weight=torch.ones_like(difference)+self.weight
        return self_weight

class Max(torch.nn.Module):
    """
    Max residue
    """
    def __init__(self,weight):
        super(Max, self).__init__()
        # if epoch==0:
        #     self.weight = weight
        # else:
        #     pass
        self.weight=weight
    def forward(self, difference,epoch):
        result=torch.allclose(self.weight, torch.zeros_like(difference))
        if not result:
            if epoch>=2 and (epoch % 2 == 0):
    #     if epoch>1:
        # 单个的话这里可以搜epoch开始数量和top_k
                batch_size = difference.size(0)
                add_weight = torch.zeros_like(difference)
                top_n=3
                for i in range(batch_size):
                    # 对每个batch找到绝对值最大的前三个元素的位置
                    _, top_idxs = torch.topk(torch.abs(difference[i]).view(-1), top_n)
                    # 在最大值位置将元素设置为 1
                    add_weight[i].view(-1)[top_idxs] = 1
                self.weight= add_weight + self.weight
                return self.weight
            else:
                return self.weight
        else:
            batch_size = difference.size(0)
            add_weight = torch.zeros_like(difference)
            top_n = 3
            for i in range(batch_size):
                # 对每个batch找到绝对值最大的前三个元素的位置
                _, top_idxs = torch.topk(torch.abs(difference[i]).view(-1), top_n)
                # 在最大值位置将元素设置为 1
                add_weight[i].view(-1)[top_idxs] = 1
            self.weight = add_weight + self.weight
            return self.weight
        # else:
        #     batch_size = difference.size(0)
        #     add_weight = torch.zeros_like(difference)
        #     top_n = 3
        #     for i in range(batch_size):
        #         # 对每个batch找到绝对值最大的前三个元素的位置
        #         _, top_idxs = torch.topk(torch.abs(difference[i]).view(-1), top_n)
        #         # 在最大值位置将元素设置为 1
        #         add_weight[i].view(-1)[top_idxs] = 1
        #     self.weight = add_weight + self.weight
        #     return self.weight

class Loss_Adaptive(torch.nn.Module):
    """
    Max residue
    """
    def __init__(self, weight):
        super(Loss_Adaptive, self).__init__()
        self.res_w = torch.ones_like(weight, requires_grad=True)

    def forward(self, difference, epoch):
        # if (200 < epoch < 1500) and (epoch % 20) == 0:
        if epoch>=2:
            loss = torch.mean(self.res_w * torch.square(difference))
            grads_res = torch.autograd.grad(loss, self.res_w, create_graph=True)[0]
            self.res_w = self.res_w.clone() + 1e-20 * grads_res

        return self.res_w.detach().clone()



WEIGHT_INIT={
    'one': lambda x: torch.ones_like(x),
    'zero': lambda x: torch.zeros_like(x),
}
