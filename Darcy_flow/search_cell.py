import sys
import numpy as np
import time

from sklearn.metrics import mean_squared_error as calMSE
import nni
import pickle
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
import torch
import nni.retiarii.nn.pytorch as nn
from collections import OrderedDict
from torch.nn.functional import interpolate
import ops
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from nni.retiarii import model_wrapper
from evaluate_main import unet_struct
import random



class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False,bn=False):
        super(EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(16, out_channels),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        x=self.encode(x)
        return x

class Noded(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            if stride==2:
                self.ops.append(
                    LayerChoice(OrderedDict([
                        ("maxpool", ops.Pool2d('max', 2,2)),
                        ("avgpool", ops.Pool2d('avg', 2,2)),
                    ]), label=choice_keys[-1]))
            else:
                self.ops.append(
                    LayerChoice(OrderedDict([
                        ("sepconv3x3", ops.SepConv(channels, channels, 3, stride, 1)),
                        ("conv3x3", ops.StdConv(channels, channels, 3, stride, 1)),
                        ("sepconv5x5", ops.SepConv(channels, channels, 5, stride, 2)),
                        ("conv5x5", ops.StdConv(channels, channels, 5, stride, 2)),
                    ]), label=choice_keys[-1]))

        # self.drop_path = ops.DropPath()
        self.input_switch = InputChoice(n_candidates=len(choice_keys), n_chosen=2, label="{}_switch".format(node_id))

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        # out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out)

class Nodeu(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            if stride==2:
                self.ops.append(
                    LayerChoice(OrderedDict([
                        ("biupsample", ops.UpsamplingBilinear2d(scale_factor=2)),
                        ("nearupsample", ops.UpsamplingNearest2d(scale_factor=2)),
                    ]), label=choice_keys[-1]))
            else:
                self.ops.append(
                    LayerChoice(OrderedDict([
                        ("sepconv3x3", ops.SepConv(channels, channels, 3, stride, 1)),
                        ("conv3x3", ops.StdConv(channels, channels, 3, stride, 1)),
                        ("sepconv5x5", ops.SepConv(channels, channels, 5, stride, 2)),
                        ("conv5x5", ops.StdConv(channels, channels, 5, stride, 2)),
                    ]), label=choice_keys[-1]))
        # self.drop_path = ops.DropPath()
        self.input_switch = InputChoice(n_candidates=len(choice_keys), n_chosen=2, label="{}_switch".format(node_id))

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        if out[1].size() !=out[0].size():
            _, _, height1, width1 = out[0].size()
            out[1] = interpolate(out[1], (height1, width1))
        # out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out)

class Cellup(nn.Module):

    def __init__(self, n_nodes, channels_pp, channels_p, channels,reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        self.preproc0 = ops.StdConv(channels_pp, channels, 1, 2, 0)
        self.preproc1 = ops.StdConv(channels_p, channels, 1, 1, 0)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Nodeu("{}_n{}".format("up", depth),
                                         depth, channels, 2 if reduction else 0))

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        # if s0.size() != s1.size():
        #     _, _, height1, width1 = s0.size()
        #     s1 = interpolate(s1, (height1, width1))

        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output

class Celldown(nn.Module):
    def __init__(self, n_nodes, channels_pp, channels_p, channels,reduction_p,reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        if reduction_p:
            self.preproc0 = ops.StdConv(channels_pp, channels, 1, 2, 0)
        else:
            self.preproc0 = ops.StdConv(channels_pp, channels, 1, 1, 0)
        self.preproc1 = ops.StdConv(channels_p, channels, 1, 1, 0)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Noded("{}_n{}".format("down", depth),
                                         depth, channels, 2 if reduction else 0))

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output

@model_wrapper
class CNN(nn.Module):

    def __init__(self, in_channels=1, channels=16, n_classes=3, n_layers=4, n_nodes=3,
                 stem_multiplier=4):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        c_cur = stem_multiplier * self.channels
        self.stem=EncoderBlock(in_channels, c_cur)
        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels
        down_cs_nfilters = []
        # create the encoder pathway and add to a list
        down_cs_nfilters += [channels_pp]
        down_cs_nfilters += [channels_p]
        self.down_cells = nn.ModuleList()
        self.up_cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, True
            c_cur *= 2
            down_cell = Celldown(n_nodes, channels_pp, channels_p, c_cur,reduction_p,reduction)
            self.down_cells.append(down_cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out
            down_cs_nfilters += [channels_p]
        for i in range(n_layers):
            channels_pp = down_cs_nfilters[-(i + 2)]  # the horizontal prev_prev input channel
            c_cur =int(c_cur/ 2)
            up_cell = Cellup(n_nodes, channels_pp, channels_p, c_cur, reduction)
            self.up_cells.append(up_cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out
        self.final = nn.Conv2d(channels_p, n_classes, kernel_size=1)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        down_cs = []
        # encoder pathway
        down_cs.append(s0)
        down_cs.append(s1)

        for i, cell in enumerate(self.down_cells):
            s0, s1 = s1, cell(s0, s1)
            down_cs.append(s1)
        for i, cell in enumerate(self.up_cells):
            # Sharing a global N*M weights matrix
            # where M : normal + up
            s0 = down_cs[-(i+2)] # horizon input
            s1 = cell(s0, s1)
        out = self.final(s1)
        return out
    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, ops.DropPath):
                module.p = p


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_space = CNN(n_layers=4)
    evaluator = FunctionalEvaluator(unet_struct)
    exp = RetiariiExperiment(model_space, evaluator, [], strategy.PolicyBasedRL(max_collect=20,trial_per_collect =1))
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'darcy flow cell'
    exp_config.trial_concurrency = 1 # 最多同时运行 2 个试验
    exp_config.max_trial_number = 20
    exp.run(exp_config, 8001)
    for model_dict in exp.export_top_models(top_k=1, formatter='dict'):
        print(model_dict)
    from nni.retiarii import fixed_arch
    exported_arch_best = exp.export_top_models(top_k=1, formatter='dict')[0]
    import json
    json.dump(exported_arch_best, open('darcy_cell.json', 'w'))
    with fixed_arch('darcy_cell.json'):
        final_model = CNN(n_layers=4)
        print('final model:', final_model)


# if __name__ == '__main__':
#     model = CNN(n_layers=4)
#     # print(model)
#     x = torch.randn(32, 1, 64, 64)
#     with torch.no_grad():
#         final = model(x)
#         print(final.shape)