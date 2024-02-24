from evaluate_main import traintest_hb
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import torch
import torch.nn as nn
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
from nni.retiarii import model_wrapper
import numpy as np
import random
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class DepthwiseSeparableConv3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3,padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DepthwiseSeparableConv5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=5,padding=2, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DepthwiseSeparableConv7(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=7,padding=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

@model_wrapper
class HBCNN_5(nn.Module):
    def __init__(self, params, In, Out):
        super().__init__()
        self.In = In
        self.Out = Out
        """
        Define net
        """
        self.relu = nn.ReLU()
        self.conv1 = nn.LayerChoice([
            nn.Conv2d(self.In, params['channel1'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(self.In, params['channel1'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.In, params['channel1'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(In, params['channel1']),
            DepthwiseSeparableConv3(In, params['channel1']),
            DepthwiseSeparableConv7(In, params['channel1'])])

        self.conv2 = nn.LayerChoice([
            nn.Conv2d(params['channel1'], params['channel2'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel1'], params['channel2'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel1'], params['channel2'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel1'], params['channel2']),
            DepthwiseSeparableConv3(params['channel1'], params['channel2']),
            DepthwiseSeparableConv7(params['channel1'], params['channel2'])])

        #
        self.conv3 = nn.LayerChoice([
            nn.Conv2d(params['channel2'], params['channel2'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel2'], params['channel2'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel2'], params['channel2'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel2'], params['channel2']),
            DepthwiseSeparableConv3(params['channel2'], params['channel2']),
            DepthwiseSeparableConv7(params['channel2'], params['channel2']),
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            nn.MaxPool2d(3, stride=1, padding=1)])

        self.conv4 = nn.LayerChoice([
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel2'], params['channel3']),
            DepthwiseSeparableConv3(params['channel2'], params['channel3']),
            DepthwiseSeparableConv7(params['channel2'], params['channel3'])])

        self.conv5 = nn.LayerChoice([
            nn.Conv2d(params['channel3'], self.Out, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel3'], self.Out, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel3'], self.Out, kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel3'], self.Out),
            DepthwiseSeparableConv3(params['channel3'], self.Out),
            DepthwiseSeparableConv7(params['channel3'], self.Out)])
        self.pixel_shuffle = nn.PixelShuffle(1)


    def forward(self, x):
        # x = self.US(x)  # 19*84变为17*82
        x = self.relu(self.conv1(x))  # channel由2变成了16,大小不变，kernel size=5， padding=2
        x = self.relu(self.conv2(x))# channel由16变成了32，大小不变，kernel size=5， padding=2
        x=self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))  # channel由32变成了16，大小不变，kernel size=5， padding=2
        x = self.conv5(x)  # channel由16变成了1，大小不变，kernel size=5， padding=2
        x = self.pixel_shuffle(x)  # 不变还是原来的维度
        return x



if __name__ == "__main__":
    params1 = {
        'channel1': 16,
        'channel2': 32,
        'channel3': 16
    }

    In = 1
    Out = 1

    model_space = HBCNN_5(params1,In, Out)

    evaluator = FunctionalEvaluator(traintest_hb)

    exp = RetiariiExperiment(model_space, evaluator, [], strategy.PolicyBasedRL(max_collect=300, trial_per_collect=1))

    # exp = RetiariiExperiment(model_space, evaluator, [], strategy.TPEStrategy())
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'heat equation boundary'

    exp_config.trial_concurrency = 1  # 最多同时运行 2 个试验
    exp_config.max_trial_number = 300
    exp.run(exp_config, 8065)
    for model_dict in exp.export_top_models(top_k=5, formatter='dict'):
        print(model_dict)
    exported_arch_best = exp.export_top_models(top_k=1, formatter='dict')[0]
    import json
    from nni.retiarii import fixed_arch
    json.dump(exported_arch_best, open('HB_cnnnew.json', 'w'))
    with fixed_arch('HB_cnnnew.json.json'):
        final_model = HBCNN_5(params1,In, Out)
        print('final model:', final_model)





