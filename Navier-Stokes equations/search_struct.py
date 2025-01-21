# Reference: https://github.com/Jianxun-Wang/phygeonet/tree/master 
import sys
import numpy as np
import torch.nn as nn
sys.path.insert(0, '../source')
from evaluate_NS import traintest_NS
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
import torch
import torch.nn.init as init
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
torch.manual_seed(123)
import random
# device = torch.device(f"cuda:{7}" if torch.cuda.is_available() else "cpu")
# print(device)


class DepthwiseSeparableConv3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3,padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        # self._initialize_weights()

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    # def _initialize_weights(self):
    #     # 这里可以根据需要应用不同的初始化方法
    #     init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
    #     init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')

class DepthwiseSeparableConv5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=5,padding=2, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        # self._initialize_weights()

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    # def _initialize_weights(self):
    #     # 这里可以根据需要应用不同的初始化方法
    #     init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
    #     init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')

class DepthwiseSeparableConv7(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=7,padding=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        # self._initialize_weights()

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    # def _initialize_weights(self):
    #     # 这里可以根据需要应用不同的初始化方法
    #     init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
    #     init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')

@model_wrapper
class HBCNN_5(nn.Module):
    def __init__(self, params,h,nx,ny,In=1,Out=1,initWay='kaiming'):
        super().__init__()
        """
        Extract basic information
        """
        self.initWay=initWay
        self.nx = nx
        self.ny = ny
        self.In = In
        self.Out = Out
        self.relu = nn.ReLU()
        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
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

        self.conv3 = nn.LayerChoice([
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel2'], params['channel3']),
            DepthwiseSeparableConv3(params['channel2'], params['channel3']),
            DepthwiseSeparableConv7(params['channel2'], params['channel3'])])

        self.conv4 = nn.LayerChoice([
            nn.Conv2d(params['channel3'], self.Out, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel3'], self.Out, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel3'], self.Out, kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel3'], self.Out),
            DepthwiseSeparableConv3(params['channel3'], self.Out),
            DepthwiseSeparableConv7(params['channel3'], self.Out)])

        self.pixel_shuffle1 = nn.PixelShuffle(1)
        self.conv11 = nn.LayerChoice([
            nn.Conv2d(self.In, params['channel1'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(self.In, params['channel1'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.In, params['channel1'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(In, params['channel1']),
            DepthwiseSeparableConv3(In, params['channel1']),
            DepthwiseSeparableConv7(In, params['channel1'])])

        self.conv22 = nn.LayerChoice([
            nn.Conv2d(params['channel1'], params['channel2'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel1'], params['channel2'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel1'], params['channel2'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel1'], params['channel2']),
            DepthwiseSeparableConv3(params['channel1'], params['channel2']),
            DepthwiseSeparableConv7(params['channel1'], params['channel2'])])

        self.conv33 = nn.LayerChoice([
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel2'], params['channel3']),
            DepthwiseSeparableConv3(params['channel2'], params['channel3']),
            DepthwiseSeparableConv7(params['channel2'], params['channel3'])])

        self.conv44 = nn.LayerChoice([
            nn.Conv2d(params['channel3'], self.Out, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel3'], self.Out, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel3'], self.Out, kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel3'], self.Out),
            DepthwiseSeparableConv3(params['channel3'], self.Out),
            DepthwiseSeparableConv7(params['channel3'], self.Out)])
        self.pixel_shuffle11 = nn.PixelShuffle(1)
        self.conv111 = nn.LayerChoice([
            nn.Conv2d(self.In, params['channel1'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(self.In, params['channel1'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.In, params['channel1'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(In, params['channel1']),
            DepthwiseSeparableConv3(In, params['channel1']),
            DepthwiseSeparableConv7(In, params['channel1'])])

        self.conv222 = nn.LayerChoice([
            nn.Conv2d(params['channel1'], params['channel2'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel1'], params['channel2'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel1'], params['channel2'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel1'], params['channel2']),
            DepthwiseSeparableConv3(params['channel1'], params['channel2']),
            DepthwiseSeparableConv7(params['channel1'], params['channel2'])])

        self.conv333 = nn.LayerChoice([
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel2'], params['channel3'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel2'], params['channel3']),
            DepthwiseSeparableConv3(params['channel2'], params['channel3']),
            DepthwiseSeparableConv7(params['channel2'], params['channel3'])])

        self.conv444 = nn.LayerChoice([
            nn.Conv2d(params['channel3'], self.Out, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel3'], self.Out, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel3'], self.Out, kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel3'], self.Out),
            DepthwiseSeparableConv3(params['channel3'], self.Out),
            DepthwiseSeparableConv7(params['channel3'], self.Out)])
        self.pixel_shuffle111 = nn.PixelShuffle(1)
        if self.initWay is not None:
            self._initialize_weights()

    def forward(self, x):
        x = self.US(x)
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv3(x1))
        x1 = self.pixel_shuffle1(self.conv4(x1))

        x2 = self.relu(self.conv11(x))
        x2 = self.relu(self.conv22(x2))
        x2 = self.relu(self.conv33(x2))
        x2 = self.pixel_shuffle11(self.conv44(x2))

        x3 = self.relu(self.conv111(x))
        x3 = self.relu(self.conv222(x3))
        x3 = self.relu(self.conv333(x3))
        x3 = self.pixel_shuffle111(self.conv444(x3))
        return torch.cat([x1, x2, x3], axis=1)

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'conv4' in name:
                    init.kaiming_normal_(module.weight)
                # 对其他的 nn.Conv2d 层使用自定义的参数
                else:
                    init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (DepthwiseSeparableConv5, DepthwiseSeparableConv7)):

                if 'conv4' in name:
                    init.kaiming_normal_(module.depthwise.weight)
                    init.kaiming_normal_(module.pointwise.weight)
                else:
                # 对于自定义层，这里假设你想要使用与其他层相同的自定义参数
                    init.kaiming_normal_(module.depthwise.weight, mode='fan_out', nonlinearity='relu')
                    init.kaiming_normal_(module.pointwise.weight, mode='fan_out', nonlinearity='relu')

if __name__ == "__main__":
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params1 = {
        'channel1': 16,
        'channel2': 32,
        'channel3': 16
    }
    NvarInput = 2
    NvarOutput =1
    nxOF = 50
    nyOF = 50
    nx = nxOF + 2
    ny = nyOF + 2
    h=0.01
    # model_space=HBCNN_5(params1,NvarInput,NvarOutput)
    model_space=HBCNN_5(params1,h,nx, ny, NvarInput, NvarOutput,'kaiming')
    evaluator = FunctionalEvaluator(traintest_NS)

    exp = RetiariiExperiment(model_space, evaluator, [], strategy.PolicyBasedRL(max_collect=200,trial_per_collect=3))
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'NS'

    exp_config.trial_concurrency = 1  # 最多同时运行 2 个试验
    # exp_config.max_trial_number = 200
    # exp_config.trial_gpu_number = 3
    # exp_config.training_service.use_active_gpu = False
    exp.run(exp_config, 8065)