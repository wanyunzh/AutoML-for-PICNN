from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from hpo_utils import *
import torch
import torch.nn as nn
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
from nni.retiarii import model_wrapper
from collections import OrderedDict
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from evaluate_main import traintest_pos
torch.manual_seed(123)

class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__()

        self.stem = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, padding_mode='reflect')),
            ("sepconv3x3", nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1))),
            #
            ("sepconv5x5", nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1))),
            ("conv5x5", nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False, padding_mode='reflect'))
        ]))

        layers1 = [
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
        ]
        self.mid = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, padding_mode='reflect')),
            ("sepconv3x3", nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(out_channels, out_channels, kernel_size=1))),
            #
            ("sepconv5x5", nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 5, 1, 2, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(out_channels, out_channels, kernel_size=1))),
            ("conv5x5", nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False, padding_mode='reflect'))
        ]))

        layers2 = [nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
                   nn.GELU()]
        # if dropout:
        #     layers1.append(nn.Dropout())
        self.encode = nn.Sequential(*layers1)
        self.out = nn.Sequential(*layers2)
        self.pool = None
        if polling:
            self.pool = LayerChoice(OrderedDict([
                ("maxpool", nn.MaxPool2d(2, 2)),
                ("avgpool", nn.AvgPool2d(2, 2)),
                # ("sepconv3x3", nn.Sequential(
                #     nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels, padding_mode='reflect'),
                #     nn.Conv2d(in_channels, in_channels, kernel_size=1))),
                # ("conv3x3", nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False, padding_mode='reflect')),
                # ("sepconv5x5", nn.Sequential(
                #     nn.Conv2d(in_channels, in_channels, 5, 2, 2, groups=in_channels, padding_mode='reflect'),
                #     nn.Conv2d(in_channels, in_channels, kernel_size=1))),
                # ("conv5x5", nn.Conv2d(in_channels, in_channels, 5, 2, 2, bias=False, padding_mode='reflect'))
            ]))

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = self.stem(x)
        x = self.encode(x)
        x = self.mid(x)
        x = self.out(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(_DecoderBlock, self).__init__()

        self.decode1 = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(in_channels, middle_channels, 3, 1, 1, bias=False, padding_mode='reflect')),
            ("sepconv3x3", nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(in_channels, middle_channels, kernel_size=1))),
            #
            ("sepconv5x5", nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(in_channels, middle_channels, kernel_size=1))),
            ("conv5x5", nn.Conv2d(in_channels, middle_channels, 5, 1, 2, bias=False, padding_mode='reflect'))
        ]))

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.GELU(),
        )
        self.decode2 = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(middle_channels, out_channels, 3, 1, 1, bias=False, padding_mode='reflect')),
            ("sepconv3x3", nn.Sequential(
                nn.Conv2d(middle_channels, middle_channels, 3, 1, 1, groups=middle_channels, padding_mode='reflect'),
                nn.Conv2d(middle_channels, out_channels, kernel_size=1))),

            ("sepconv5x5", nn.Sequential(
                nn.Conv2d(middle_channels, middle_channels, 5, 1, 2, groups=middle_channels, padding_mode='reflect'),
                nn.Conv2d(middle_channels, out_channels, kernel_size=1))),
            ("conv5x5", nn.Conv2d(middle_channels, out_channels, 5, 1, 2, bias=False, padding_mode='reflect'))
        ]))
        layers2 = [nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
                   nn.GELU()]

        self.out = nn.Sequential(*layers2)

    def forward(self, x):
        x = self.decode1(x)
        x = self.layer1(x)
        x = self.decode2(x)
        x = self.out(x)
        return x


# def initialize_weights(*models):
#     for model in models:
#         for module in model.modules():
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 nn.init.kaiming_normal_(module.weight)
#                 if module.bias is not None:
#                     module.bias.data.zero_()
#             elif isinstance(module, nn.BatchNorm2d):
#                 module.weight.data.fill_(1)
#                 module.bias.data.zero_()


class UpsamplingNearest2d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class UpsamplingBilinear2d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode='bilinear', align_corners=True)


class upStdConv(nn.Module):
    """
    Standard conv:  Conv-GeLU-GN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        group = 32 if C_out % 32 == 0 else 16
        self.net = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, kernel_size,
                               stride, padding, output_padding=1),
            nn.GroupNorm(group, C_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


@model_wrapper
class UNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, bn=False, factors=2):
        super().__init__()
        self.enc1 = _EncoderBlock(in_channels, 32 * factors, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32 * factors, 64 * factors, bn=bn)
        self.enc3 = _EncoderBlock(64 * factors, 128 * factors, bn=bn)
        self.polling = LayerChoice(OrderedDict([
                ("maxpool", nn.MaxPool2d(2, 2)),
                ("avgpool", nn.AvgPool2d(2, 2))]))
        self.center = _DecoderBlock(128 * factors, 256 * factors, 128 * factors, bn=bn)
        self.dec3 = _DecoderBlock(256 * factors, 128 * factors, 64 * factors, bn=bn)
        self.dec2 = _DecoderBlock(128 * factors, 64 * factors, 32 * factors, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 * factors, 32 * factors, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
            nn.Conv2d(32 * factors, 32 * factors, kernel_size=1, padding=0),
            nn.BatchNorm2d(32 * factors) if bn else nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
        )
        self.final = nn.Conv2d(32 * factors, num_classes, kernel_size=1)
        self.up1 = LayerChoice(OrderedDict([
            ("biupsample", UpsamplingBilinear2d(scale_factor=2)),
            ("nearupsample", UpsamplingNearest2d(scale_factor=2)),
            # ("upconv3x3", upStdConv(256 * factors, 256 * factors, 3, 2, 1))
        ]))

        # # self.up1=upDilConv(256 * factors, 256 * factors, 3, 2,2,2)
        self.up2 = LayerChoice(OrderedDict([
            ("biupsample", UpsamplingBilinear2d(scale_factor=2)),
            ("nearupsample", UpsamplingNearest2d(scale_factor=2)),
            # ("upconv3x3", upStdConv(128 * factors, 128 * factors, 3, 2, 1))
        ]))
        self.up3 = LayerChoice(OrderedDict([
            ("biupsample", UpsamplingBilinear2d(scale_factor=2)),
            ("nearupsample", UpsamplingNearest2d(scale_factor=2)),
            # ("upconv3x3", upStdConv(64 * factors, 64 * factors, 3, 2, 1))
        ]))


    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(self.polling(enc3))
        mid3 = self.up1(center)

        if mid3.size() != enc3.size():
            _, _, height1, width1 = enc3.size()
            mid3 = F.interpolate(mid3, (height1, width1), mode='bilinear')

        dec3 = self.dec3(torch.cat([mid3, enc3], 1))

        mid2 = self.up2(dec3)
        if mid2.size() != enc2.size():
            _, _, height1, width1 = enc2.size()
            mid2 = F.interpolate(mid2, (height1, width1), mode='bilinear')

        dec2 = self.dec2(torch.cat([mid2, enc2], 1))

        mid1 = self.up3(dec2)
        if mid1.size() != enc1.size():
            _, _, height1, width1 = enc1.size()
            mid1 = F.interpolate(mid1, (height1, width1), mode='bilinear')

        dec1 = self.dec1(torch.cat([mid1, enc1], 1))
        final = self.final(dec1)
        return final


if __name__ == "__main__":
    model_space = UNet(num_classes=1, in_channels=1)
    evaluator = FunctionalEvaluator(traintest_pos)
    exp = RetiariiExperiment(model_space, evaluator, [], strategy.PolicyBasedRL(max_collect=70, trial_per_collect=1))
    # exp = RetiariiExperiment(model_space, evaluator, [], strategy.TPEStrategy())
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'possion equation'

    exp_config.trial_concurrency = 1  # 最多同时运行 2 个试验
    exp_config.max_trial_number = 70
    # exp_config.trial_gpu_number = 1
    # exp_config.training_service.use_active_gpu = True
    exp.run(exp_config, 8075)
    for model_dict in exp.export_top_models(top_k=5, formatter='dict'):
        print(model_dict)
    exported_arch_best = exp.export_top_models(top_k=1, formatter='dict')[0]
    import json
    from nni.retiarii import fixed_arch
    json.dump(exported_arch_best, open('possion_modelnew.json', 'w'))
    with fixed_arch('possion_modelnew.json'):
        final_model = UNet(num_classes=1, in_channels=1)
        print('final model:', final_model)



# if __name__ == '__main__':
#     model = UNet(in_channels=1, num_classes=1)
#     print(model)
#     x = torch.randn(32, 1, 29, 29)
#     with torch.no_grad():
#         final = model(x)
#         print(final.shape)



