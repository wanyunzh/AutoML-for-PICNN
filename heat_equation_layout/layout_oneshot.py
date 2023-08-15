import sys
import numpy as np
import torch
import nni.retiarii.nn.pytorch as nn
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from nni.retiarii import model_wrapper
import os
from torch.backends import cudnn
import torch.nn.functional as F
import random
from trainer_enas import My_EnasTrainer,My_DartsTrainer




class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__()

        self.stem = LayerChoice([
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, padding_mode='reflect'),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)),
                 nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)),
                nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False, padding_mode='reflect')
        ])

        layers1 = [
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
        ]
        self.mid = LayerChoice([
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, padding_mode='reflect'),
           nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)),
           nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 5, 1, 2, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)),
        nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False, padding_mode='reflect')
        ])

        layers2 = [nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
                   nn.GELU()]
        # if dropout:
        #     layers1.append(nn.Dropout())
        self.encode = nn.Sequential(*layers1)
        self.out = nn.Sequential(*layers2)
        self.pool = None
        if polling:
            self.pool = LayerChoice([
                nn.MaxPool2d(2, 2),
                nn.AvgPool2d(2, 2),
                # ("sepconv3x3", nn.Sequential(
                #     nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels, padding_mode='reflect'),
                #     nn.Conv2d(in_channels, in_channels, kernel_size=1))),
                # ("conv3x3", nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False, padding_mode='reflect')),
                # ("sepconv5x5", nn.Sequential(
                #     nn.Conv2d(in_channels, in_channels, 5, 2, 2, groups=in_channels, padding_mode='reflect'),
                #     nn.Conv2d(in_channels, in_channels, kernel_size=1))),
                # ("conv5x5", nn.Conv2d(in_channels, in_channels, 5, 2, 2, bias=False, padding_mode='reflect'))
            ])

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x=self.stem(x)
        x = self.encode(x)
        x = self.mid(x)
        x = self.out(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(_DecoderBlock, self).__init__()

        self.decode1 = LayerChoice([
            nn.Conv2d(in_channels, middle_channels, 3, 1, 1, bias=False, padding_mode='reflect'),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(in_channels, middle_channels, kernel_size=1)),
           nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups=in_channels, padding_mode='reflect'),
                nn.Conv2d(in_channels, middle_channels, kernel_size=1)),
            nn.Conv2d(in_channels, middle_channels, 5, 1, 2, bias=False, padding_mode='reflect')
        ])

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.GELU(),
        )
        self.decode2 = LayerChoice([
            nn.Conv2d(middle_channels, out_channels, 3, 1, 1, bias=False, padding_mode='reflect'),
            nn.Sequential(
                nn.Conv2d(middle_channels, middle_channels, 3, 1, 1, groups=middle_channels, padding_mode='reflect'),
                nn.Conv2d(middle_channels, out_channels, kernel_size=1)),

            nn.Sequential(
                nn.Conv2d(middle_channels, middle_channels, 5, 1, 2, groups=middle_channels, padding_mode='reflect'),
                nn.Conv2d(middle_channels, out_channels, kernel_size=1)),
            nn.Conv2d(middle_channels, out_channels, 5, 1, 2, bias=False, padding_mode='reflect')
        ])
        layers2 = [nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
                   nn.GELU()]

        self.out = nn.Sequential(*layers2)

    def forward(self, x):
        x = self.decode1(x)
        x=self.layer1(x)

        x = self.decode2(x)

        x = self.out(x)
        return x

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
    def __init__(self, num_classes=3, in_channels=1, bn=False, factors=2):
        super().__init__()
        self.enc1 = _EncoderBlock(in_channels, 32 * factors, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32 * factors, 64 * factors, bn=bn)
        self.enc3 = _EncoderBlock(64 * factors, 128 * factors, bn=bn)
        self.enc4 = _EncoderBlock(128 * factors, 256 * factors, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(256 * factors, 512 * factors, 256 * factors, bn=bn)
        self.dec4 = _DecoderBlock(512 * factors, 256 * factors, 128 * factors, bn=bn)
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
        self.up1 = LayerChoice([ UpsamplingBilinear2d(scale_factor=2),
                                UpsamplingNearest2d(scale_factor=2),
                                            ])

        # # self.up1=upDilConv(256 * factors, 256 * factors, 3, 2,2,2)
        self.up2 = LayerChoice([ UpsamplingBilinear2d(scale_factor=2),
                                UpsamplingNearest2d(scale_factor=2),
                                            ])
        self.up3 =LayerChoice([ UpsamplingBilinear2d(scale_factor=2),
                                UpsamplingNearest2d(scale_factor=2),
                                            ])
        self.up4 = LayerChoice([ UpsamplingBilinear2d(scale_factor=2),
                                UpsamplingNearest2d(scale_factor=2),
                                            ])
        # initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        mid4 = self.up1(center)
        if mid4.size() != enc4.size():
            _, _, height1, width1 = enc4.size()
            mid4 = F.interpolate(mid4, (height1, width1), mode='bilinear')

        dec4 = self.dec4(torch.cat([mid4, enc4], 1))

        mid3 = self.up2(dec4)
        # if mid3.size() != enc3.size():
        #     _, _, height1, width1 = enc3.size()
        #     mid3 = F.interpolate(mid3, (height1, width1), mode='bilinear')

        dec3 = self.dec3(torch.cat([mid3, enc3], 1))
        mid2 = self.up3(dec3)
        dec2 = self.dec2(torch.cat([mid2, enc2], 1))
        mid1 = self.up4(dec2)
        dec1 = self.dec1(torch.cat([mid1, enc1], 1))
        final = self.final(dec1)
        return final


def search():
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = UNet(in_channels=1, num_classes=1)
    print('start searching')

    trainer = My_DartsTrainer(model,
                             num_epochs=30
                            )

    # trainer=My_EnasTrainer(model,num_epochs=30)


    trainer.fit()

    exported_arch = trainer.export()
    from nni.retiarii import fixed_arch
    import json
    json.dump(exported_arch, open('layout_darts30.json', 'w'))


    with fixed_arch('layout_darts30.json'):
        final_model = UNet(in_channels=1, num_classes=1)
        print('final model:',final_model)
    return final_model

# search()

    # with open('best_cellarch.pkl', 'rb') as file:
    #     exported_arch1 = pickle.load(file)
    #
    # with fixed_arch(exported_arch1):
    #     final_model = CNN(n_layers=4)
    #     print('model2:',final_model)
    # final_architecture = trainer.export()
    # print('Final architecture:', trainer.export())
    # import json
    # json.dump(trainer.export(), open('checkpoint.json', 'w'))
    # with fixed_arch('checkpoint.json'):
    #     model = CNN(n_layers=4)
    #     print('model',model)
    # exported_arch1 = experiment.export_top_models(top_k =2,formatter='dict')[1]

# if __name__ == '__main__':
#     model = CNN(n_layers=3)
#     # print(model)
#     x = torch.randn(32, 1, 64, 64)
#     with torch.no_grad():
#         final = model(x)
#         print(final.shape)