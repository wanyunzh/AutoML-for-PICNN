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
from torch.nn.utils import weight_norm
import random
from collections import OrderedDict
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from torch.autograd import Variable


def initialize_weights(module):

    if isinstance(module, nn.Conv2d):
        #nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1 #0.5
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))
     
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class ConvLSTMCell(nn.Module):
    ''' Convolutional LSTM '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):

        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = 4

        # padding for hidden state
        self.padding = int((self.hidden_kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='circular')

        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')

        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')

        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='circular')

        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')       

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden_tensor(self, prev_state):
        device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        return (Variable(prev_state[0]).to(device), Variable(prev_state[1]).to(device))
    
class ConvLSTMCell2(nn.Module):
    ''' Simplified Convolutional LSTM with LayerChoice '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
                 input_stride, input_padding):

        super(ConvLSTMCell2, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding

        # 定义 LayerChoice 卷积层（只包含标准卷积的 3x3 和 5x5 选项）
        self.Wxi = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(self.input_channels, self.hidden_channels, 3, 1, 1, bias=True, padding_mode='circular')),
            ("conv5x5", nn.Conv2d(self.input_channels, self.hidden_channels, 5, 1, 2, bias=True, padding_mode='circular'))
        ]))

        self.Whi = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, bias=False, padding_mode='circular')),
            ("conv5x5", nn.Conv2d(self.hidden_channels, self.hidden_channels, 5, 1, 2, bias=False, padding_mode='circular'))
        ]))

        self.Wxf = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(self.input_channels, self.hidden_channels, 3, 1, 1, bias=True, padding_mode='circular')),
            ("conv5x5", nn.Conv2d(self.input_channels, self.hidden_channels, 5, 1, 2, bias=True, padding_mode='circular'))
        ]))

        self.Whf = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, bias=False, padding_mode='circular')),
            ("conv5x5", nn.Conv2d(self.hidden_channels, self.hidden_channels, 5, 1, 2, bias=False, padding_mode='circular'))
        ]))

        self.Wxc = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(self.input_channels, self.hidden_channels, 3, 1, 1, bias=True, padding_mode='circular')),
            ("conv5x5", nn.Conv2d(self.input_channels, self.hidden_channels, 5, 1, 2, bias=True, padding_mode='circular'))
        ]))

        self.Whc = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, bias=False, padding_mode='circular')),
            ("conv5x5", nn.Conv2d(self.hidden_channels, self.hidden_channels, 5, 1, 2, bias=False, padding_mode='circular'))
        ]))

        self.Wxo = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(self.input_channels, self.hidden_channels, 3, 1, 1, bias=True, padding_mode='circular')),
            ("conv5x5", nn.Conv2d(self.input_channels, self.hidden_channels, 5, 1, 2, bias=True, padding_mode='circular'))
        ]))

        self.Who = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, bias=False, padding_mode='circular')),
            ("conv5x5", nn.Conv2d(self.hidden_channels, self.hidden_channels, 5, 1, 2, bias=False, padding_mode='circular'))
        ]))

        # 初始化偏置
        nn.init.zeros_(self.Wxi[0].bias)
        nn.init.zeros_(self.Wxf[0].bias)
        nn.init.zeros_(self.Wxc[0].bias)
        self.Wxo[0].bias.data.fill_(1.0)

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()

        # 使用 LayerChoice 并添加 weight_norm
        self.conv1 = LayerChoice(OrderedDict([
            ("conv3x3", weight_norm(nn.Conv2d(in_channels, middle_channels, 3, 1, 1, bias=False, padding_mode='circular'))),
            ("sepconv3x3", nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, padding_mode='circular')),
                weight_norm(nn.Conv2d(in_channels, middle_channels, kernel_size=1)))),
            ("sepconv5x5", nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups=in_channels, padding_mode='circular')),
                weight_norm(nn.Conv2d(in_channels, middle_channels, kernel_size=1)))),
            ("conv5x5", weight_norm(nn.Conv2d(in_channels, middle_channels, 5, 1, 2, bias=False, padding_mode='circular')))
        ]))

        self.norm1 = nn.GroupNorm(32, middle_channels)
        self.act1 = nn.GELU()

        self.conv2 = LayerChoice(OrderedDict([
            ("conv3x3", weight_norm(nn.Conv2d(middle_channels, out_channels, 3, 1, 1, bias=False, padding_mode='circular'))),
            ("sepconv3x3", nn.Sequential(
                weight_norm(nn.Conv2d(middle_channels, middle_channels, 3, 1, 1, groups=middle_channels, padding_mode='circular')),
                weight_norm(nn.Conv2d(middle_channels, out_channels, kernel_size=1)))),
            ("sepconv5x5", nn.Sequential(
                weight_norm(nn.Conv2d(middle_channels, middle_channels, 5, 1, 2, groups=middle_channels, padding_mode='circular')),
                weight_norm(nn.Conv2d(middle_channels, out_channels, kernel_size=1)))),
            ("conv5x5", weight_norm(nn.Conv2d(middle_channels, out_channels, 5, 1, 2, bias=False, padding_mode='circular')))
        ]))

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, polling=True):
        super(_EncoderBlock, self).__init__()

        # 使用 LayerChoice 并添加 weight_norm
        self.conv1 = LayerChoice(OrderedDict([
            ("conv3x3", weight_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, padding_mode='circular'))),
            ("sepconv3x3", nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, padding_mode='circular')),
                weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)))),
            ("sepconv5x5", nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups=in_channels, padding_mode='circular')),
                weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)))),
            ("conv5x5", weight_norm(nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False, padding_mode='circular')))
        ]))

        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.GELU()

        self.conv2 = LayerChoice(OrderedDict([
            ("conv3x3", weight_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, padding_mode='circular'))),
            ("sepconv3x3", nn.Sequential(
                weight_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, padding_mode='circular')),
                weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=1)))),
            ("sepconv5x5", nn.Sequential(
                weight_norm(nn.Conv2d(out_channels, out_channels, 5, 1, 2, groups=out_channels, padding_mode='circular')),
                weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=1)))),
            ("conv5x5", weight_norm(nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False, padding_mode='circular')))
        ]))

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.GELU()

        # 可选的池化层
        self.pool = LayerChoice(OrderedDict([
            ("maxpool", nn.MaxPool2d(2, 2)),
            ("avgpool", nn.AvgPool2d(2, 2))
        ])) if polling else None

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x
    
@model_wrapper
class UNet_burgers(nn.Module):
    ''' physics-informed convolutional-recurrent neural networks '''
    def __init__(self, input_channels, hidden_channels, 
        input_kernel_size, input_stride, input_padding, dt, 
        num_layers, upscale_factor):

        super().__init__()

        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        
        self.step = 101
        self.effective_step = list(range(0, 101))
        self._all_layers = []
        self.dt = dt
        self.upscale_factor = upscale_factor

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]
        self.pn=[False,True,True,True]
        # encoder - downsampling  
        for i in range(self.num_encoder+1):
            name = 'encoder{}'.format(i)
            cell = _EncoderBlock(self.input_channels[i], self.hidden_channels[i], polling=self.pn[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)   
        # encoder
        self.enc1 = getattr(self, 'encoder0')
        self.enc2 = getattr(self, 'encoder1')
        self.enc3 = getattr(self, 'encoder2')
        self.enc4 = getattr(self, 'encoder3')         
            
        # ConvLSTM
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = 'convlstm{}'.format(i)
            cell = ConvLSTMCell(
                input_channels = self.input_channels[i+1],
                hidden_channels = self.hidden_channels[i],
                input_kernel_size = self.input_kernel_size[i],
                input_stride = self.input_stride[i],
                input_padding = self.input_padding[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)  
        factors=1
        self.dec4 = _DecoderBlock(512 * factors, 256 * factors, 128 * factors)
        self.dec3 = _DecoderBlock(256 * 1, 128 * factors, 64 * factors)
        self.dec2 = _DecoderBlock(128 * factors, 64 * factors, 32 * factors)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 * factors, 32 * factors, kernel_size=3, padding=1, padding_mode='circular'),
            nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
            nn.Conv2d(32 * factors, 32 * factors, kernel_size=1, padding=0),
            nn.GroupNorm(32, 32 * factors),
            nn.GELU(),
        )
        
        # output layer
        self.output_layer = nn.Conv2d(32, 2, kernel_size = 5, stride = 1, 
                                      padding=2, padding_mode='circular')

        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, initial_state, x):
        
        self.initial_state = initial_state
        internal_state = []
        outputs = []
        second_last_state = []

        for step in range(self.step):
            xt = x
            enc1 = self.enc1(x)
            enc2 = self.enc2(enc1)
            enc3 = self.enc3(enc2)
            enc4 = self.enc4(enc3)

            # convlstm
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = 'convlstm{}'.format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state = self.initial_state[i - self.num_encoder])  
                    internal_state.append((h,c))
                
                # one-step forward
                (h, c) = internal_state[i - self.num_encoder]
                center, new_c = getattr(self, name)(enc4, h, c) #x:经过encoder得到的输出。c，v: ConvLSTM的输入
                internal_state[i - self.num_encoder] = (center, new_c)                               

            dec4 = self.dec4(torch.cat([center, enc4], 1))
            dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                    mode='bilinear'), enc3], 1))
            dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                    mode='bilinear'), enc2], 1))
            dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
            x = self.output_layer(dec1)

            # residual connection
            x = xt + self.dt * x

            if step == (self.step - 2):
                second_last_state = internal_state.copy()
                
            if step in self.effective_step:
                outputs.append(x)                

        return outputs, second_last_state

import json

# 定义映射规则
def get_mapping(key, value):
    # 特殊处理 model_5, model_8, model_11
    if key in ["model_5", "model_8", "model_11"]:
        return "maxpool" if value == 0 else "avgpool"
    # 默认映射规则
    mapping = {
        0: "conv3x3",
        1: "sepconv3x3",
        2: "sepconv5x5",
        3: "conv5x5"
    }
    return mapping.get(value, "unknown")  # 防止意外值出错

def search():
    seed = 66
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.set_default_dtype(torch.float32)
    dt = 0.002
    time_batch_size = 100
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))

    model =  UNet_burgers(
        input_channels = 2, 
        hidden_channels = [32, 64, 128, 256], 
        input_kernel_size = [4, 4, 4, 3], 
        input_stride = [2, 2, 2, 1], 
        input_padding = [1, 1, 1, 1],  
        dt = dt,
        num_layers = [3, 1],
        upscale_factor = 8)

    trainer = My_DartsTrainer(model,
                             num_epochs=4500
                            )

    # trainer=My_EnasTrainer(model,num_epochs=5000)


    trainer.fit()

    exported_arch = trainer.export()
    from nni.retiarii import fixed_arch
    import json
    json.dump(exported_arch, open('burgers_darts_5000.json', 'w'))



    # input_file = 'burgers_enas_5000.json'

    # with open(input_file, 'r') as f:
    #     architecture = json.load(f)

    # # 转换模型编号为具体名称
    # converted_architecture = {key: get_mapping(key, value) for key, value in architecture.items()}

    # # 写回原始文件
    # with open(input_file, 'w') as f:
    #     json.dump(converted_architecture, f, indent=4)

    # print(f"Converted architecture written back to {input_file}")


    with fixed_arch('burgers_darts_5000.json'):
        final_model = UNet_burgers(
        input_channels = 2, 
        hidden_channels = [32, 64, 128, 256], 
        input_kernel_size = [4, 4, 4, 3], 
        input_stride = [2, 2, 2, 1], 
        input_padding = [1, 1, 1, 1],  
        dt = dt,
        num_layers = [3, 1],
        upscale_factor = 8)
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