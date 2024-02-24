import numpy as np
import torch.nn as nn
from hb_enastrain import *
from hb_enastrain import My_Sampling
import random
import sys
sys.path.insert(0, '../source')
from dataset import VaryGeoDataset, FixGeoDataset,VaryGeoDataset_PairedSolution
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda,to4DTensor

import torch
import torch.nn.init as init
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
torch.manual_seed(123)
params1 = {
        'channel1': 16,
        'channel2': 32,
        'channel3': 16
    }
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device(f"cuda:{3}" if torch.cuda.is_available() else "cpu")
print(device)
nxOF = 50
nyOF = 50
nx = nxOF+2
ny = nyOF+2
h = 0.01
NvarInput = 2
NvarOutput = 1
scalarList = [-0.1, 0.0, 0.1]
SolutionList = []
MeshList = []
for scalar in scalarList:
    OFcaseName_ = './TemplateCase' + str(scalar)
    nx = nxOF + 2
    ny = nyOF + 2
    R = 0.5
    L = 0
    l = 0.5
    h = 0.01
    idx = np.asarray(range(1, nx - 1, 1))
    idy = np.asarray(range(1, ny - 1, 1))
    leftY = np.linspace(-l / 2 - L / 2, l / 2 + L / 2, ny)
    rightY = np.linspace(-l / 2 - L / 2, l / 2 + L / 2, ny)
    leftX = []
    rightX = []
    for i in leftY:
        if i > -l / 2 and i < l / 2:
            leftX.append(+np.cos(2 * np.pi * i) * scalar - R)
            rightX.append(-np.cos(2 * np.pi * i) * scalar + R)
        else:  # 顶点值直接就知道了
            leftX.append(-R);
            rightX.append(R)
    leftX = np.asarray(leftX)
    rightX = np.asarray(rightX)
    lowX = np.linspace(-R, R, nx);
    lowY = lowX * 0 - l / 2 - L / 2
    upX = lowX
    upY = lowY + l + L
    myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                       lowX, lowY, upX, upY, h, False, True, './Mesh' + str(scalar) + '.pdf')

    MeshList.append(myMesh)
    OFPic = convertOFMeshToImage(OFcaseName_ + '/3200/C', [OFcaseName_ + '/3200/U', OFcaseName_ + '/3200/p'],
                                 [0, 1, 0, 1], 0.0, False)
    OFU = OFPic[:, :, 2]
    OFV = OFPic[:, :, 3]
    OFP = OFPic[:, :, 4]
    Vmag_True = np.sqrt(OFU ** 2 + OFV ** 2)
    SolutionList.append(OFPic[:, :, 2:])
batchSize = len(scalarList)

nEpochs = 20000
lr = 0.001
Ns = 1
nu = 0.01
criterion = nn.MSELoss()

padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)
train_set = VaryGeoDataset_PairedSolution(MeshList, SolutionList)

# training_data_loader = DataLoader(dataset=train_set,
#                                   batch_size=batchSize)
# r = 0.5
# R = 1
# dtheta = 0
# leftX = r * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
# rightX = R * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
# lowX = np.linspace(leftX[0], rightX[0], 49)
# ny = len(leftX);
# nx = len(lowX)
In = 2
Out = 1

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

    # def _initialize_weights(self):
    #
    #     for layer in [self.conv1, self.conv2, self.conv3,
    #                   self.conv11, self.conv22, self.conv33,
    #                   self.conv111, self.conv222, self.conv333]:
    #         for i in [0,1,2]:
    #             try:
    #                 index=layer[i]
    #                 init.kaiming_normal_(index.weight, mode='fan_out', nonlinearity='relu')
    #             except:
    #                 pass
    #
    #
    #         # for i in [3,4,5]:
    #         #     layer_dw =layer[i]
    #         #     init.kaiming_normal_(layer_dw.depthwise.weight, mode='fan_out', nonlinearity='relu')
    #         #     init.kaiming_normal_(layer_dw.pointwise.weight, mode='fan_out', nonlinearity='relu')
    #
    #     for layer in [self.conv4,self.conv44,self.conv444]:
    #         for i in [0, 1, 2]:
    #             init.kaiming_normal_(layer[i].weight)
    #         # for i in [3,4,5]:
    #         #     layer_dw = layer[i]
    #         #     init.kaiming_normal_(layer_dw.depthwise.weight)
    #         #     init.kaiming_normal_(layer_dw.pointwise.weight)

def prediction_error(truth, output):
    criterion = torch.nn.MSELoss()
    loss = torch.sqrt(criterion(truth, output[:, 2:3, :, :]) / criterion(truth, truth * 0))
    res = dict()
    res["pred_error"] = loss.item()
    return res

def reward_loss(truth, output):
    criterion = torch.nn.MSELoss()
    loss = torch.sqrt(criterion(truth, output[:, 2:3, :, :]) / criterion(truth, truth * 0))
    reward=1/loss
    return reward


def enaself_model():

    # train_set = get_dataset(mode='train')
    criterion = torch.nn.MSELoss()
    model = HBCNN_5(params1,h,nx, ny, NvarInput, NvarOutput,'kaiming')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batchSize = 3
    num_epochs = 30000

    trainer = My_EnasTrainer(model,
                          loss=criterion,
                          metrics=prediction_error,
                          reward_function=reward_loss,
                          optimizer=optimizer,
                          batch_size=batchSize,
                          num_epochs=num_epochs,
                          dataset=train_set,
                          device=device,
                          log_frequency=10,
                          ctrl_kwargs={})

    # trainer = My_DartsTrainer(model,num_epochs,dataset=train_set,device=device)
    trainer.fit()
    exported_arch = trainer.export()
    from nni.retiarii import fixed_arch
    import json
    json.dump(exported_arch, open('enas30000.json', 'w'))
    with fixed_arch('enas30000.json'):
        final_model = HBCNN_5(params1,h,nx, ny, NvarInput, NvarOutput,'kaiming')
        print('final model:', final_model)
    return final_model
def model5():
    model = HBCNN_5(params1, In, Out)
    a=My_Sampling(model,dataset_train=train_set,dataset_valid=train_set)
    model=[]
    for i in [1,2]:
        model_self=a._resample(seed=i)
        print(model_self)
        from nni.retiarii import fixed_arch
        with fixed_arch(model_self):
            final_model = HBCNN_5(params1, In, Out)
        model.append(final_model)
    return model

# enaself_model()