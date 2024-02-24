import numpy as np
import torch
import torch.nn as nn
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from get_dataset import get_dataset
from hb_enastrain import My_Sampling,My_EnasTrainer,My_DartsTrainer
import random
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
r = 0.5
R = 1
dtheta = 0
leftX = r * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
rightX = R * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
lowX = np.linspace(leftX[0], rightX[0], 49)
ny = len(leftX)
nx = len(lowX)
In = 1
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
    def __init__(self, params, In, Out):
        super().__init__()
        """
        Extract basic information
        """
        self.In = In
        self.Out = Out
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
        self.conv1_1 = nn.LayerChoice([
            nn.Conv2d(params['channel2'], params['channel2'], kernel_size=5, stride=1, padding=2),
            nn.Conv2d(params['channel2'], params['channel2'], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(params['channel2'], params['channel2'], kernel_size=7, stride=1, padding=3),
            DepthwiseSeparableConv5(params['channel2'], params['channel2']),
            DepthwiseSeparableConv3(params['channel2'], params['channel2']),
            DepthwiseSeparableConv7(params['channel2'], params['channel2']),
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            nn.MaxPool2d(3, stride=1, padding=1)])

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
        self.pixel_shuffle = nn.PixelShuffle(1)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x=self.relu(self.conv1_1(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        return x


def reward_loss(truth, outputV):
    criterion = torch.nn.MSELoss()
    loss = torch.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0))
    reward=1/loss
    return reward

def prediction_error(truth, outputV):
    criterion = torch.nn.MSELoss()
    loss = torch.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0))
    res = dict()
    res["pred_error"] = loss.item()
    return res


def enaself_model():

    train_set = get_dataset(mode='train')
    criterion = torch.nn.MSELoss()
    model = HBCNN_5(params1, In, Out)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batchSize = 2
    num_epochs = 2000

    # trainer = My_EnasTrainer(model,
    #                       loss=criterion,
    #                       metrics=prediction_error,
    #                       reward_function=reward_loss,
    #                       optimizer=optimizer,
    #                       batch_size=batchSize,
    #                       num_epochs=num_epochs,
    #                       dataset=train_set,
    #                       log_frequency=10,
    #                       ctrl_kwargs={})

    trainer = My_DartsTrainer(model,num_epochs,dataset=train_set)
    trainer.fit()
    exported_arch = trainer.export()
    from nni.retiarii import fixed_arch
    import json
    json.dump(exported_arch, open('darts2000.json', 'w'))
    with fixed_arch('darts2000.json'):
        final_model = HBCNN_5(params1, In, Out)
        print('final model:', final_model)
    return final_model

def model5():
    train_set = get_dataset(mode='train')
    model = HBCNN_5(params1, In, Out)
    a=My_Sampling(model,dataset_train=train_set,dataset_valid=train_set)
    model=[]
    for i in [1,2,3,4,5]:
        model_self=a._resample(seed=i)
        print(model_self)
        from nni.retiarii import fixed_arch
        with fixed_arch(model_self):
            final_model = HBCNN_5(params1, In, Out)
        model.append(final_model)
    return model
