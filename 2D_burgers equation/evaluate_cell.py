import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import nni
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.nn.utils import weight_norm
import random
from hpo_utils import *
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
seed=66
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float32)
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# define the high-order finite difference kernels
lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

partial_y = [[[[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1/12, -8/12, 0, 8/12, -1/12],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]]]

partial_x = [[[[0, 0, 1/12, 0, 0],
               [0, 0, -8/12, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 8/12, 0, 0],
               [0, 0, -1/12, 0, 0]]]]

# generalized version
# def initialize_weights(module):
#     ''' starting from small initialized parameters '''
#     if isinstance(module, nn.Conv2d):
#         c = 0.1
#         module.weight.data.uniform_(-c*np.sqrt(1 / np.prod(module.weight.shape[:-1])),
#                                      c*np.sqrt(1 / np.prod(module.weight.shape[:-1])))
     
#     elif isinstance(module, nn.Linear):
#         module.bias.data.zero_()

# specific parameters for burgers equation
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
        return (Variable(prev_state[0]).to(device), Variable(prev_state[1]).to(device))
    
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()

        # 第一个卷积层、归一化和激活函数
        self.conv1 = weight_norm(nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, padding_mode='circular'))
        self.norm1 = nn.GroupNorm(32, middle_channels)
        self.act1 = nn.GELU()

        # 第二个卷积层、归一化和激活函数
        self.conv2 = weight_norm(nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'))
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        # 第一个卷积块：卷积 -> 归一化 -> 激活
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        # 第二个卷积块：卷积 -> 归一化 -> 激活
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x

class encoder_block(nn.Module):
    ''' encoder with CNN '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):
        
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = weight_norm(nn.Conv2d(self.input_channels, 
            self.hidden_channels, self.input_kernel_size, self.input_stride, 
            self.input_padding, bias=True, padding_mode='circular')) 
        

        self.conv = weight_norm(nn.Conv2d(self.input_channels, 
            self.hidden_channels, 3, 1, 1, bias=True, padding_mode='circular'))
        
        # layers1 = [
        #     nn.GroupNorm(32, out_channels),
        #     nn.GELU(),
        # ]


        #Conv2d(self.input_channels(8), self.hidden_channels32(32), kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), padding_mode=circular)
        self.act = nn.ReLU()

        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))
class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, polling=True):
        super(_EncoderBlock, self).__init__()

        # 第一个卷积层、归一化和激活函数
        self.conv1 = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'))
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.GELU()

        # 第二个卷积层、归一化和激活函数
        self.conv2 = weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'))
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.GELU()

        # 可选的池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if polling else None

    def forward(self, x):
        # 先进行池化操作（如果有的话）
        if self.pool is not None:
            x = self.pool(x)

        # 第一个卷积块：卷积 -> 归一化 -> 激活
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        # 第二个卷积块：卷积 -> 归一化 -> 激活
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x

class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (10.0/200), dx = (20.0/128)):
        ''' Construct the derivatives, X = Width, Y = Height '''
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lapl_op,
            resol = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').to(device)

        self.dx = Conv2dDerivative(
            DerFilter = partial_x,
            resol = (dx*1),
            kernel_size = 5,
            name = 'dx_operator').to(device)

        self.dy = Conv2dDerivative(
            DerFilter = partial_y,
            resol = (dx*1),
            kernel_size = 5,
            name = 'dy_operator').to(device)

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (dt*2),
            kernel_size = 3,
            name = 'partial_t').to(device)

    def get_phy_Loss(self, output):

        # spatial derivatives
        laplace_u = self.laplace(output[1:-1, 0:1, :, :])  # [t,c,h,w],([102, 2, 133, 133]) to ([100, 1, 133, 133])
        laplace_v = self.laplace(output[1:-1, 1:2, :, :])

        u_x = self.dx(output[1:-1, 0:1, :, :])
        u_y = self.dy(output[1:-1, 0:1, :, :])
        v_x = self.dx(output[1:-1, 1:2, :, :])
        v_y = self.dy(output[1:-1, 1:2, :, :])

        # temporal derivative - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny,1,lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # temporal derivative - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny,1,lent)
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, 1, lent-2)
        v_t = v_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u = output[1:-1, 0:1, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]
        v = output[1:-1, 1:2, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]

        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        R = 200.0

        # 2D burgers eqn
        f_u = u_t + u * u_x + v * u_y - (1/R) * laplace_u
        f_v = v_t + u * v_x + v * v_y - (1/R) * laplace_v

        return f_u, f_v


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
    ''' 
    axis_lim: [xmin, xmax, ymin, ymax]
    uv_lim: [u_min, u_max, v_min, v_max]
    num: Number of time step
    '''

    # get the limit 
    xmin, xmax, ymin, ymax = axis_lim
    u_min, u_max, v_min, v_max = uv_lim

    # grid
    x = np.linspace(xmin, xmax, 128+1)
    x = x[:-1]
    x_star, y_star = np.meshgrid(x, x)
    
    u_star = true[num, 0, 1:-1, 1:-1]
    u_pred = output[num, 0, 1:-1, 1:-1].detach().cpu().numpy()

    v_star = true[num, 1, 1:-1, 1:-1]
    v_pred = output[num, 1, 1:-1, 1:-1].detach().cpu().numpy()
    return u_star, u_pred, v_star, v_pred


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''

    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded with strict=False (ignoring missing keys).')

    return model, optimizer, scheduler


def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))

def relative_l2_norm(ten_pred, ten_true):
    # 将输入转换为 NumPy 数组
    ten_pred = np.array(ten_pred)
    ten_true = np.array(ten_true)
    
    # 计算预测误差和真实值的 L2 norm，按样本计算
    l2_error = np.sqrt(np.sum((ten_pred - ten_true) ** 2, axis=(1, 2, 3)))
    l2_true = np.sqrt(np.sum(ten_true ** 2, axis=(1, 2, 3)))
    
    # 计算每个样本的 relative L2 norm，并求平均
    relative_l2 = np.mean(l2_error / l2_true)
    
    return relative_l2


def traintest_burgers(model_cls):
    params = {
        'UNARY_OPS': 'square',
        'WEIGHT_INIT': 'one',
        'WEIGHT_OPS': 'max',
    }
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(device)
    ######### download the ground truth data ############
    data_dir = '../data/2dBurgers/burgers_1501x2x128x128.mat'    
    data = scio.loadmat(data_dir)
    uv = data['uv'] # [t,c,h,w]  

    # initial conidtion
    uv0 = uv[0:1,...] 
    input = torch.tensor(uv0, dtype=torch.float32).to(device)

    # set initial states for convlstm
    num_convlstm = 1
    (h0, c0) = (torch.randn(1, 384, 16, 16), torch.randn(1, 384, 16, 16))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))
    
    # grid parameters
    time_steps = 151
    dt = 0.002
    dx = 1.0 / 128

    ################# build the model #####################
    time_batch_size = 100
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    n_iters_adam = 4000
    lr_adam = 1e-3 #1e-3 
    model_save_path = '../model/loss_searche_nas.pt'
    fig_save_path = '../figures/'  

    model = model_cls().to(device)
    # model = CNN()
    model = model.to(device)

    start = time.time()
    train_loss_list = []
    second_last_state = []
    prev_output = []

    batch_loss = 0.0
    # best_loss = 1e4
    best_error=100

    # load previous model
    optimizer = optim.Adam(model.parameters(), lr=lr_adam) 
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)  

    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    loss_func = loss_generator(dt, dx)
        
    for epoch in range(n_iters_adam):
        # input: [t,b,c,h,w]
        optimizer.zero_grad()
        batch_loss = 0 
        
        for time_batch_id in range(num_time_batch):
            # update the first input for each time batch
            if time_batch_id == 0:
                hidden_state = initial_state
                u0 = input
            else:
                hidden_state = state_detached
                u0 = prev_output[-2:-1].detach() # second last output

            # output is a list
            output, second_last_state = model(hidden_state, u0)

            output_valid=output

            # [t, c, height (Y), width (X)]
            output = torch.cat(tuple(output), dim=0)  

            # concatenate the initial state to the output for central diff
            output = torch.cat((u0.to(device), output), dim=0)
            outputv = torch.cat(tuple(output_valid), dim=0)  
            outputv = torch.cat((input.to(device), outputv), dim=0)
            # Padding x and y axis due to periodic boundary condition
            outputv = torch.cat((outputv[:, :, :, -1:], outputv, outputv[:, :, :, 0:2]), dim=3)
            outputv = torch.cat((outputv[:, :, -1:, :], outputv, outputv[:, :, 0:2, :]), dim=2)
            # [t, c, h, w]
            truth = uv[0:time_steps,:,:,:]
            # [101, 2, 131, 131]
            truth = np.concatenate((truth[:, :, :, -1:], truth, truth[:, :, :, 0:2]), axis=3)
            truth = np.concatenate((truth[:, :, -1:, :], truth, truth[:, :, 0:2, :]), axis=2)
            # post-process
            ten_true = []
            ten_pred = []
            for i in range(50): 
                u_star, u_pred, v_star, v_pred = post_process(outputv, truth, [0,1,0,1], 
                    [-0.7,0.7,-1.0,1.0], num=i, fig_save_path=fig_save_path)

                ten_true.append([u_star, v_star])
                ten_pred.append([u_pred, v_pred])

            error = relative_l2_norm(ten_pred, ten_true)
            print('The predicted L2 error is: ', error)
            
            # get loss
            # Padding x axis due to periodic boundary condition
            output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3) #左边两列，output,右边三列，shape: ([102, 2, 128, 133])
            # Padding y axis due to periodic boundary condition
            # shape: [t, c, h, w]
            output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2) #shape: ([102, 2, 133, 133])

            f_u, f_v = loss_func.get_phy_Loss(output) #fu,fv是residue

            if (epoch == 0) and (time_batch_id == 0):
                if params['WEIGHT_INIT'] == 'one':
                    init_weight = torch.ones_like(f_u)
                else:
                    init_weight = torch.zeros_like(f_u)

                WEIGHT_OPS1 = {
                            'normalize': P_OHEM1(init_weight),
                            'adaptive': Loss_Adaptive1(init_weight),
                            'max': Max1(init_weight),
                            'one': One(init_weight),
                        }

                WEIGHT_OPS2 = {
                    'normalize': P_OHEM2(init_weight),
                    'adaptive': Loss_Adaptive2(init_weight),
                    'max': Max2(init_weight),
                    'one': One(init_weight),
                }

            post_difference1 = UNARY_OPS[params['UNARY_OPS']](f_u)
            post_difference2 = UNARY_OPS[params['UNARY_OPS']](f_v)
            weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_difference1, epoch,time_batch_id)
            weight_search2 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_difference2, epoch,time_batch_id)
            loss_search1 = torch.mean(torch.abs(post_difference1 * weight_search1))
            loss_search2 = torch.mean(torch.abs(post_difference2 * weight_search2))
            loss = loss_search1 + loss_search2
            # mse_loss = nn.MSELoss()
            # loss =  mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(f_v, torch.zeros_like(f_v).cuda())
            loss.backward(retain_graph=True)
            batch_loss += loss.item()
            # update the state and output for next batch
            prev_output = output
            state_detached = []
            for i in range(len(second_last_state)):
                (h, c) = second_last_state[i]
                state_detached.append((h.detach(), c.detach())) # hidden state

        optimizer.step()
        scheduler.step()

        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.10f' % ((epoch+1), n_iters_adam, ((epoch+1)/n_iters_adam*100.0), 
            batch_loss))
        train_loss_list.append(batch_loss)

        if error < best_error:
            save_checkpoint(model, optimizer, scheduler, model_save_path)
            epochbest = epoch
            best_error = error
    print('Best epoch: ',epochbest)
    end = time.time()
    os.makedirs('./model/', exist_ok=True)  
    print('The training time is: ', (end-start))

    ########### model inference ##################
    time_batch_size_load = 150
    steps_load = time_batch_size_load + 1
    num_time_batch = int(time_steps / time_batch_size_load)
    effective_step = list(range(0, steps_load))  
    # print('Model is:')
    # print(model)
    
    model = model_cls().to(device)
    # 设置模型的 step 和 effective_step 属性
    model.step = steps_load
    model.effective_step = effective_step

    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_dir=model_save_path) 
    output, _ = model(initial_state, input)

    # shape: [t, c, h, w] 
    output = torch.cat(tuple(output), dim=0)  
    output = torch.cat((input.to(device), output), dim=0)
  
    # Padding x and y axis due to periodic boundary condition
    output = torch.cat((output[:, :, :, -1:], output, output[:, :, :, 0:2]), dim=3)
    output = torch.cat((output[:, :, -1:, :], output, output[:, :, 0:2, :]), dim=2)

    # [t, c, h, w]
    truth = uv[0:time_steps,:,:,:]

    # [101, 2, 131, 131]
    truth = np.concatenate((truth[:, :, :, -1:], truth, truth[:, :, :, 0:2]), axis=3)
    truth = np.concatenate((truth[:, :, -1:, :], truth, truth[:, :, 0:2, :]), axis=2)

    # post-process
    ten_true = []
    ten_pred = []
    for i in range(100,150): 
        u_star, u_pred, v_star, v_pred = post_process(output, truth, [0,1,0,1], 
            [-0.7,0.7,-1.0,1.0], num=i, fig_save_path=fig_save_path)

        ten_true.append([u_star, v_star])
        ten_pred.append([u_pred, v_pred])

    # # compute the error
    error_f = frobenius_norm(np.array(ten_pred)-np.array(ten_true)) / frobenius_norm(
        np.array(ten_true))
    print('The predicted errorf is: ', error_f)
    # compute the L2 error
    error = relative_l2_norm(ten_pred, ten_true)
    print('The predicted error is: ', error)
    nni.report_final_result(1 / float(error))




    # u_pred = output[:-1, 0, :, :].detach().cpu().numpy()
    # u_pred = np.swapaxes(u_pred, 1, 2) # [h,w] = [y,x]
    # u_true = truth[:, 0, :, :]

    # t_true = np.linspace(0, 2, time_steps)
    # t_pred = np.linspace(0, 2, time_steps)

    # plt.plot(t_pred, u_pred[:, 32, 32], label='x=32, y=32, CRL')
    # plt.plot(t_true, u_true[:, 32, 32], '--', label='x=32, y=32, Ref.')
    # plt.xlabel('t')
    # plt.ylabel('u')
    # plt.xlim(0, 2)
    # plt.legend()
    # plt.savefig(fig_save_path + "x=32,y=32.png")
    # plt.close("all")
    # # plt.show()

    # # plot train loss
    # plt.figure()
    # plt.plot(train_loss, label = 'train loss')
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig(fig_save_path + 'train loss.png', dpi = 300)




# traintest_burgers(5)
















