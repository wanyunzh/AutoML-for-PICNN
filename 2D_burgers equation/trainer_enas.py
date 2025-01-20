import torch
import torch.optim as optim
import random   
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from nni.retiarii import fixed_arch
from nni.retiarii.oneshot.pytorch import EnasTrainer,DartsTrainer,ProxylessTrainer,SinglePathTrainer
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup
import scipy.io as scio
from torch.optim.lr_scheduler import StepLR
import os
from torch.nn.utils import weight_norm
import random
from hpo_utils import *
import time
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
            DerFilter = dxdy_laplace,
            resol = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').to(device)

        self.dx = Conv2dDerivative(
            DerFilter = grad_x,
            resol = (dx*1),
            kernel_size = 5,
            name = 'dx_operator').to(device)

        self.dy = Conv2dDerivative(
            DerFilter = grad_y,
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



device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# define the high-order finite difference kernels
dxdy_laplace = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

grad_y = [[[[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1/12, -8/12, 0, 8/12, -1/12],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]]]

grad_x = [[[[0, 0, 1/12, 0, 0],
               [0, 0, -8/12, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 8/12, 0, 0],
               [0, 0, -1/12, 0, 0]]]]

data_dir = 'data/2dBurgers/burgers_1501x2x128x128.mat'    
data = scio.loadmat(data_dir)
uv = data['uv'] # [t,c,h,w]  

# initial conidtion
uv0 = uv[0:1,...] 
input = torch.tensor(uv0, dtype=torch.float32).to(device)

# set initial states for convlstm
num_convlstm = 1
(h0, c0) = (torch.randn(1, 256, 16, 16), torch.randn(1, 256, 16, 16))
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
n_iters_adam = 5000
lr_adam = 1e-3 #1e-3 
loss_func = loss_generator(dt, dx)
start = time.time()
train_loss_list = []
second_last_state = []
prev_output = []
batch_loss = 0.0
best_error=100
fig_save_path = '../figures/' 
params = {
        'UNARY_OPS': 'square',
        'WEIGHT_INIT': 'zero',
        'WEIGHT_OPS': 'one',
    }
class My_EnasTrainer(EnasTrainer):

    def __init__(self, model, num_epochs, optimizer=None, loss=None, metrics=None,
                 dataset=None,reward_function=None,
                 batch_size=32, workers=4, device=None, log_frequency=None,
                 grad_clip=5., entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999,
                 ctrl_lr=0.00035, ctrl_steps_aggregate=1, ctrl_kwargs=None):
        super(My_EnasTrainer,self).__init__(model, loss, metrics, reward_function,
                 optimizer, num_epochs, dataset,
                 batch_size, workers, device, log_frequency,
                 grad_clip, entropy_weight, skip_weight, baseline_decay,
                 ctrl_lr, ctrl_steps_aggregate, ctrl_kwargs)



    def init_dataloader(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler =  StepLR(self.optimizer, step_size=100, gamma=0.97)  

    def _train_model(self, epoch):

        self.model.train()
        self.controller.eval()
        meters = AverageMeterGroup()
        hidden_state = initial_state
        u0 = input
        self.model.zero_grad()
        self._resample()
        self.model.to(device)
        output, second_last_state = self.model(hidden_state, u0)
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
        weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_difference1, epoch,0)
        weight_search2 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_difference2, epoch,0)
        loss_search1 = torch.mean(torch.abs(post_difference1 * weight_search1))
        loss_search2 = torch.mean(torch.abs(post_difference2 * weight_search2))
        loss = loss_search1 + loss_search2
        if epoch%10==0:
            print('loss pde:',loss)
        metrics = {'res': loss.item()}
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        metrics['loss'] = loss.item()
        meters.update(metrics)


    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        meters = AverageMeterGroup()
        self.ctrl_optim.zero_grad()
        hidden_state = initial_state
        u0 = input
        self._resample()

        output, second_last_state = self.model(hidden_state, u0)
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
        reward=1/error
        if epoch%10==0:
            print('val_reward: ', reward)
        metrics = {}
        if self.entropy_weight:
            reward += self.entropy_weight * self.controller.sample_entropy.item()
        self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
        loss = self.controller.sample_log_prob * (reward - self.baseline)
        if self.skip_weight:
            loss += self.skip_weight * self.controller.sample_skip_penalty
        metrics['reward'] = reward
        metrics['loss'] = loss.item()
        metrics['ent'] = self.controller.sample_entropy.item()
        metrics['log_prob'] = self.controller.sample_log_prob.item()
        metrics['baseline'] = self.baseline
        metrics['skip'] = self.controller.sample_skip_penalty

        loss /= self.ctrl_steps_aggregate
        loss.backward()
        meters.update(metrics)
        ctrl_step=0
        if (ctrl_step + 1) % self.ctrl_steps_aggregate == 0:
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip)
            self.ctrl_optim.step()
            self.ctrl_optim.zero_grad()


    def fit(self):
        for i in range(self.num_epochs):
            self._train_model(i)
            if i%2==0:
                self._train_controller(i)



class My_DartsTrainer(DartsTrainer):

    def __init__(self, model, num_epochs, optimizer=None, loss=None, metrics=None,
                 dataset=None,grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=32, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False):
        super(My_DartsTrainer,self).__init__(model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip,
                 learning_rate, batch_size, workers,
                 device, log_frequency,
                 arc_learning_rate, unrolled)



    def _init_dataloader(self):
        self.model_optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler =  StepLR(self.model_optim, step_size=100, gamma=0.97)  


    def _train_one_epoch(self, epoch):
        print('Epoch is: ',epoch)
        self.model.train()
        meters = AverageMeterGroup()
        
        hidden_state = initial_state
        u0 = input

        hidden_state_1 = initial_state
        u0_1 = input
        self.model.to(device)
        output, second_last_state = self.model(hidden_state, u0)
        output1, second_last_state1 = self.model(hidden_state_1, u0_1)
        output_valid=output
        # [t, c, height (Y), width (X)]
        output = torch.cat(tuple(output), dim=0)  
        # concatenate the initial state to the output for central diff
        output = torch.cat((u0.to(device), output), dim=0)

        output1 = torch.cat(tuple(output1), dim=0)  
        # concatenate the initial state to the output for central diff
        output1 = torch.cat((u0.to(device), output1), dim=0)

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
        output1 = torch.cat((output1[:, :, :, -2:], output1.clone(), output1[:, :, :, 0:3]), dim=3) #左边两列，output,右边三列，shape: ([102, 2, 128, 133])
        # Padding y axis due to periodic boundary condition
        # shape: [t, c, h, w]
        output1 = torch.cat((output1[:, :, -2:, :], output1.clone(), output1[:, :, 0:3, :]), dim=2) #shape: ([102, 2, 133, 133])

        f_u1, f_v1 = loss_func.get_phy_Loss(output1) #fu,fv是residue
        
        if params['WEIGHT_INIT'] == 'one':
            init_weight = torch.ones_like(f_u1)
        else:
            init_weight = torch.zeros_like(f_u1)

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

        post_difference1 = UNARY_OPS[params['UNARY_OPS']](f_u1)
        post_difference2 = UNARY_OPS[params['UNARY_OPS']](f_v1)
        weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_difference1, epoch,0)
        weight_search2 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_difference2, epoch,0)
        loss_search1 = torch.mean(torch.abs(post_difference1 * weight_search1))
        loss_search2 = torch.mean(torch.abs(post_difference2 * weight_search2))
        loss1 = loss_search1 + loss_search2
        if epoch%10==0:
            print('loss pde:',loss1)
        metrics = {'res': loss1.item()}

        # phase 1. architecture step
        self.ctrl_optim.zero_grad()
        loss1.backward()
        # loss.backward()
        self.ctrl_optim.step()
        # get loss
        # Padding x axis due to periodic boundary condition
        output = torch.cat((output[:, :, :, -2:], output.clone(), output[:, :, :, 0:3]), dim=3) #左边两列，output,右边三列，shape: ([102, 2, 128, 133])
        # Padding y axis due to periodic boundary condition
        # shape: [t, c, h, w]
        output = torch.cat((output[:, :, -2:, :], output.clone(), output[:, :, 0:3, :]), dim=2) #shape: ([102, 2, 133, 133])

        f_u, f_v = loss_func.get_phy_Loss(output) #fu,fv是residue
        
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

        post_difference11 = UNARY_OPS[params['UNARY_OPS']](f_u)
        post_difference21 = UNARY_OPS[params['UNARY_OPS']](f_v)
        weight_search11 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_difference11, epoch,0)
        weight_search21 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_difference21, epoch,0)
        loss_search11 = torch.mean(torch.abs(post_difference11 * weight_search11))
        loss_search21 = torch.mean(torch.abs(post_difference21 * weight_search21))
        loss = loss_search11 + loss_search21
        # phase 2: child network step
        self.model_optim.zero_grad()
        loss.backward()
        if epoch%5 ==0:
            print('loss is:',loss)
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
        self.model_optim.step()
        self.scheduler.step()
        metrics = {'res': loss.item()}
        metrics['loss'] = loss.item()
        meters.update(metrics)


# class My_SPOS(SinglePathTrainer):
#     """
#     Single-path trainer. Samples a path every time and backpropagates on that path.
#
#     Parameters
#     ----------
#     model : nn.Module
#         Model with mutables.
#     loss : callable
#         Called with logits and targets. Returns a loss tensor.
#     metrics : callable
#         Returns a dict that maps metrics keys to metrics data.
#     optimizer : Optimizer
#         Optimizer that optimizes the model.
#     num_epochs : int
#         Number of epochs of training.
#     dataset_train : Dataset
#         Dataset of training.
#     dataset_valid : Dataset
#         Dataset of validation.
#     batch_size : int
#         Batch size.
#     workers: int
#         Number of threads for data preprocessing. Not used for this trainer. Maybe removed in future.
#     device : torch.device
#         Device object. Either ``torch.device("cuda")`` or ``torch.device("cpu")``. When ``None``, trainer will
#         automatic detects GPU and selects GPU first.
#     log_frequency : int
#         Number of mini-batches to log metrics.
#     """
#
#     def __init__(self, model, loss, metrics,
#                  optimizer, num_epochs, dataset_train, dataset_valid,
#                  batch_size=64, workers=4, device=None, log_frequency=None):
#         super(My_SPOS, self).__init__(model, loss, metrics,
#                  optimizer, num_epochs, dataset_train, dataset_valid,
#                  batch_size, workers, device, log_frequency)
#
#
#
#     def _train_one_epoch(self, epoch):
#         self.model.train()
#         meters = AverageMeterGroup()
#         for step, x in enumerate(self.train_loader):
#             [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(x)
#             Para.to(self.device)
#             truth.to(self.device)
#             self.optimizer.zero_grad()
#
#             self._resample()
#             self.model.to(self.device)
#             output = self.model(Para)
#             output_pad = udfpad(output)
#             outputV = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
#                                                      output_pad.shape[2],
#                                                      output_pad.shape[3])
#             # print(outputV)
#             # print(outputV.shpae)
#             for j in range(batchSize):
#                 # Impose BC
#                 outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
#                     outputV[j, 0, 1:2, padSingleSide:-padSingleSide])
#                 outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
#                     outputV[j, 0, -2:-1, padSingleSide:-padSingleSide])
#                 outputV[j, 0, :, -padSingleSide:] = 0
#                 outputV[j, 0, :, 0:padSingleSide] = Para[j, 0, 0, 0]
#             dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
#             d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
#             dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
#             d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
#             continuity = (d2vdy2 + d2vdx2);
#             loss = self.loss(continuity, continuity * 0)
#             loss.backward()
#             self.optimizer.step()
#             metrics = self.metrics(outputV, truth)
#             metrics["loss"] = loss.item()
#             meters.update(metrics)
#             if self.log_frequency is not None and step % self.log_frequency == 0:
#                 _logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
#                              self.num_epochs, step + 1, len(self.train_loader), meters)
#
#     def _validate_one_epoch(self, epoch):
#         self.model.eval()
#         meters = AverageMeterGroup()
#         with torch.no_grad():
#             for step, x in enumerate(self.valid_loader):
#                 [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(x)
#                 Para.to(self.device)
#                 truth.to(self.device)
#
#                 self._resample()
#                 with torch.no_grad():
#                     output = self.model(Para)
#                 output_pad = udfpad(output)
#                 outputV = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
#                                                          output_pad.shape[2],
#                                                          output_pad.shape[3])
#                 for j in range(batchSize):
#                     # Impose BC
#                     outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
#                         outputV[j, 0, 1:2, padSingleSide:-padSingleSide])
#                     outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
#                         outputV[j, 0, -2:-1, padSingleSide:-padSingleSide])
#                     outputV[j, 0, :, -padSingleSide:] = 0
#                     outputV[j, 0, :, 0:padSingleSide] = Para[j, 0, 0, 0]
#                 criterion = torch.nn.MSELoss()
#                 loss = torch.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0))
#                 metrics = self.metrics(truth, outputV)
#                 metrics["loss"] = loss.item()
#                 meters.update(metrics)
#                 if self.log_frequency is not None and step % self.log_frequency == 0:
#                     _logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
#                                  self.num_epochs, step + 1, len(self.valid_loader), meters)
#
class My_Sampling(EnasTrainer):


    def __init__(self, model, loss=None, metrics=None,
                 optimizer=None, num_epochs=None, dataset_train=None, dataset_valid=None,
                 batch_size=64, workers=4, device=None, log_frequency=None):
        super(My_Sampling, self).__init__(model, loss, metrics,
                                      optimizer, num_epochs, dataset_train, dataset_valid,
                                      batch_size, workers, device, log_frequency)
    def _resample(self,seed):
        result = {}
        for step,(name, module) in enumerate(self.nas_modules):
            if name not in result:
                random.seed(seed+step)
                result[name] = random.randint(0, len(module) - 1)
            module.sampled = result[name]
        return result

