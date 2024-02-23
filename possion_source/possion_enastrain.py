import torch.optim as optim
from torch.utils.data import DataLoader
from search_struct import UNet
from get_dataset import get_dataset
from hpo_utils import *
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup
import nni
import random
from nni.retiarii.oneshot.pytorch import EnasTrainer,DartsTrainer,ProxylessTrainer,SinglePathTrainer
def convert_to_4D_tensor(data_list):
    result_list = []
    for item in data_list:
        if len(item.shape) == 3:
            item = torch.tensor(item.reshape([item.shape[0], 1, item.shape[1], item.shape[2]]))
            result_list.append(item.float().to(device))
        else:
            item = torch.tensor(item)
            result_list.append(item.float().to(device))
    return result_list

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device(f"cuda:{7}" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
lr = 0.001
train_size = 256
all_set,dydeta, dydxi, dxdxi, dxdeta, Jinv = get_dataset()
train_set = all_set[:train_size]
training_data_loader = DataLoader(train_set, batch_size=32)
epochs = 15000
input_scale_factor = 500

class My_EnasTrainer(EnasTrainer):


    def __init__(self, model, num_epochs, optimizer=None, loss=None, metrics=None,
                 dataset=None,reward_function=None,
                 batch_size=32, workers=4, device=None, log_frequency=None,
                 grad_clip=5., entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999,
                 ctrl_lr=0.00035, ctrl_steps_aggregate=2, ctrl_kwargs=None):
        super(My_EnasTrainer,self).__init__(model, loss, metrics, reward_function,
                 optimizer, num_epochs, dataset,
                 batch_size, workers, device, log_frequency,
                 grad_clip, entropy_weight, skip_weight, baseline_decay,
                 ctrl_lr, ctrl_steps_aggregate, ctrl_kwargs)



    def init_dataloader(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001,
                                                        steps_per_epoch=len(training_data_loader),
                                                        epochs=epochs, div_factor=2,
                                                        pct_start=0.5)

        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.batch_size,shuffle=False)


        self.valid_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=16,shuffle=False)


    def _train_model(self, epoch):

        self.model.train()
        self.controller.eval()
        meters = AverageMeterGroup()
        # if epoch%10==0:
        #     self._resample()
        # self._resample()
        for step, x in enumerate(self.train_loader):
            [input, truth] = convert_to_4D_tensor(x)
            input=input.to(device)
            self.model.zero_grad()
            if step%3==0:
                self._resample()
            self.model.to(device)
            output = self.model(input / input_scale_factor)
            output_h = output.clone()
            output_h[:, 0, :, -1:] = 10
            output_h[:, 0, :, 0:1] = 10
            output_h[:, 0, -1:, :] = 10
            output_h[:, 0, :1, :] = 10
            kernel = 3
            sobel_filter = Filter((output_h.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
            continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            loss = criterion(continuity, continuity * 0)
            if epoch%1000==0:
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
        for ctrl_step, x in enumerate(self.valid_loader):

            [input, truth] = convert_to_4D_tensor(x)
            input = input.to(device)
            truth = truth.to(device)

            self._resample()
            output = self.model(input / input_scale_factor)
            output_h = output.clone()
            # Impose BC
            output_h[:, 0, :, -1:] = 10
            output_h[:, 0, :, 0:1] = 10
            output_h[:, 0, -1:, :] = 10
            output_h[:, 0, :1, :] = 10
            kernel = 3
            sobel_filter = Filter((output_h.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
            continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            loss = criterion(continuity, continuity * 0)
            lossval = torch.sqrt(criterion(truth, output_h) / criterion(truth, truth * 0)).item()
            reward = 1 / lossval
            if epoch%1000==0:
                print('val_reward: ', reward)
            metrics = {'res': loss.item()}
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

            if (ctrl_step + 1) % self.ctrl_steps_aggregate == 0:
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip)
                self.ctrl_optim.step()
                self.ctrl_optim.zero_grad()

    def fit(self):
        for i in range(self.num_epochs):
            self._train_model(i)
            if i%3==0:
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

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.model_optim, max_lr=0.001,
                                                             steps_per_epoch=len(training_data_loader),
                                                             epochs=epochs, div_factor=2,
                                                             pct_start=0.5)

        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.batch_size, shuffle=False)

        self.valid_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=16, shuffle=False)


    def _train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()
        for step, x in enumerate(self.train_loader):
            [input, truth] = convert_to_4D_tensor(x)
            trn_X = input.to(device)
            val_X, val_y = input.to(device), truth.to(device)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            # if self.unrolled:
            #     self._unrolled_backward(trn_X, _, val_X, val_y)
            self._backward(val_X, val_y)
            self.ctrl_optim.step()
            self.scheduler.step()

            # phase 2: child network step
            self.model_optim.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, 5)
            loss.backward()
            if epoch%1000==0:
                print('loss is:',loss)
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
            self.model_optim.step()

            metrics = {'res': loss.item()}
            metrics['loss'] = loss.item()
            meters.update(metrics)

    def _logits_and_loss(self, X, y):
        output = self.model(X / input_scale_factor)
        output_h=output.clone()
        output_h[:, 0, :, -1:] = 10
        output_h[:, 0, :, 0:1] = 10
        output_h[:, 0, -1:, :] = 10
        output_h[:, 0, :1, :] = 10
        kernel = 3
        sobel_filter = Filter((output_h.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
        continuity = pde_residue(X, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        loss = criterion(continuity, continuity * 0)
        return output_h, loss

    def _backward(self, val_X, val_y):
        """
        Simple backward with gradient descent
        """
        _, loss = self._logits_and_loss(val_X, val_y)
        loss.backward()




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

