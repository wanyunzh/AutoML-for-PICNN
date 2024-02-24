import torch
import torch.optim as optim
import random
from utils.get_dataset import get_dataset
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from search_structure import UNet
from utils.hpo_utils import *
from nni.retiarii import fixed_arch
from nni.retiarii.oneshot.pytorch import EnasTrainer,DartsTrainer,ProxylessTrainer,SinglePathTrainer
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup

def boundary_condition(output):
    bc_left = output[:, 0, :, 0]
    bc_right=output[:, 0, :, -1]
    top=torch.unsqueeze(output[:,2,0,:],dim=1)
    down=torch.unsqueeze(output[:,2,-1,:],dim=1)
    flux=torch.cat([top, down], dim=1)
    loss_dir = criterion(bc_left,torch.ones_like(bc_left))+criterion(bc_right,bc_right*0)
    loss_neu = criterion(flux,flux*0)
    loss_b=loss_dir+loss_neu
    return loss_b

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 0.001
training_data_loader, valid_data_loader, test_data_loader = get_dataset(batch_size=batch_size)
epochs = 300
criterion = nn.MSELoss()


class My_EnasTrainer(EnasTrainer):
    """
    ENAS trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    reward_function : callable
        Receives logits and ground truth label, return a tensor, which will be feeded to RL controller as reward.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    grad_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    entropy_weight : float
        Weight of sample entropy loss.
    skip_weight : float
        Weight of skip penalty loss.
    baseline_decay : float
        Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
    ctrl_lr : float
        Learning rate for RL controller.
    ctrl_steps_aggregate : int
        Number of steps that will be aggregated into one mini-batch for RL controller.
    ctrl_steps : int
        Number of mini-batches for each epoch of RL controller learning.
    ctrl_kwargs : dict
        Optional kwargs that will be passed to :class:`ReinforceController`.
    """

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
                                                        epochs=epochs, div_factor=2)
        self.train_loader = training_data_loader
        self.valid_loader = valid_data_loader

    def _train_model(self, epoch):

        self.model.train()
        self.controller.eval()
        meters = AverageMeterGroup()
        # if epoch%10==0:
        #     self._resample()
        # self._resample()
        for step, x in enumerate(self.train_loader):
            input = x
            input=input.to(device)
            self.model.zero_grad()
            if step%3==0:
                self._resample()
            self.model.to(device)
            output = self.model(input)
            loss_boundary = boundary_condition(output)
            sobel_filter = Filter(output.shape[2], correct=True, filter_size=3, device=device)
            loss_pde = loss_origin(input, output, sobel_filter)
            if epoch%100==0:
                print('loss pde:',loss_pde)
            loss = loss_pde + loss_boundary * 10
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
            input, target = x
            input = input.to(device)
            target = target.to(device)
            self._resample()
            output = self.model(input)
            loss_boundary = boundary_condition(output)
            sobel_filter = Filter(output.shape[2], correct=True, filter_size=3, device=device)
            loss_pde = loss_origin(input, output, sobel_filter)
            loss = loss_pde + loss_boundary * 10
            output1 = output[:, 0:1, :, :]
            truth1 = target[:, 0:1, :, :]
            valid_error = torch.sqrt(
                criterion(truth1, output1) / criterion(truth1, truth1 * 0)).item()
            reward = 1 / valid_error
            if epoch%100==0:
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
                                                             epochs=epochs, div_factor=2)
        self.train_loader = training_data_loader
        self.valid_loader = valid_data_loader


    def _train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()
        for step, ((trn_X), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X = trn_X.to(device)
            val_X, val_y = val_X.to(device), val_y.to(device)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            # if self.unrolled:
            #     self._unrolled_backward(trn_X, _, val_X, val_y)
            self._backward(val_X, val_y)
            self.ctrl_optim.step()
            # phase 2: child network step
            self.model_optim.zero_grad()
            logits, loss = self._logits_and_loss(trn_X)
            loss.backward()
            if epoch%100 ==0:
                print('loss is:',loss)
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
            self.model_optim.step()
            self.scheduler.step()
            metrics = {'res': loss.item()}
            metrics['loss'] = loss.item()
            meters.update(metrics)

    def _logits_and_loss(self, X):
        output = self.model(X)
        loss_boundary = boundary_condition(output)
        sobel_filter = Filter(output.shape[2], correct=True, filter_size=3, device=device)
        loss_pde = loss_origin(X, output, sobel_filter)
        loss = loss_pde + loss_boundary * 10
        return output, loss

    def _backward(self, val_X, val_y):
        """
        Simple backward with gradient descent
        """
        _, loss = self._logits_and_loss(val_X)
        loss.backward()



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

