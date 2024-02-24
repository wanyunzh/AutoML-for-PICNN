import torch
from torch.utils.data import DataLoader
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
batchSize = 2
lr = 0.001
Ns = 1
nu = 0.01
padSingleSide = 1
id_list = []
from hpo_utils import *
import logging
import torch
import torch.nn as nn
from nni.retiarii.oneshot.pytorch import EnasTrainer,DartsTrainer,ProxylessTrainer,SinglePathTrainer
_logger = logging.getLogger(__name__)

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

    def __init__(self, model, loss, metrics, reward_function,
                 optimizer, num_epochs, dataset,
                 batch_size=2, workers=4, device=None, log_frequency=None,
                 grad_clip=5., entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999,
                 ctrl_lr=0.00035, ctrl_steps_aggregate=1, ctrl_kwargs=None):
        super(My_EnasTrainer,self).__init__(model, loss, metrics, reward_function,
                 optimizer, num_epochs, dataset,
                 batch_size, workers, device, log_frequency,
                 grad_clip, entropy_weight, skip_weight, baseline_decay,
                 ctrl_lr, ctrl_steps_aggregate, ctrl_kwargs)



    def init_dataloader(self):

        self.train_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size)


        self.valid_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size)


    def _train_model(self, epoch):

        self.model.train()
        self.controller.eval()
        meters = AverageMeterGroup()
        for step, x in enumerate(self.train_loader):
            [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = convert_to_4D_tensor(x)
            Para.to(self.device)
            truth.to(self.device)
            self.optimizer.zero_grad()

            self._resample()

            self.model.to(self.device)
            output = self.model(Para)

            outputV = output[:, 0, :, :].reshape(output.shape[0], 1,
                                                output.shape[2],
                                                output.shape[3])
            sobel_filter = SobelFilter((outputV.shape[2], outputV.shape[3]), filter_size=2, device=device)
            loss_boundary = 0
            for j in range(batchSize):
                bc1 = outputV[j, 0, :, -padSingleSide:] - 0
                bc2 = outputV[j, 0, :, 0:padSingleSide] - Para[j, 0, 0, 0]
                loss_boundary = loss_boundary + 1 * (bc1 ** 2).sum() + 1 * (bc2 ** 2).sum()
            loss_boundary = loss_boundary / (2 * batchSize * bc1.shape[0])

            for j in range(batchSize):
                # Impose BC
                outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
                    outputV[j, 0, 1:2, padSingleSide:-padSingleSide])
                outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
                    outputV[j, 0, -2:-1, padSingleSide:-padSingleSide])
                outputV[j, 0, :, -padSingleSide:] = 0
                outputV[j, 0, :, 0:padSingleSide] = Para[j, 0, 0, 0]
            continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            loss = (continuity ** 2).sum()
            if 200 < epoch < 1000 and epoch % 20 == 0:
                step += 1
                for j in range(batchSize):
                    out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                    index = torch.argmax(abs(out))
                    max_res = out.max()
                    id_list.append(index)
                    for num, id in enumerate(id_list):
                        remain = (num) % batchSize
                        if remain == (batchSize * step + j) % batchSize:
                            add_res = out.view(-1)[id]
                            loss = loss + add_res ** 2
            # Code For RAR+residue
            loss_pde = loss / (
                    batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                3])
            loss = loss_pde + loss_boundary
            if epoch%100==0:
                print(loss)
            metrics = self.metrics(outputV, truth)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            metrics['loss'] = loss.item()
            meters.update(metrics)

            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.info('Model Epoch [%d/%d] Step [%d/%d]  %s', epoch + 1,
                             self.num_epochs, step + 1, len(self.train_loader), meters)

    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        meters = AverageMeterGroup()
        self.ctrl_optim.zero_grad()
        for ctrl_step, x in enumerate(self.valid_loader):
            [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = convert_to_4D_tensor(x)
            Para.to(self.device)
            truth.to(self.device)

            self._resample()
            with torch.no_grad():
                output = self.model(Para)

            outputV = output[:, 0, :, :].reshape(output.shape[0], 1,
                                                     output.shape[2],
                                                     output.shape[3])
            for j in range(batchSize):
                # Impose BC
                outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
                    outputV[j, 0, 1:2, padSingleSide:-padSingleSide])
                outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
                    outputV[j, 0, -2:-1, padSingleSide:-padSingleSide])
                outputV[j, 0, :, -padSingleSide:] = 0
                outputV[j, 0, :, 0:padSingleSide] = Para[j, 0, 0, 0]
            # loss = torch.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0))
            metrics = self.metrics(truth, outputV)
            reward = self.reward_function(truth, outputV)
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

            if self.log_frequency is not None and ctrl_step % self.log_frequency == 0:
                _logger.info('RL Epoch [%d/%d] Step [%d/%d]  %s', epoch + 1, self.num_epochs,
                             ctrl_step + 1, len(self.valid_loader), meters)


class My_DartsTrainer(DartsTrainer):
    def __init__(self, model, num_epochs, optimizer=None, loss=None, metrics=None,
                 dataset=None,grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=2, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False):
        super(My_DartsTrainer,self).__init__(model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip,
                 learning_rate, batch_size, workers,
                 device, log_frequency,
                 arc_learning_rate, unrolled)
    def _init_dataloader(self):

        self.model_optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.train_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size,shuffle=False)


        self.valid_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=16,shuffle=False)

    def _train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()

        for step, x in enumerate(self.train_loader):
            [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = convert_to_4D_tensor(x)
            trn_X = Para.to(device)
            val_X, val_y = Para.to(device), truth.to(device)
            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            self._backward(val_X, Jinv, dxdxi, dydxi, dxdeta, dydeta,step,epoch,val_y)
            self.ctrl_optim.step()
            # phase 2: child network step
            self.model_optim.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, Jinv, dxdxi, dydxi, dxdeta, dydeta,step,epoch,5)
            loss.backward()
            if epoch%500==0:
                print('loss is:',loss)
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
            self.model_optim.step()

            metrics = {'res': loss.item()}
            metrics['loss'] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.info('Epoch [%s/%s] Step [%s/%s]  %s', epoch + 1,
                             self.num_epochs, step + 1, len(self.train_loader), meters)

    def _logits_and_loss(self, X, Jinv, dxdxi, dydxi, dxdeta, dydeta,step,epoch,y):


        output = self.model(X)

        outputV = output[:, 0, :, :].reshape(output.shape[0], 1,
                                             output.shape[2],
                                             output.shape[3])
        sobel_filter = SobelFilter((outputV.shape[2], outputV.shape[3]), filter_size=2, device=device)
        loss_boundary = 0
        for j in range(batchSize):
            bc1 = outputV[j, 0, :, -padSingleSide:] - 0
            bc2 = outputV[j, 0, :, 0:padSingleSide] - X[j, 0, 0, 0]
            loss_boundary = loss_boundary + 1 * (bc1 ** 2).sum() + 1 * (bc2 ** 2).sum()
        loss_boundary = loss_boundary / (2 * batchSize * bc1.shape[0])

        for j in range(batchSize):
            # Impose BC
            outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
                outputV[j, 0, 1:2, padSingleSide:-padSingleSide])
            outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
                outputV[j, 0, -2:-1, padSingleSide:-padSingleSide])
            outputV[j, 0, :, -padSingleSide:] = 0
            outputV[j, 0, :, 0:padSingleSide] = X[j, 0, 0, 0]
        continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        loss = (continuity ** 2).sum()
        if 200 < epoch < 1000 and epoch % 20 == 0:
            step += 1
            for j in range(batchSize):
                out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                index = torch.argmax(abs(out))
                max_res = out.max()
                id_list.append(index)
                for num, id in enumerate(id_list):
                    remain = (num) % batchSize
                    if remain == (batchSize * step + j) % batchSize:
                        add_res = out.view(-1)[id]
                        loss = loss + add_res ** 2
        # Code For RAR+residue
        loss_pde = loss / (
                batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
            3])
        loss = loss_pde + loss_boundary

        return outputV, loss

    def _backward(self, val_X, Jinv, dxdxi, dydxi, dxdeta, dydeta,step,epoch,val_y):
        """
        Simple backward with gradient descent
        """
        _, loss = self._logits_and_loss(val_X, Jinv, dxdxi, dydxi, dxdeta, dydeta,step,epoch,val_y)
        loss.backward()

class My_Sampling(SinglePathTrainer):

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



