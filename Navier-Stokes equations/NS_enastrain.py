import torch
from torch.utils.data import DataLoader
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import random
batchSize = 2
lr = 0.001
Ns = 1
nu = 0.01
padSingleSide = 1
id_list = []
from hpo_utils import *
from loss_operation import *
import logging
import torch
import torch.nn as nn
from nni.retiarii.oneshot.pytorch import EnasTrainer,DartsTrainer,ProxylessTrainer,SinglePathTrainer
_logger = logging.getLogger(__name__)

def convert_to_4D_tensor(data_list,device):
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
                 batch_size=3, workers=4, device=None, log_frequency=None,
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
            [JJInv, coord, xi,
                 eta, J, Jinv,
                 dxdxi, dydxi,
                 dxdeta, dydeta,
                 Utrue, Vtrue, Ptrue] = convert_to_4D_tensor(x, self.device)
            coord.to(self.device)
            Utrue.to(self.device)
            Vtrue.to(self.device)
            Ptrue.to(self.device)
            self.optimizer.zero_grad()

            self._resample()

            self.model.to(self.device)
            output = self.model(coord)

            params = {
                'UNARY_OPS': 1,
                'WEIGHT_INIT': 0,
                'WEIGHT_OPS': 3,
                'gradient': 0,
                'kernel': 2,
            }
            nxOF = 50
            nyOF = 50
            nx = nxOF + 2
            ny = nyOF + 2
            padSingleSide = 1
            udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)

            output_pad = udfpad(output)


            if params['WEIGHT_INIT'] == 1:
                init_weight = torch.ones_like(output_pad)
            else:
                init_weight = torch.zeros_like(output_pad)
            WEIGHT_OPS1 = {
                0: P_OHEM1(init_weight),
                1: Loss_Adaptive1(init_weight),
                2: Max1(init_weight),
                3: One(init_weight),
            }

            WEIGHT_OPS2 = {
                0: P_OHEM2(init_weight),
                1: Loss_Adaptive2(init_weight),
                2: Max2(init_weight),
                3: One(init_weight),
            }

            WEIGHT_OPS3 = {
                0: P_OHEM3(init_weight),
                1: Loss_Adaptive3(init_weight),
                2: Max3(init_weight),
                3: One(init_weight),
            }

            outputU = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])
            outputV = output_pad[:, 1, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])
            outputP = output_pad[:, 2, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])

            for j in range(batchSize):
                outputU[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 0, -1, :].reshape(1,
                                                                                                           nx - 2 * padSingleSide)
                outputU[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0
                outputU[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
                outputU[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
                outputU[j, 0, 0, 0] = 0.5 * (outputU[j, 0, 0, 1] + outputU[j, 0, 1, 0])
                outputU[j, 0, 0, -1] = 0.5 * (outputU[j, 0, 0, -2] + outputU[j, 0, 1, -1])
                outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 1, -1, :].reshape(1,
                                                                                                           nx - 2 * padSingleSide)
                outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0.4
                outputV[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
                outputV[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
                outputV[j, 0, 0, 0] = 0.5 * (outputV[j, 0, 0, 1] + outputV[j, 0, 1, 0])
                outputV[j, 0, 0, -1] = 0.5 * (outputV[j, 0, 0, -2] + outputV[j, 0, 1, -1])
                outputP[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0
                outputP[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = output[j, 2, 0, :].reshape(1,
                                                                                                         nx - 2 * padSingleSide)
                outputP[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = output[j, 2, :, -1].reshape(
                    ny - 2 * padSingleSide, 1)
                outputP[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = output[j, 2, :, 0].reshape(
                    ny - 2 * padSingleSide, 1)
                outputP[j, 0, 0, 0] = 0.5 * (outputP[j, 0, 0, 1] + outputP[j, 0, 1, 0])
                outputP[j, 0, 0, -1] = 0.5 * (outputP[j, 0, 0, -2] + outputP[j, 0, 1, -1])
            kernel = params['kernel']
            diff_filter = Filter((outputV.shape[2], outputV.shape[3]), filter_size=kernel, device=self.device)

            dudx, dudy = pde_out(outputU, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            d2udx2, _ = pde_out(dudx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            _, d2udy2 = pde_out(dudy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            dvdx, dvdy = pde_out(outputV, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            d2vdx2, _ = pde_out(dvdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            _, d2vdy2 = pde_out(dvdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            dpdx, dpdy = pde_out(outputP, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            d2pdx2, _ = pde_out(dpdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            _, d2pdy2 = pde_out(dpdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            continuity, Xresidual, Yresidual = pde_residue(dudx, dudy, dvdx, dvdy, dpdx, dpdy, d2udx2, d2udy2,
                                                           d2vdx2, d2vdy2, outputU, outputV)

            difference1 = continuity - torch.zeros_like(continuity)
            difference2 = Xresidual - torch.zeros_like(Xresidual)
            difference3 = Yresidual - torch.zeros_like(Yresidual)

            post_difference1 = UNARY_OPS[params['UNARY_OPS']](difference1)
            post_difference2 = UNARY_OPS[params['UNARY_OPS']](difference2)
            post_difference3 = UNARY_OPS[params['UNARY_OPS']](difference3)
            weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_difference1, epoch)
            loss_search1 = torch.mean(torch.abs(post_difference1 * weight_search1))
            weight_search2 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_difference2, epoch)
            loss_search2 = torch.mean(torch.abs(post_difference2 * weight_search1))
            weight_search3 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_difference3, epoch)
            loss_search3 = torch.mean(torch.abs(post_difference3 * weight_search3))
            loss_search = loss_search1 + loss_search2 + loss_search3

            if params['gradient'] == 1:
                if epoch >= 2000:

                    dr1dx, dr1dy = pde_out(difference1, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    dr2dx, dr2dy = pde_out(difference2, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    dr3dx, dr3dy = pde_out(difference3, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    post_gradient1 = UNARY_OPS[params['UNARY_OPS']](dr1dx)
                    post_gradient2 = UNARY_OPS[params['UNARY_OPS']](dr1dy)
                    post_gradient3 = UNARY_OPS[params['UNARY_OPS']](dr2dx)
                    post_gradient4 = UNARY_OPS[params['UNARY_OPS']](dr2dy)
                    post_gradient5 = UNARY_OPS[params['UNARY_OPS']](dr3dx)
                    post_gradient6 = UNARY_OPS[params['UNARY_OPS']](dr3dy)

                    gradient_weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_gradient1, epoch)
                    gradient_weight_search2 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_gradient2, epoch)
                    gradient_weight_search3 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_gradient3, epoch)
                    gradient_weight_search4 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_gradient4, epoch)
                    gradient_weight_search5 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_gradient5, epoch)
                    gradient_weight_search6 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_gradient6, epoch)

                    gradient_loss_search1 = torch.mean(torch.abs(post_gradient1 * gradient_weight_search1))
                    gradient_loss_search2 = torch.mean(torch.abs(post_gradient2 * gradient_weight_search2))
                    gradient_loss_search3 = torch.mean(torch.abs(post_gradient3 * gradient_weight_search3))
                    gradient_loss_search4 = torch.mean(torch.abs(post_gradient4 * gradient_weight_search4))
                    gradient_loss_search5 = torch.mean(torch.abs(post_gradient5 * gradient_weight_search5))
                    gradient_loss_search6 = torch.mean(torch.abs(post_gradient6 * gradient_weight_search6))
                    if epoch % 200 == 0:
                        print(gradient_loss_search1)
                        print(gradient_loss_search2)
                        print(gradient_loss_search3)
                        print(gradient_loss_search4)
                        print(gradient_loss_search5)
                        print(gradient_loss_search6)

                else:
                    gradient_loss_search1 = 0
                    gradient_loss_search2 = 0
                    gradient_loss_search3 = 0
                    gradient_loss_search4 = 0
                    gradient_loss_search5 = 0
                    gradient_loss_search6 = 0
            else:
                gradient_loss_search1 = 0
                gradient_loss_search2 = 0
                gradient_loss_search3 = 0
                gradient_loss_search4 = 0
                gradient_loss_search5 = 0
                gradient_loss_search6 = 0
            loss = loss_search + 5 * 1e-5 * (
                    gradient_loss_search1 + gradient_loss_search2 + gradient_loss_search3 + gradient_loss_search4 + gradient_loss_search5 + gradient_loss_search6)

        if epoch%2000==0:
            print(loss)
        metrics = self.metrics(Ptrue,output)
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
        nxOF = 50
        nyOF = 50
        nx = nxOF + 2
        ny = nyOF + 2
        padSingleSide = 1
        udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)
        for ctrl_step, x in enumerate(self.valid_loader):
            [JJInv, coord, xi,
             eta, J, Jinv,
             dxdxi, dydxi,
             dxdeta, dydeta,
             Utrue, Vtrue, Ptrue] = convert_to_4D_tensor(x, self.device)
            coord.to(self.device)
            Utrue.to(self.device)
            Vtrue.to(self.device)
            Ptrue.to(self.device)


            self._resample()
            with torch.no_grad():
                output = self.model(coord)
            # loss = torch.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0))
            metrics = self.metrics(Ptrue, output)
            reward = self.reward_function(Ptrue, output)
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
                 learning_rate=2.5E-3, batch_size=3, workers=4,
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
        # print('epoch is: ',epoch)
        self.model.train()
        meters = AverageMeterGroup()

        for step, x in enumerate(self.train_loader):
            [JJInv, coord, xi,
            eta, J, Jinv,
            dxdxi, dydxi,
            dxdeta, dydeta,
            Utrue, Vtrue, Ptrue] = convert_to_4D_tensor(x,self.device)

            trn_X = coord.to(self.device)
            val_X, val_y_u, val_y_v,val_y_p= coord.to(self.device), Utrue.to(self.device),Vtrue.to(self.device), Ptrue.to(self.device)
            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            self._backward(val_X, Jinv, dxdxi, dydxi, dxdeta, dydeta,epoch,val_y_u, val_y_v,val_y_p,step,self.device)
            self.ctrl_optim.step()
            # phase 2: child network step
            self.model_optim.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, Jinv, dxdxi, dydxi, dxdeta, dydeta,epoch,val_y_u, val_y_v,val_y_p,step,self.device)
            loss.backward()
            if epoch%5000==0:
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

    def _logits_and_loss(self, X, Jinv, dxdxi, dydxi, dxdeta, dydeta,epoch,val_y_u, val_y_v,val_y_p,step,device):
        params = {
            'UNARY_OPS': 1,
            'WEIGHT_INIT': 0,
            'WEIGHT_OPS': 3,
            'gradient': 0,
            'kernel': 2,
        }
        nxOF = 50
        nyOF = 50
        nx = nxOF + 2
        ny = nyOF + 2
        padSingleSide = 1
        udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)

        output = self.model(X)

        output_pad = udfpad(output)
        if True:

            if params['WEIGHT_INIT'] == 1:
                init_weight = torch.ones_like(output_pad)
            else:
                init_weight = torch.zeros_like(output_pad)
            WEIGHT_OPS1 = {
                0: P_OHEM1(init_weight),
                1: Loss_Adaptive1(init_weight),
                2: Max1(init_weight),
                3: One(init_weight),
            }

            WEIGHT_OPS2 = {
                0: P_OHEM2(init_weight),
                1: Loss_Adaptive2(init_weight),
                2: Max2(init_weight),
                3: One(init_weight),
            }

            WEIGHT_OPS3 = {
                0: P_OHEM3(init_weight),
                1: Loss_Adaptive3(init_weight),
                2: Max3(init_weight),
                3: One(init_weight),
            }

        outputU = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
                                                 output_pad.shape[2],
                                                 output_pad.shape[3])
        outputV = output_pad[:, 1, :, :].reshape(output_pad.shape[0], 1,
                                                 output_pad.shape[2],
                                                 output_pad.shape[3])
        outputP = output_pad[:, 2, :, :].reshape(output_pad.shape[0], 1,
                                                 output_pad.shape[2],
                                                 output_pad.shape[3])

        for j in range(batchSize):
            outputU[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 0, -1, :].reshape(1,
                                                                                                       nx - 2 * padSingleSide)
            outputU[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0
            outputU[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
            outputU[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
            outputU[j, 0, 0, 0] = 0.5 * (outputU[j, 0, 0, 1] + outputU[j, 0, 1, 0])
            outputU[j, 0, 0, -1] = 0.5 * (outputU[j, 0, 0, -2] + outputU[j, 0, 1, -1])
            outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 1, -1, :].reshape(1,
                                                                                                       nx - 2 * padSingleSide)
            outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0.4
            outputV[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
            outputV[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
            outputV[j, 0, 0, 0] = 0.5 * (outputV[j, 0, 0, 1] + outputV[j, 0, 1, 0])
            outputV[j, 0, 0, -1] = 0.5 * (outputV[j, 0, 0, -2] + outputV[j, 0, 1, -1])
            outputP[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0
            outputP[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = output[j, 2, 0, :].reshape(1,
                                                                                                     nx - 2 * padSingleSide)
            outputP[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = output[j, 2, :, -1].reshape(
                ny - 2 * padSingleSide, 1)
            outputP[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = output[j, 2, :, 0].reshape(
                ny - 2 * padSingleSide, 1)
            outputP[j, 0, 0, 0] = 0.5 * (outputP[j, 0, 0, 1] + outputP[j, 0, 1, 0])
            outputP[j, 0, 0, -1] = 0.5 * (outputP[j, 0, 0, -2] + outputP[j, 0, 1, -1])
        kernel = params['kernel']
        diff_filter = Filter((outputV.shape[2], outputV.shape[3]), filter_size=kernel, device=device)

        dudx, dudy = pde_out(outputU, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        d2udx2, _ = pde_out(dudx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        _, d2udy2 = pde_out(dudy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        dvdx, dvdy = pde_out(outputV, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        d2vdx2, _ = pde_out(dvdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        _, d2vdy2 = pde_out(dvdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        dpdx, dpdy = pde_out(outputP, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        d2pdx2, _ = pde_out(dpdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        _, d2pdy2 = pde_out(dpdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
        continuity, Xresidual, Yresidual = pde_residue(dudx, dudy, dvdx, dvdy, dpdx, dpdy, d2udx2, d2udy2,
                                                       d2vdx2, d2vdy2, outputU, outputV)

        difference1 = continuity - torch.zeros_like(continuity)
        difference2 = Xresidual - torch.zeros_like(Xresidual)
        difference3 = Yresidual - torch.zeros_like(Yresidual)

        post_difference1 = UNARY_OPS[params['UNARY_OPS']](difference1)
        post_difference2 = UNARY_OPS[params['UNARY_OPS']](difference2)
        post_difference3 = UNARY_OPS[params['UNARY_OPS']](difference3)
        weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_difference1, epoch)
        loss_search1 = torch.mean(torch.abs(post_difference1 * weight_search1))
        weight_search2 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_difference2, epoch)
        loss_search2 = torch.mean(torch.abs(post_difference2 * weight_search1))
        weight_search3 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_difference3, epoch)
        loss_search3 = torch.mean(torch.abs(post_difference3 * weight_search3))
        loss_search = loss_search1 + loss_search2 + loss_search3

        if params['gradient'] == 1:
            if epoch >= 2000:

                dr1dx, dr1dy = pde_out(difference1, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                dr2dx, dr2dy = pde_out(difference2, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                dr3dx, dr3dy = pde_out(difference3, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                post_gradient1 = UNARY_OPS[params['UNARY_OPS']](dr1dx)
                post_gradient2 = UNARY_OPS[params['UNARY_OPS']](dr1dy)
                post_gradient3 = UNARY_OPS[params['UNARY_OPS']](dr2dx)
                post_gradient4 = UNARY_OPS[params['UNARY_OPS']](dr2dy)
                post_gradient5 = UNARY_OPS[params['UNARY_OPS']](dr3dx)
                post_gradient6 = UNARY_OPS[params['UNARY_OPS']](dr3dy)

                gradient_weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_gradient1, epoch)
                gradient_weight_search2 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_gradient2, epoch)
                gradient_weight_search3 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_gradient3, epoch)
                gradient_weight_search4 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_gradient4, epoch)
                gradient_weight_search5 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_gradient5, epoch)
                gradient_weight_search6 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_gradient6, epoch)

                gradient_loss_search1 = torch.mean(torch.abs(post_gradient1 * gradient_weight_search1))
                gradient_loss_search2 = torch.mean(torch.abs(post_gradient2 * gradient_weight_search2))
                gradient_loss_search3 = torch.mean(torch.abs(post_gradient3 * gradient_weight_search3))
                gradient_loss_search4 = torch.mean(torch.abs(post_gradient4 * gradient_weight_search4))
                gradient_loss_search5 = torch.mean(torch.abs(post_gradient5 * gradient_weight_search5))
                gradient_loss_search6 = torch.mean(torch.abs(post_gradient6 * gradient_weight_search6))
                if epoch % 200 == 0:
                    print(gradient_loss_search1)
                    print(gradient_loss_search2)
                    print(gradient_loss_search3)
                    print(gradient_loss_search4)
                    print(gradient_loss_search5)
                    print(gradient_loss_search6)

            else:
                gradient_loss_search1 = 0
                gradient_loss_search2 = 0
                gradient_loss_search3 = 0
                gradient_loss_search4 = 0
                gradient_loss_search5 = 0
                gradient_loss_search6 = 0
        else:
            gradient_loss_search1 = 0
            gradient_loss_search2 = 0
            gradient_loss_search3 = 0
            gradient_loss_search4 = 0
            gradient_loss_search5 = 0
            gradient_loss_search6 = 0
        loss = loss_search + 5 * 1e-5 * (
                gradient_loss_search1 + gradient_loss_search2 + gradient_loss_search3 + gradient_loss_search4 + gradient_loss_search5 + gradient_loss_search6)

        return outputV, loss

    def _backward(self, val_X, Jinv, dxdxi, dydxi, dxdeta, dydeta,epoch,val_y_u, val_y_v,val_y_p,step,device):
        """
        Simple backward with gradient descent
        """
        _, loss = self._logits_and_loss(val_X, Jinv, dxdxi, dydxi, dxdeta, dydeta,epoch,val_y_u, val_y_v,val_y_p,step,device)
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



