import torch.optim as optim
from torch.utils.data import DataLoader
from get_dataset import get_dataset
from hpo_utils import *
import torch
import nni.retiarii.nn.pytorch as nn
from hb_oneshot import model5
import random

class HBCNN(nn.Module):
    def __init__(self, params, In, Out):
        super(HBCNN,self).__init__()
        """
        Extract basic information
        """
        self.params = params
        self.In = In
        self.Out = Out
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(self.In, params['channel1'], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(params['channel1'], params['channel2'], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(params['channel2'], params['channel2'], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(params['channel2'], params['channel3'], kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(params['channel3'], self.Out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

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




if __name__ == "__main__":
    params = {
        'constraint': 2,
        'loss function': 0,
        'kernel': 4,
    }
    import nni
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params1 = {
        'channel1': 16,
        'channel2': 32,
        'channel3': 16
    }
    import nni
    criterion=nn.MSELoss()
    In = 1
    Out = 1
    lr=0.001
    batchSize=2
    epochs = 1000
    train_set = get_dataset(mode='train')
    training_data_loader=DataLoader(train_set,batch_size=batchSize)
    train_result=[]
    model_origin = HBCNN(params1, In, Out)
    model_all = model5()
    model_all.append(model_origin)
    for step_m in range(6):
        Res_list = []
        Error = []
        value = 10
        id_list=[]
        step=0
        model_train=model_all[step_m]
        model=model_train.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(0, epochs + 1):
            Res = 0
            error = 0
            for iteration, batch in enumerate(training_data_loader):
                [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = convert_to_4D_tensor(batch)
                optimizer.zero_grad()
                output = model(Para)
                output = output[:, 0, :, :].reshape(output.shape[0], 1,
                                                         output.shape[2],
                                                         output.shape[3])
                output_temp = output.clone()
                output_h = output_temp
                loss_boundary=0
                for j in range(batchSize):
                    bc1 = output_h[j, 0, :, -1:] - 0
                    bc2 = output_h[j, 0, :, 0:1] - Para[j, 0, 0, 0]
                    loss_boundary = loss_boundary+1 * (bc1 ** 2).sum() + 1 * (bc2 ** 2).sum()
                loss_boundary = loss_boundary / (2 *batchSize* bc1.shape[0])

                for j in range(batchSize):
                    # Impose BC
                    output_h[j, 0, -1:, 1:-1] = (
                        output_h[j, 0, 1:2, 1:-1])
                    output_h[j, 0, :1, 1:-1] = (
                        output_h[j, 0, -2:-1, 1:-1])
                    output_h[j, 0, :, -1:] = 0
                    output_h[j, 0, :, 0:1] = Para[j, 0, 0, 0]

                if params['constraint'] == 0:
                    kernel = params['kernel']
                    diff_filter = Filter((output.shape[2], output_h.shape[3]),filter_size=kernel, device=device)
                    if params['loss function'] == 0:
                        continuity=pde_residue(output, diff_filter,dydeta, dydxi, dxdxi,dxdeta,Jinv)
                        loss = criterion(continuity, continuity * 0)+loss_boundary
                        loss.backward()
                    if params['loss function'] == 1:
                        continuity = pde_residue(output, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        loss=(continuity**2).sum()
                        if 200<epoch<1000 and epoch % 20 == 0:
                            step += 1
                            for j in range(batchSize):
                                out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                                index = torch.argmax(abs(out))
                                max_res = out.max()
                                id_list.append(index)
                                for num, id in enumerate(id_list):
                                    remain = (num) % batchSize
                                    if remain == (batchSize * iteration + j) % batchSize:
                                        add_res = out.view(-1)[id]
                                        loss = loss + add_res ** 2
                        # Code For RAR+residue
                        loss_pde = loss / (
                                    batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                                3])
                        loss = loss_pde + loss_boundary
                        loss.backward()
                    if params['loss function'] == 2:
                        continuity = pde_residue(output, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        drdx,drdy = pde_out(continuity, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        loss_g1 = criterion(drdx, drdx * 0)
                        loss_g2 = criterion(drdy, drdy * 0)
                        loss_res = criterion(continuity, continuity * 0)
                        loss_all = loss_res + 0.001 * loss_g1 + 0.001 * loss_g2
                        if epoch >= 200:
                            loss = loss_all
                        else:
                            loss = loss_res
                        loss = loss + loss_boundary
                        loss.backward()
                    if params['loss function'] == 3:
                        continuity = pde_residue(output, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        if epoch >= 200:
                            loss_fun = P_OHEM(loss_fun=F.l1_loss)
                            loss = loss_fun(continuity, continuity * 0)
                        else:
                            loss = criterion(continuity, continuity * 0)
                        loss = loss + loss_boundary
                        loss.backward()

                # soft+hard constraint
                if params['constraint'] == 1:
                    kernel = params['kernel']
                    diff_filter = Filter((output.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
                    if params['loss function'] == 0:
                        continuity = pde_residue(output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        loss = criterion(continuity, continuity * 0) + loss_boundary
                        loss.backward()
                    if params['loss function'] == 1:
                        continuity = pde_residue(output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        loss=(continuity**2).sum()
                        if 200<epoch<1000 and epoch % 20 == 0:
                            step += 1
                            for j in range(batchSize):
                                out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                                index = torch.argmax(abs(out))
                                max_res = out.max()
                                id_list.append(index)
                                for num, id in enumerate(id_list):
                                    remain = (num) % batchSize
                                    if remain == (batchSize * iteration + j) % batchSize:
                                        add_res = out.view(-1)[id]
                                        loss = loss + add_res**2
                        # Code For RAR+residue
                        loss_pde = loss / (
                                batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                            3])
                        loss = loss_pde + loss_boundary
                        loss.backward()
                    if params['loss function'] == 2:
                        continuity = pde_residue(output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        drdx, drdy = pde_out(continuity, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        loss_g1 = criterion(drdx, drdx * 0)
                        loss_g2 = criterion(drdy, drdy * 0)
                        loss_res = criterion(continuity, continuity * 0)
                        loss_all = loss_res + 0.001 * loss_g1 + 0.001 * loss_g2
                        if epoch >= 200:
                            loss = loss_all
                        else:
                            loss = loss_res
                        loss = loss + loss_boundary
                        loss.backward()
                    if params['loss function'] == 3:
                        continuity = pde_residue(output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        if epoch >= 200:
                            loss_fun = P_OHEM(loss_fun=F.l1_loss)
                            loss = loss_fun(continuity, continuity * 0)
                        else:
                            loss = criterion(continuity, continuity * 0)
                        loss = loss + loss_boundary
                        loss.backward()
                # hard constraint
                if params['constraint'] == 2:
                    kernel = params['kernel']
                    diff_filter = Filter((output_h.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
                    if params['loss function'] == 0:
                        continuity = pde_residue(output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        loss = criterion(continuity, continuity * 0)
                        loss.backward()
                    if params['loss function'] == 1:
                        continuity = pde_residue(output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        loss=(continuity**2).sum()
                        if 200<epoch<1000 and epoch % 20 == 0:
                            step += 1
                            for j in range(batchSize):
                                out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                                index = torch.argmax(abs(out))
                                max_res = out.max()
                                id_list.append(index)
                                for num, id in enumerate(id_list):
                                    remain = (num) % batchSize
                                    if remain == (batchSize * iteration + j) % batchSize:
                                        add_res = out.view(-1)[id]
                                        loss = loss + add_res**2
                        # Code For RAR+residue
                        loss_pde = loss / (
                                batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                            3])
                        loss = loss_pde
                        loss.backward()
                    if params['loss function'] == 2:
                        continuity = pde_residue(output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        drdx, drdy = pde_out(continuity, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        loss_g1 = criterion(drdx, drdx * 0)
                        loss_g2 = criterion(drdy, drdy * 0)
                        loss_res = criterion(continuity, continuity * 0)
                        loss_all = loss_res + 0.001 * loss_g1 + 0.001 * loss_g2
                        if epoch >= 200:
                            loss = loss_all
                        else:
                            loss = loss_res
                        loss = loss
                        loss.backward()
                    if params['loss function'] == 3:
                        continuity = pde_residue(output_h, filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        if epoch >= 200:
                            loss_fun = P_OHEM(loss_fun=F.l1_loss)
                            loss = loss_fun(continuity, continuity * 0)
                        else:
                            loss = criterion(continuity, continuity * 0)
                        loss = loss
                        loss.backward()

                optimizer.step()
                loss_mass = loss
                Res += loss_mass.item()
                error = error + torch.sqrt(criterion(truth, output_h) / criterion(truth, truth * 0))

                # if epoch % 1000 == 0 or epoch % nEpochs == 0:
                #     torch.save(model, str(epoch) + '.pth')
            print('Epoch is ', epoch)
            print("Res Loss is", (Res / len(training_data_loader)))
            print(" relative_l2 error is", (error / len(training_data_loader)))
            res = Res / len(training_data_loader)
            relative_l2 = error / len(training_data_loader)
            if epoch%50==0:
                nni.report_intermediate_result(float(relative_l2))
            Res_list.append(res)
            Error.append(relative_l2)
            if epoch > 500:
                if relative_l2 <= value:
                    value = relative_l2
                    numepoch = epoch
                    # print(numepoch)
                    torch.save(model.state_dict(), 'loss_search_'+str(step_m)+'.pth')
        # EV=EV.cpu().detach().numpy()
        error_final = Error[numepoch]
        # ev_final = EV[2]
        error_final = error_final.cpu().detach().numpy()
        error_final = float(error_final)
        print('training l2 error',error_final)
        train_result.append(error_final)
    print('training error:',train_result)
    report=np.mean(np.array(train_result))
    nni.report_final_result(float(report))






