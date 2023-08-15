import torch.optim as optim
from torch.utils.data import DataLoader
from search_struct import HBCNN_5
from get_dataset import get_dataset
from hpo_utils import *
from nni.retiarii import fixed_arch
import nni
import random

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
        'constraint': 1,
        'loss function': 1,
        'kernel': 2,
    }

    device = torch.device("cpu")
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    criterion = nn.MSELoss()
    In = 1
    Out = 1
    lr = 0.001
    batchSize = 2
    params1 = {
        'channel1': 16,
        'channel2': 32,
        'channel3': 16
    }

    # model = model_cls().to(device)
    with fixed_arch('HB_cnn.json'):
        model = HBCNN_5(params1,In, Out)
        print('final model:', model)
    model = model.to(device)
    torch.save(model.state_dict(), 'model_init.pth')
    Epochs = 1500
    train_set = get_dataset(mode='train')
    training_data_loader = DataLoader(train_set, batch_size=batchSize)
    all_result = []
    Res_list = []
    Error = []
    value = 10
    id_list = []
    step = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(training_data_loader),
                                                    epochs=Epochs, div_factor=2,
                                                    pct_start=0.5)

    for epoch in range(0, Epochs):
        Res=0
        error=0

        for iteration, batch in enumerate(training_data_loader):
            [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = convert_to_4D_tensor(batch)
            optimizer.zero_grad()
            output = model(Para)

            output = output[:, 0, :, :].reshape(output.shape[0], 1,
                                                output.shape[2],
                                                output.shape[3])
            output_tmp = output.clone()
            output_h = output_tmp
            loss_boundary = 0
            for j in range(batchSize):
                bc1 = output_h[j, 0, :, -1:] - 0
                bc2 = output_h[j, 0, :, 0:1] - Para[j, 0, 0, 0]
                loss_boundary = loss_boundary + 1 * (bc1 ** 2).sum() + 1 * (bc2 ** 2).sum()
            loss_boundary = loss_boundary / (2 * batchSize * bc1.shape[0])

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
                sobel_filter = Filter((output.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    continuity = pde_residue(output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = criterion(continuity, continuity * 0) + loss_boundary
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = pde_residue(output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = (continuity ** 2).sum()
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
                    continuity = pde_residue(output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    drdx, drdy = pde_out(continuity, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
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
                    continuity = pde_residue(output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
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
                sobel_filter = Filter((output.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    continuity = pde_residue(output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = criterion(continuity, continuity * 0) + loss_boundary
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = pde_residue(output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = (continuity ** 2).sum()
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
                    continuity = pde_residue(output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    drdx, drdy = pde_out(continuity, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss_g1 = criterion(drdx, drdx * 0)
                    loss_g2 = criterion(drdy, drdy * 0)
                    loss_res = criterion(continuity, continuity * 0)
                    loss_all = loss_res + 0.001* loss_g1 + 0.001 * loss_g2
                    if epoch >= 200:
                        loss = loss_all
                    else:
                        loss = loss_res
                    loss = loss + loss_boundary
                    loss.backward()
                if params['loss function'] == 3:
                    continuity = pde_residue(output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
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
                sobel_filter = Filter((output_h.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    continuity = pde_residue(output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = criterion(continuity, continuity * 0)
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = pde_residue(output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = (continuity ** 2).sum()
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
                    loss = loss_pde
                    loss.backward()
                if params['loss function'] == 2:
                    continuity = pde_residue(output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    drdx, drdy = pde_out(continuity, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
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
                    continuity = pde_residue(output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    if epoch >= 200:
                        loss_fun = P_OHEM(loss_fun=F.l1_loss)
                        loss = loss_fun(continuity, continuity * 0)
                    else:
                        loss = criterion(continuity, continuity * 0)
                    loss = loss
                    loss.backward()

            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_lr()
            if iteration == 0:
                print("lr:", current_lr)
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
        if epoch % 200 == 0:
            nni.report_intermediate_result(float(relative_l2))
        Res_list.append(res)
        Error.append(relative_l2)
        if epoch > 500:
            if relative_l2 <= value:
                value = relative_l2
                numepoch = epoch
                # print(numepoch)
                torch.save(model.state_dict(), 'retrain.pth')
    # EV=EV.cpu().detach().numpy()
    error_final = Error[numepoch]
    print('numepoch:', numepoch)
    print('train error', error_final)
    test_set = get_dataset(mode='test')
    Error_test = []
    model.load_state_dict(torch.load('retrain.pth', map_location=torch.device('cpu')))
    for i in range(len(test_set)):
        [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = convert_to_4D_tensor(test_set[i])
        Para = Para.reshape(1, 1, Para.shape[0], Para.shape[1])
        truth = truth.reshape(1, 1, truth.shape[0], truth.shape[1])
        coord = coord.reshape(1, 2, coord.shape[2], coord.shape[3])
        model.eval()
        # with torch.no_grad():
        output = model(Para)
        output = output[:, 0, :, :].reshape(output.shape[0], 1,
                                            output.shape[2],
                                            output.shape[3])
        output_tmp = output.clone()
        output_h = output_tmp
        # Impose BC
        output_h[0, 0, -1:, 1:-1] = (
            output_h[0, 0, 1:2, 1:-1])  # up outlet bc zero gradient
        output_h[0, 0, :1, 1:-1] = (
            output_h[0, 0, -2:-1, 1:-1])  # down inlet bc
        output_h[0, 0, :, -1:] = 0  # right wall bc
        output_h[0, 0, :, 0:1] = Para[0, 0, 0, 0]  # left  wall bc

        Error_test.append(torch.sqrt(criterion(truth, output_h) / criterion(truth, truth * 0)))
    Error_test = np.asarray([i.cpu().detach().numpy() for i in Error_test])
    print(Error_test)
    mean_error = float(np.mean(Error_test))
    print('test error:', mean_error)











