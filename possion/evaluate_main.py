import torch.optim as optim
from torch.utils.data import DataLoader
from hpo_utils import *
from get_dataset import get_dataset
import nni
import random
device = torch.device("cpu")
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

def traintest_pos(model_cls):
    params = {
        'constraint': 2,
        'loss function': 0,
        'kernel': 3,
    }
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = model_cls().to(device)
    torch.save(model.state_dict(), 'model_init0.pth')
    criterion = nn.MSELoss()
    lr = 0.001
    batchSize = 32
    train_size = 256
    all_set,dydeta, dydxi, dxdxi, dxdeta, Jinv = get_dataset()
    train_set = all_set[:train_size]
    training_data_loader = DataLoader(train_set, batch_size=batchSize)
    epochs = 15000
    scalefactor = 1
    input_scale_factor = 500
    Res_list = []
    Error = []
    value = 10
    id_list = []
    step = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(training_data_loader),
                                                    epochs=epochs, div_factor=2,
                                                    pct_start=0.5)

    for epoch in range(0, epochs):
        Res = 0
        error = 0
        for iteration, batch in enumerate(training_data_loader):
            [input, truth] = convert_to_4D_tensor(batch)
            optimizer.zero_grad()
            output = model(input / input_scale_factor)
            output_h = output.clone()
            bc1 = output_h[:, 0, :, -1:] - 10
            bc2 = output_h[:, 0, :, 0:1] - 10
            bc3 = output_h[:, 0, -1:, :][:, :, 1:-1] - 10
            bc4 = output_h[:, 0, :1, :][:, :, 1:-1] - 10
            loss_b=criterion(bc1,bc1*0)+criterion(bc2,bc2*0)+criterion(bc3,bc3*0)+criterion(bc4,bc4*0)

            output_h[:, 0, :, -1:] = 10
            output_h[:, 0, :, 0:1] = 10
            output_h[:, 0, -1:, :] = 10
            output_h[:, 0, :1, :] = 10

            if params['constraint'] == 0:
                kernel = params['kernel']
                sobel_filter = Filter((output.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    continuity = pde_residue(input, output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = criterion(continuity, continuity * 0) + loss_b
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = pde_residue(input, output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = (continuity ** 2).sum()
                    if 0 < epoch < 10000 and epoch % 150 == 0:
                        step += 1
                        for j in range(batchSize):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(abs(out))
                            max_res = out.max()
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % 256
                                if remain == (batchSize * iteration + j) % 256:
                                    add_res = out.view(-1)[id]
                                    loss = loss + add_res ** 2
                    # Code For RAR+residue
                    loss_pde = loss / (
                            batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                        3])
                    loss = loss_pde + loss_b
                    loss.backward()
                if params['loss function'] == 2:
                    continuity = pde_residue(input, output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    drdx, drdy = pde_out(continuity, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss_g1 = criterion(drdx, drdx * 0)
                    loss_g2 = criterion(drdy, drdy * 0)
                    loss_res = criterion(continuity, continuity * 0)
                    loss_all = loss_res + 0.001 * loss_g1 + 0.001 * loss_g2
                    if epoch >= 1000:
                        loss = loss_all
                    else:
                        loss = loss_res
                    loss = loss + loss_b
                    loss.backward()
                if params['loss function'] == 3:
                    continuity = pde_residue(input, output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    if epoch >= 1000:
                        loss_fun = P_OHEM(loss_fun=F.l1_loss)
                        loss = loss_fun(continuity, continuity * 0)
                    else:
                        loss = criterion(continuity, continuity * 0)
                    loss = loss + loss_b
                    loss.backward()

            # soft+hard constraint
            if params['constraint'] == 1:
                kernel = params['kernel']
                sobel_filter = Filter((output.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = criterion(continuity, continuity * 0) + loss_b
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = (continuity ** 2).sum()
                    if 0 < epoch < 10000 and epoch % 150 == 0:
                        step += 1
                        for j in range(batchSize):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(abs(out))
                            max_res = out.max()
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % 256
                                if remain == (batchSize * iteration + j) % 256:
                                    add_res = out.view(-1)[id]
                                    loss = loss + add_res ** 2
                    # Code For RAR+residue
                    loss_pde = loss / (
                            batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                        3])
                    loss = loss_pde + loss_b
                    loss.backward()
                if params['loss function'] == 2:
                    continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    drdx, drdy = pde_out(continuity, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss_g1 = criterion(drdx, drdx * 0)
                    loss_g2 = criterion(drdy, drdy * 0)
                    loss_res = criterion(continuity, continuity * 0)
                    loss_all = loss_res + 0.001 * loss_g1 + 0.001 * loss_g2
                    if epoch >= 1000:
                        loss = loss_all
                    else:
                        loss = loss_res
                    loss = loss + loss_b
                    loss.backward()
                if params['loss function'] == 3:
                    continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    if epoch >= 1000:
                        loss_fun = P_OHEM(loss_fun=F.l1_loss)
                        loss = loss_fun(continuity, continuity * 0)
                    else:
                        loss = criterion(continuity, continuity * 0)
                    loss = loss + loss_b
                    loss.backward()
            # hard constraint
            if params['constraint'] == 2:
                kernel = params['kernel']
                sobel_filter = Filter((output_h.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = criterion(continuity, continuity * 0)
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = (continuity ** 2).sum()
                    if 0 < epoch < 10000 and epoch % 150 == 0:
                        step += 1
                        for j in range(batchSize):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(abs(out))
                            max_res = out.max()
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % 256
                                if remain == (batchSize * iteration + j) % 256:
                                    add_res = out.view(-1)[id]
                                    loss = loss + add_res ** 2
                    # Code For RAR+residue
                    loss_pde = loss / (
                            batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                        3])
                    loss = loss_pde
                    loss.backward()
                if params['loss function'] == 2:
                    continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    drdx, drdy = pde_out(continuity, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss_g1 = criterion(drdx, drdx * 0)
                    loss_g2 = criterion(drdy, drdy * 0)
                    loss_res = criterion(continuity, continuity * 0)
                    loss_all = loss_res + 0.001 * loss_g1 + 0.001 * loss_g2
                    if epoch >= 1000:
                        loss = loss_all
                    else:
                        loss = loss_res
                    loss = loss
                    loss.backward()
                if params['loss function'] == 3:
                    continuity = pde_residue(input, output_h, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    if epoch >= 1000:
                        loss_fun = P_OHEM(loss_fun=F.l1_loss)
                        loss = loss_fun(continuity, continuity * 0)
                    else:
                        loss = criterion(continuity, continuity * 0)
                    loss = loss
                    loss.backward()

            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_lr()
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
        Res_list.append(float(res))
        Error.append(float(relative_l2))
        if epoch > 4000:
            if relative_l2 <= value:
                value = relative_l2
                numepoch = epoch
                torch.save(model.state_dict(), 'possion_struct.pth')
    error_final = Error[numepoch]
    print('train error', error_final)
    print('numepoch:',numepoch)
    nni.report_final_result( 1 / float(error_final))





