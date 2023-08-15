import torch
import torch.optim as optim
import random
from utils.get_dataset import get_dataset
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from search_structure import UNet
import nni
from utils.hpo_utils import *

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

if __name__ == '__main__':
    params = {
        'constraint': 0,
        'loss function': 0,
        'kernel': 3,
    }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    from nni.retiarii import fixed_arch
    with fixed_arch('darcy_ori.json'):
        model = UNet(in_channels=1, num_classes=3)
    model=model.to(device)
    batch_size=32
    lr=0.001
    training_data_loader, valid_data_loader, test_data_loader=get_dataset(batch_size=batch_size)
    epochs = 300
    Res_list = []
    Error = []
    value = 10
    id_list = []
    step = 0
    criterion =nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(training_data_loader),
                                                    epochs=epochs, div_factor=2)
    for epoch in range(0,epochs):
        model.train()
        Res = 0
        relative_l2_valid=[]
        for iteration, batch in enumerate(training_data_loader):
            input=batch
            input = input.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss_boundary = boundary_condition(output)
            output_h = output.clone()
            output_h[:, 0, :, 0]=1
            output_h[:, 0, :, -1] = 0
            output_h[:, 2, [0, -1], :] = 0
            #soft constraint
            if params['constraint']==0:
                kernel=params['kernel']
                sobel_filter = Filter(output.shape[2], correct=True, filter_size=kernel, device=device)
                if params['loss function']==0:
                    loss_pde = loss_origin(input, output, sobel_filter)
                    loss = loss_pde + loss_boundary * 10
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = flux_diff_kdim(input, output, sobel_filter) \
                                 + source_diff_kdim(output, sobel_filter)
                    loss = continuity.sum()
                    if epoch != 0 and epoch % 30 == 0:
                        step += 1
                        for j in range(batch_size):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(out)
                            max_res = out.max()
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % 4096
                                if remain == (batch_size* iteration+j)% 4096:
                                    add_res = out.view(-1)[id]
                                    loss = loss + add_res
                    # Code For RAR+residue
                    loss_pde = loss / (batch_size * step + batch_size * continuity.shape[2] * continuity.shape[3])
                    loss = loss_pde + loss_boundary * 10
                    loss.backward()

                if params['loss function'] == 2:
                    loss_pde=loss_gpinn(input, output, sobel_filter,epoch)
                    loss = loss_pde + loss_boundary * 10
                    loss.backward()

                if params['loss function'] == 3:
                    loss_pde=loss_ohem(input, output, sobel_filter,epoch)
                    loss = loss_pde + loss_boundary * 10
                    loss.backward()
            #soft+hard constraint
            if params['constraint'] == 1:
                kernel = params['kernel']
                sobel_filter = Filter(output.shape[2], correct=True, filter_size=kernel, device=device)
                if params['loss function']==0:
                    loss_pde = loss_origin(input, output_h, sobel_filter)
                    loss = loss_pde + loss_boundary * 10
                    loss.backward()

                if params['loss function'] == 1:
                    continuity = flux_diff_kdim(input, output_h, sobel_filter) \
                                 + source_diff_kdim(output_h, sobel_filter)
                    loss = continuity.sum()
                    if epoch != 0 and epoch % 30 == 0:
                        step += 1
                        for j in range(batch_size):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(out)
                            max_res = out.max()
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % 4096
                                if remain == (batch_size* iteration+j)% 4096:
                                    add_res = out.view(-1)[id]
                                    loss = loss + add_res
                    # Code For RAR+residue
                    loss_pde = loss / (batch_size * step + batch_size * continuity.shape[2] * continuity.shape[3])
                    loss = loss_pde + loss_boundary * 10
                    loss.backward()

                if params['loss function'] == 2:
                    loss_pde=loss_gpinn(input, output_h, sobel_filter,epoch)
                    loss = loss_pde + loss_boundary * 10
                    loss.backward()

                if params['loss function'] == 3:
                    loss_pde=loss_ohem(input, output_h, sobel_filter,epoch)
                    loss = loss_pde + loss_boundary * 10
                    loss.backward()
            # hard constraint
            if params['constraint'] == 2:
                kernel = params['kernel']
                sobel_filter = Filter(output.shape[2], correct=True, filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    loss_pde = loss_origin(input, output_h, sobel_filter)
                    loss = loss_pde
                    loss.backward()

                if params['loss function'] == 1:
                    continuity = flux_diff_kdim(input, output_h, sobel_filter) \
                                 + source_diff_kdim(output_h, sobel_filter)
                    loss = continuity.sum()
                    if epoch != 0 and epoch % 30 == 0:
                        step += 1
                        for j in range(batch_size):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(out)
                            max_res = out.max()
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % 4096
                                if remain == (batch_size * iteration + j) % 4096:
                                    add_res = out.view(-1)[id]
                                    loss = loss + add_res
                    # Code For RAR+residue
                    loss_pde = loss / (
                                batch_size * step + batch_size * continuity.shape[2] * continuity.shape[
                            3])
                    loss = loss_pde
                    loss.backward()

                if params['loss function'] == 2:
                    loss_pde = loss_gpinn(input, output_h, sobel_filter, epoch)
                    loss = loss_pde
                    loss.backward()

                if params['loss function'] == 3:
                    loss_pde = loss_ohem(input, output_h, sobel_filter, epoch)
                    loss = loss_pde
                    loss.backward()
            optimizer.step()
            scheduler.step()
            Res = Res + loss.item()
        model.eval()
        for iteration, batch in enumerate(valid_data_loader):
            (input,truth) = batch
            input, truth = input.to(device), truth.to(device)
            output = model(input)
            output1=output[:,0:1,:,:]
            truth1=truth[:,0:1,:,:]

            for i in range(batch_size):
                output1_each=output1[i,0:1,:,:]
                truth1_each = truth1[i, 0:1, :, :]
                error_each=torch.sqrt(criterion(truth1_each, output1_each) / criterion(truth1_each, truth1_each * 0)).item()
                relative_l2_valid.append(error_each)

        res = Res / len(training_data_loader)
        relative_l2 = float(np.mean(relative_l2_valid))
        Res_list.append(res)
        Error.append(relative_l2)
        print('Epoch is ', epoch)
        print("Res Loss is", (Res / len(training_data_loader)))
        print(" relative_l2 error is", (relative_l2))
        if epoch % 50 == 0:
            nni.report_intermediate_result(relative_l2)
        if epoch>100:
            if relative_l2<=value:
                value=relative_l2
                numepoch=epoch
                print('min epoch:',numepoch)
    min_o_error = Error[numepoch]
    print('valid error:',min_o_error)
    nni.report_final_result(min_o_error)



