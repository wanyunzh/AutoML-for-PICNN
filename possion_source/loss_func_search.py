import torch.optim as optim
from torch.utils.data import DataLoader
from get_dataset import get_dataset
from hpo_utils import *
import torch
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import fixed_arch
import random
from search_struct import UNet
from loss_operation import *
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
        'constraint': 0,
        'UNARY_OPS': 'square',
        'WEIGHT_INIT': 'zero',
        'WEIGHT_OPS': 'one',
        'gradient': 0,
        'kernel': 3,
    }

    import nni
    device = torch.device(f"cuda:{2}" if torch.cuda.is_available() else "cpu")
    optimized_params = nni.get_next_parameter()

    params.update(optimized_params)
    print(params)
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import nni
    criterion=nn.MSELoss()
    lr=0.001
    batchSize=32
    train_size = 256
    all_set,dydeta, dydxi, dxdxi, dxdeta, Jinv = get_dataset(device)
    train_set=all_set[:train_size]
    training_data_loader=DataLoader(train_set,batch_size=batchSize)
    with fixed_arch('possion_ori.json'):
        model = UNet(num_classes=1, in_channels=1)
        print('model is:', model)
    model = model.to(device)
    epochs = 15000
    scalefactor = 1
    input_scale_factor = 500
    Res_list = []
    Error = []
    value = 10000
    id_list=[]
    step=0


    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(training_data_loader),
                                                    epochs=epochs, div_factor=2,
                                                    pct_start=0.25)


    for epoch in range(0, epochs):
        Res = 0
        error = 0
        for iteration, batch in enumerate(training_data_loader):
            [input, truth] = convert_to_4D_tensor(batch)
            optimizer.zero_grad()
            output = model(input / input_scale_factor)
            output_h = output.clone()
            if (epoch == 0) and (iteration) == 0:
                if params['WEIGHT_INIT'] == 'one':
                    init_weight = torch.ones_like(output)
                else:
                    init_weight = torch.zeros_like(output)
                WEIGHT_OPS = {
                    'normalize': P_OHEM(init_weight),
                    # 'adaptive': Loss_Adaptive(init_weight),
                    'max': Max(init_weight),
                    'one': One(init_weight),
                }
            bc1 = output_h[:, 0, :, -1:] - 10
            bc2 = output_h[:, 0, :, 0:1] - 10
            bc3 = output_h[:, 0, -1:, :][:, :, 1:-1] - 10
            bc4 = output_h[:, 0, :1, :][:, :, 1:-1] - 10
            loss_b=UNARY_OPS[params['UNARY_OPS']](bc1).mean()+UNARY_OPS[params['UNARY_OPS']](bc2).mean()+UNARY_OPS[params['UNARY_OPS']](bc3).mean()+UNARY_OPS[params['UNARY_OPS']](bc4).mean()
            # loss_b=criterion(bc1,bc1*0)+criterion(bc2,bc2*0)+criterion(bc3,bc3*0)+criterion(bc4,bc4*0)

            output_h[:, 0, :, -1:] = 10
            output_h[:, 0, :, 0:1] = 10
            output_h[:, 0, -1:, :] = 10
            output_h[:, 0, :1, :] = 10

            kernel = params['kernel']
            diff_filter = Filter((output_h.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
            if params['constraint'] == 2:
                loss_b = 0
            if params['constraint'] == 0:
                continuity = pde_residue(input,output, diff_filter,dydeta, dydxi, dxdxi,dxdeta,Jinv)
            else:
                continuity = pde_residue(input,output_h, diff_filter,dydeta, dydxi, dxdxi,dxdeta,Jinv)

            difference = continuity - torch.zeros_like(continuity)
            post_difference = UNARY_OPS[params['UNARY_OPS']](difference)
            weight_search = WEIGHT_OPS[params['WEIGHT_OPS']](post_difference, epoch,iteration)
            loss_search = torch.mean(torch.abs(post_difference * weight_search))
            # if params['gradient'] == 1:
            #     if epoch >= 500:
            #         drdx, drdy = pde_out(difference, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            #         post_gradient1 = UNARY_OPS[params['UNARY_OPS']](drdx)
            #         post_gradient2 = UNARY_OPS[params['UNARY_OPS']](drdy)
            #         gradient_weight_search1 = WEIGHT_OPS[params['WEIGHT_OPS']](post_gradient1, epoch)
            #         gradient_weight_search2 = WEIGHT_OPS[params['WEIGHT_OPS']](post_gradient2, epoch)
            #         gradient_loss_search1 = torch.mean(torch.abs(post_gradient1 * gradient_weight_search1))
            #         gradient_loss_search2 = torch.mean(torch.abs(post_gradient2 * gradient_weight_search2))
            #     else:
            #         gradient_loss_search1 = 0
            #         gradient_loss_search2 = 0
            # else:
            #     gradient_loss_search1 = 0
            #     gradient_loss_search2 = 0
            # loss = loss_search + loss_b + 0.005 * (gradient_loss_search1 + gradient_loss_search2)
            loss = loss_search + loss_b
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_lr()
            loss_mass = loss
            Res += loss_mass.item()
            error = error + torch.sqrt(criterion(truth, output_h) / criterion(truth, truth * 0)).item()

            # if epoch % 1000 == 0 or epoch % nEpochs == 0:
            #     torch.save(model, str(epoch) + '.pth')
        print('Epoch is ', epoch)
        print("Res Loss is", (Res / len(training_data_loader)))
        print(" relative_l2 error is", (error / len(training_data_loader)))
        res = Res / len(training_data_loader)
        relative_l2 = error / len(training_data_loader)
        if epoch%500==0:
            nni.report_intermediate_result(float(relative_l2))
        Res_list.append(res)
        Error.append(relative_l2)
        if epoch > 1500:
            if relative_l2 <= value:
                value = relative_l2
                numepoch = epoch
                torch.save(model.state_dict(), 'loss_search.pth')
    # EV=EV.cpu().detach().numpy()
    error_final = Error[numepoch]
    # ev_final = EV[2]
    # error_final = error_final.cpu().detach().numpy()
    report = float(error_final)
    print('training l2 error',report)
    nni.report_final_result(float(report))






