# Reference： https://github.com/cics-nd/pde-surrogate/tree/master
import torch
import torch.optim as optim
import random
from utils.get_dataset import get_dataset
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from search_structure import UNet
import nni
from utils.hpo_utils import *
from loss_operation import *

# def boundary_condition(output):
#
#     bc_left = output[:, 0, :, 0]
#     bc_right=output[:, 0, :, -1]
#     top=torch.unsqueeze(output[:,2,0,:],dim=1)
#     down=torch.unsqueeze(output[:,2,-1,:],dim=1)
#     flux=torch.cat([top, down], dim=1)
#     loss_dir = criterion(bc_left,torch.ones_like(bc_left))+criterion(bc_right,bc_right*0)
#     loss_neu = criterion(flux,flux*0)
#     loss_b=loss_dir+loss_neu
#     return loss_b

if __name__ == '__main__':
    params = {
        'constraint': 0,
        'UNARY_OPS': 'square',
        'WEIGHT_INIT': 'one',
        'WEIGHT_OPS': 'max',
        'gradient': 0,
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
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(device)
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
            bc_left = output[:, 0, :, 0]
            bc_right = output[:, 0, :, -1]
            top = torch.unsqueeze(output[:, 2, 0, :], dim=1)
            down = torch.unsqueeze(output[:, 2, -1, :], dim=1)
            flux = torch.cat([top, down], dim=1)
            differ_bcl=bc_left-torch.ones_like(bc_left)
            differ_bcr=bc_right-bc_right * 0
            differ_flux= flux - flux * 0
            loss_dir = UNARY_OPS[params['UNARY_OPS']](differ_bcl).mean()+UNARY_OPS[params['UNARY_OPS']](differ_bcr).mean()
            loss_neu = UNARY_OPS[params['UNARY_OPS']](differ_flux).mean()
            loss_boundary = loss_dir + loss_neu
            output_h = output.clone()
            output_h[:, 0, :, 0]=1
            output_h[:, 0, :, -1] = 0
            output_h[:, 2, [0, -1], :] = 0
            if (epoch == 0) and (iteration) == 0:
                if params['WEIGHT_INIT'] == 'one':
                    init_weight = torch.ones_like(output)
                else:
                    init_weight = torch.zeros_like(output)
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

                WEIGHT_OPS3 = {
                    'normalize': P_OHEM3(init_weight),
                    'adaptive': Loss_Adaptive3(init_weight),
                    'max': Max3(init_weight),
                    'one': One(init_weight),
                }
            kernel = params['kernel']
            sobel_filter = Filter(output.shape[2], correct=True, filter_size=kernel, device=device)
            if params['constraint'] == 2:
                loss_boundary = 0
            if params['constraint'] == 0:
                if sobel_filter.filter_size == 2 or sobel_filter.filter_size == 4:
                    grad_h = sobel_filter.grad_h_2(output[:, [0]])
                    grad_v = sobel_filter.grad_v_2(output[:, [0]])
                if sobel_filter.filter_size == 3 or sobel_filter.filter_size == 5:
                    grad_h = sobel_filter.grad_h(output[:, [0]])
                    grad_v = sobel_filter.grad_v(output[:, [0]])
                flux1 = - input * grad_h
                flux2 = - input * grad_v
                if sobel_filter.filter_size == 2 or sobel_filter.filter_size == 4:
                    flux1_g = sobel_filter.grad_h_2(output[:, [1]])
                    flux2_g = sobel_filter.grad_v_2(output[:, [2]])
                if sobel_filter.filter_size == 3 or sobel_filter.filter_size == 5:
                    flux1_g = sobel_filter.grad_h(output[:, [1]])
                    flux2_g = sobel_filter.grad_v(output[:, [2]])
            else:
                if sobel_filter.filter_size == 2 or sobel_filter.filter_size == 4:
                    grad_h = sobel_filter.grad_h_2(output_h[:, [0]])
                    grad_v = sobel_filter.grad_v_2(output_h[:, [0]])
                if sobel_filter.filter_size == 3 or sobel_filter.filter_size == 5:
                    grad_h = sobel_filter.grad_h(output_h[:, [0]])
                    grad_v = sobel_filter.grad_v(output_h[:, [0]])
                flux1 = - input * grad_h
                flux2 = - input * grad_v
                if sobel_filter.filter_size == 2 or sobel_filter.filter_size == 4:
                    flux1_g = sobel_filter.grad_h_2(output_h[:, [1]])
                    flux2_g = sobel_filter.grad_v_2(output_h[:, [2]])
                if sobel_filter.filter_size == 3 or sobel_filter.filter_size == 5:
                    flux1_g = sobel_filter.grad_h(output_h[:, [1]])
                    flux2_g = sobel_filter.grad_v(output_h[:, [2]])

            difference1 = output_h[:, [1]] - flux1
            difference2 = output_h[:, [2]] - flux2
            sum_flux = flux1_g + flux2_g
            target = torch.zeros_like(sum_flux)
            difference3 = sum_flux - target
            post_difference1 = UNARY_OPS[params['UNARY_OPS']](difference1)
            post_difference2 = UNARY_OPS[params['UNARY_OPS']](difference2)
            post_difference3 = UNARY_OPS[params['UNARY_OPS']](difference3)
            weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_difference1, epoch,iteration)
            loss_search1 = torch.mean(torch.abs(post_difference1 * weight_search1))
            weight_search2 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_difference2, epoch,iteration)
            loss_search2 = torch.mean(torch.abs(post_difference2 * weight_search2))
            weight_search3 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_difference3, epoch,iteration)
            loss_search3 = torch.mean(torch.abs(post_difference3 * weight_search3))
            loss_search = loss_search1 + loss_search2 + loss_search3
            if params['gradient'] == 1:
                if epoch >= 49:
                    dr1dx, dr1dy = conv_gpinn(difference1, sobel_filter)
                    dr2dx, dr2dy = conv_gpinn(difference2, sobel_filter)
                    dr3dx, dr3dy = conv_gpinn(difference3, sobel_filter)
                    post_gradient1 = UNARY_OPS[params['UNARY_OPS']](dr1dx)
                    post_gradient2 = UNARY_OPS[params['UNARY_OPS']](dr1dy)
                    post_gradient3 = UNARY_OPS[params['UNARY_OPS']](dr2dx)
                    post_gradient4 = UNARY_OPS[params['UNARY_OPS']](dr2dy)
                    post_gradient5 = UNARY_OPS[params['UNARY_OPS']](dr3dx)
                    post_gradient6 = UNARY_OPS[params['UNARY_OPS']](dr3dy)

                    gradient_weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_gradient1, epoch,iteration)
                    gradient_weight_search2 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_gradient2, epoch,iteration)
                    gradient_weight_search3 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_gradient3, epoch,iteration)
                    gradient_weight_search4 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_gradient4, epoch,iteration)
                    gradient_weight_search5 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_gradient5, epoch,iteration)
                    gradient_weight_search6 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_gradient6, epoch,iteration)

                    gradient_loss_search1 = torch.mean(torch.abs(post_gradient1 * gradient_weight_search1))
                    gradient_loss_search2 = torch.mean(torch.abs(post_gradient2 * gradient_weight_search2))
                    gradient_loss_search3 = torch.mean(torch.abs(post_gradient3 * gradient_weight_search3))
                    gradient_loss_search4 = torch.mean(torch.abs(post_gradient4 * gradient_weight_search4))
                    gradient_loss_search5 = torch.mean(torch.abs(post_gradient5 * gradient_weight_search5))
                    gradient_loss_search6 = torch.mean(torch.abs(post_gradient6 * gradient_weight_search6))
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
            loss = loss_search + 10 * loss_boundary + 0.005 * (gradient_loss_search1 + gradient_loss_search2+gradient_loss_search3 + gradient_loss_search4+gradient_loss_search5 + gradient_loss_search6)
                    # loss = loss_search + loss_boundary * 10 + 0.05 * (dr1dx + dr1dy + dr2dx + dr2dy)
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



