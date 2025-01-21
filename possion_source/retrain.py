import torch.optim as optim
from torch.utils.data import DataLoader
from search_struct import UNet
from get_dataset import get_dataset
from hpo_utils import *
from nni.retiarii import fixed_arch
import nni
import random
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
        'constraint': 2,
        'UNARY_OPS': 'square',
        'WEIGHT_INIT': 'zero',
        'WEIGHT_OPS': 'one',
        'gradient': 0,
        'kernel': 3,
    }
    device = torch.device(f"cuda:{4}" if torch.cuda.is_available() else "cpu")
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    with fixed_arch('possion_model.json'):
        model = UNet(num_classes=1, in_channels=1)
        print('final model:', model)
    model = model.to(device)
    # torch.save(model.state_dict(), 'model_init.pth')
    criterion = nn.MSELoss()
    lr = 0.001
    batchSize = 32
    train_size = 256
    all_set,dydeta, dydxi, dxdxi, dxdeta, Jinv = get_dataset(device)
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
            if (epoch == 0) and (iteration) == 0:
                if params['WEIGHT_INIT'] == 'one':
                    init_weight = torch.ones_like(output)
                else:
                    init_weight = torch.zeros_like(output)
                WEIGHT_OPS = {
                    'normalize': P_OHEM(init_weight),
                    'adaptive': Loss_Adaptive(init_weight),
                    'max': Max(init_weight, epoch),
                    'one': One(init_weight),
                }
            bc1 = output_h[:, 0, :, -1:] - 10
            bc2 = output_h[:, 0, :, 0:1] - 10
            bc3 = output_h[:, 0, -1:, :][:, :, 1:-1] - 10
            bc4 = output_h[:, 0, :1, :][:, :, 1:-1] - 10
            loss_b = UNARY_OPS[params['UNARY_OPS']](bc1).mean() + UNARY_OPS[params['UNARY_OPS']](bc2).mean() + \
                     UNARY_OPS[params['UNARY_OPS']](bc3).mean() + UNARY_OPS[params['UNARY_OPS']](bc4).mean()
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
                continuity = pde_residue(input, output, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            else:
                continuity = pde_residue(input, output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)

            difference = continuity - torch.zeros_like(continuity)
            post_difference = UNARY_OPS[params['UNARY_OPS']](difference)
            weight_search = WEIGHT_OPS[params['WEIGHT_OPS']](post_difference, epoch)
            loss_search = torch.mean(torch.abs(post_difference * weight_search))
            if params['gradient'] == 1:
                if epoch >= 500:
                    drdx, drdy = pde_out(difference, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    post_gradient1 = UNARY_OPS[params['UNARY_OPS']](drdx)
                    post_gradient2 = UNARY_OPS[params['UNARY_OPS']](drdy)
                    gradient_weight_search1 = WEIGHT_OPS[params['WEIGHT_OPS']](post_gradient1, epoch)
                    gradient_weight_search2 = WEIGHT_OPS[params['WEIGHT_OPS']](post_gradient2, epoch)
                    gradient_loss_search1 = torch.mean(torch.abs(post_gradient1 * gradient_weight_search1))
                    gradient_loss_search2 = torch.mean(torch.abs(post_gradient2 * gradient_weight_search2))
                else:
                    gradient_loss_search1 = 0
                    gradient_loss_search2 = 0
            else:
                gradient_loss_search1 = 0
                gradient_loss_search2 = 0
            loss = loss_search + loss_b + 0.005 * (gradient_loss_search1 + gradient_loss_search2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_lr()
            loss_mass = loss
            Res += loss_mass.item()
            error = error + torch.sqrt(criterion(truth, output_h) / criterion(truth, truth * 0)).item()
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
                torch.save(model.state_dict(), 'retrain_possion.pth')
    error_final = Error[numepoch]
    print('train error', error_final)
    print('numepoch:', numepoch)
    test_set = all_set[train_size:]
    Error_test = []
    model.load_state_dict(torch.load('retrain_possion.pth', map_location=torch.device('cpu')))
    for i in range(len(test_set)):
        [input, truth] = convert_to_4D_tensor(test_set[i])
        input = input.reshape(1, 1, input.shape[0], input.shape[1])
        truth = truth.reshape(1, 1, truth.shape[0], truth.shape[1])
        model.eval()
        output = model(input / input_scale_factor)
        output_h = output.clone()
        # Impose BC
        output_h[:, 0, :, -1:] = 10
        output_h[:, 0, :, 0:1] = 10
        output_h[:, 0, -1:, :] = 10
        output_h[:, 0, :1, :] = 10
        Error_test.append(torch.sqrt(criterion(truth, output_h) / criterion(truth, truth * 0)))
    Error_test = np.asarray([i.cpu().detach().numpy() for i in Error_test])
    mean_error = float(np.mean(Error_test))
    print('test error:', mean_error)

    import matplotlib.pyplot as plt

    fig1 = plt.figure(1)
    plt.plot(Res_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.yscale('log')

    # 保存训练损失曲线图
    fig1.savefig('retrain_loss_curve.tif', dpi=550)
    fig2 = plt.figure(2)

    # 绘制预测误差曲线
    plt.plot(Error, label='Prediction Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Prediction Error of Validation set')
    plt.legend()
    plt.yscale('log')
    # 保存预测误差曲线图
    fig2.savefig('retrain_error_curve.tif', dpi=550)
    print('drawing done')










