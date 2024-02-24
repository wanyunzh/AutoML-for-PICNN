import torch.optim as optim
from torch.utils.data import DataLoader
from hpo_utils import *
from get_dataset import get_dataset
import nni
import random
from loss_operation import *
device = torch.device(f"cuda:{3}" if torch.cuda.is_available() else "cpu")
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

def traintest_hb(model_cls):
    params = {
        'constraint': 1,
        'UNARY_OPS': 'identity',
        'WEIGHT_INIT': 'zero',
        'WEIGHT_OPS': 'adaptive',
        'gradient': 1,
        'kernel': 3,
    }
    device = torch.device(f"cuda:{3}" if torch.cuda.is_available() else "cpu")
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
    batchSize = 2
    Epochs = 1000
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
        Res = 0
        error = 0
        for iteration, batch in enumerate(training_data_loader):
            [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = convert_to_4D_tensor(batch)
            optimizer.zero_grad()
            output = model(Para)
            output = output[:, 0, :, :].reshape(output.shape[0], 1,
                                                output.shape[2],
                                                output.shape[3])
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

            output_temp = output.clone()
            output_h = output_temp
            loss_boundary = 0
            for j in range(batchSize):
                bc1 = output_h[j, 0, :, -1:] - 0
                bc2 = output_h[j, 0, :, 0:1] - Para[j, 0, 0, 0]
                loss_boundary = loss_boundary + 1 * UNARY_OPS[params['UNARY_OPS']](bc1).sum() + 1 * UNARY_OPS[
                    params['UNARY_OPS']](bc2).sum()
            loss_boundary = loss_boundary / (2 * batchSize * bc1.shape[0])

            for j in range(batchSize):
                # Impose BC
                output_h[j, 0, -1:, 1:-1] = (
                    output_h[j, 0, 1:2, 1:-1])
                output_h[j, 0, :1, 1:-1] = (
                    output_h[j, 0, -2:-1, 1:-1])
                output_h[j, 0, :, -1:] = 0
                output_h[j, 0, :, 0:1] = Para[j, 0, 0, 0]
            kernel = params['kernel']
            diff_filter = Filter((output_h.shape[2], output_h.shape[3]), filter_size=kernel, device=device)
            if params['constraint'] == 2:
                loss_boundary = 0
            if params['constraint'] == 0:
                continuity = pde_residue(output, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            else:
                continuity = pde_residue(output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)

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
            loss = loss_search + loss_boundary + 0.005 * (gradient_loss_search1 + gradient_loss_search2)
            loss.backward()

            optimizer.step()
            # scheduler.step()
            loss_mass = loss
            Res += loss_mass.item()
            error = error + torch.sqrt(criterion(truth, output_h) / criterion(truth, truth * 0))

        print('Epoch is ', epoch)
        print("Res Loss is", (Res / len(training_data_loader)))
        print(" relative_l2 error is", (error / len(training_data_loader)))
        res = Res / len(training_data_loader)
        relative_l2 = error / len(training_data_loader)
        Res_list.append(res)
        Error.append(relative_l2)
        if epoch > 500:
            if relative_l2 <= value:
                value = relative_l2
                numepoch = epoch
                torch.save(model.state_dict(), 'hb_search_struct.pth')
    # EV=EV.cpu().detach().numpy()
    error_final = Error[numepoch]
    print('train error', error_final)
    print('numepoch:',numepoch)

    test_set = get_dataset(mode='test')
    Error_test = []
    model.load_state_dict(torch.load('hb_search_struct.pth', map_location=torch.device('cpu')))
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
    # nni.report_intermediate_result(1/mean_error)
    nni.report_final_result( 1 / float(error_final))






