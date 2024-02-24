import torch.optim as optim
from torch.utils.data import DataLoader
from get_dataset import get_dataset
from hpo_utils import *
import torch
import nni.retiarii.nn.pytorch as nn
from hb_oneshot import model5
import random
from loss_operation import *
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
        'UNARY_OPS': 1,
        'WEIGHT_INIT': 0,
         'WEIGHT_OPS': 3,
        'gradient':0,
        'kernel': 4,
    }

    import nni
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
    # params['WEIGHT_OPS'] = 2
    # params['UNARY_OPS'] = 0
    # params['WEIGHT_INIT'] = 0
    # params['gradient'] = 1
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
                if (epoch==0) and (iteration)==0:
                    # if params['WEIGHT_INIT']=='one':
                    #     init_weight=torch.ones_like(output)
                    # else:
                    #     init_weight = torch.zeros_like(output)
                    # WEIGHT_OPS = {
                    #     'normalize': P_OHEM(init_weight),
                    #     'adaptive': Loss_Adaptive(init_weight) ,
                    #     'max': Max(init_weight,epoch),
                    #     'one': One(init_weight),
                    # }
                    if params['WEIGHT_INIT']==1:
                        init_weight=torch.ones_like(output)
                    else:
                        init_weight = torch.zeros_like(output)
                    WEIGHT_OPS = {
                        0: P_OHEM(init_weight),
                        1: Loss_Adaptive(init_weight) ,
                        2: Max(init_weight,epoch),
                        3: One(init_weight),
                    }

                output_temp = output.clone()
                output_h = output_temp
                loss_boundary=0
                for j in range(batchSize):
                    bc1 = output_h[j, 0, :, -1:] - 0
                    bc2 = output_h[j, 0, :, 0:1] - Para[j, 0, 0, 0]
                    loss_boundary = loss_boundary+1 * UNARY_OPS[params['UNARY_OPS']](bc1).sum() + 1 * UNARY_OPS[params['UNARY_OPS']](bc2).sum()
                loss_boundary = loss_boundary / (2 *batchSize* bc1.shape[0])

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
                    loss_boundary=0
                if params['constraint'] == 0:
                    continuity = pde_residue(output, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                else:
                    continuity = pde_residue(output_h, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)

                difference = continuity - torch.zeros_like(continuity)


                post_difference=UNARY_OPS[params['UNARY_OPS']](difference)
                weight_search= WEIGHT_OPS[params['WEIGHT_OPS']](post_difference,epoch)
                loss_search= torch.mean(torch.abs(post_difference * weight_search))
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
                    gradient_loss_search1=0
                    gradient_loss_search2=0
                loss=loss_search+loss_boundary+0.005*(gradient_loss_search1 + gradient_loss_search2)
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






