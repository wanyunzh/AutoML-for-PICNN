from nni.retiarii import fixed_arch
import argparse
from utils.get_dataset import get_dataset
# import nni
import random
from torch.optim.lr_scheduler import ExponentialLR
from utils.hpo_utils import *
from search_structure import UNet
from loss_operation import *
parser = argparse.ArgumentParser(description='This is hyperparameter for this PDE dataset')
parser.add_argument("--input_mean", default=0, type=float)
parser.add_argument("--input_std", default=10000, type=float)
parser.add_argument("--seed", type=int, default=34, help="seed")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--input_dim", default=200, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--nx", default=200,type=int)
parser.add_argument("--cuda", default=0,type=int)
parser.add_argument("--length", default=0.1,type=float)
parser.add_argument("--bc", default=[[0.0450, 0.0], [0.0550, 0.0]],type=list,help="Dirichlet boundaries", )
args, unknown_args = parser.parse_known_args()

if __name__ == '__main__':
    params = {
        'constraint': 2,
        'UNARY_OPS': 0,
        'WEIGHT_INIT': 0,
        'WEIGHT_OPS': 0,
        'kernel': 2,
    }
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    training_data_loader, valid_data_loader, test_data_loader = get_dataset(args=args)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    with fixed_arch('layout_struct.json'):
        model = UNet(num_classes=1, in_channels=1)
    model = model.to(device)
    epochs = 30
    Res_list = []
    Error = []
    value = 100000
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.85)
    filter = Get_loss(params=params, device=device, nx=args.nx, length=args.length,
                      bcs=args.bc)
    for epoch in range(0, epochs):
        model.train()
        Res = 0
        mae_valid = []
        for iteration, batch in enumerate(training_data_loader):
            input, truth = batch
            input_tmp = input.cpu().numpy()
            truth_batch = truth.cpu().numpy()
            input, truth = input.to(device), truth.to(device)
            optimizer.zero_grad()
            output = model(input)
            if (epoch == 0) and (iteration) == 0:
                if params['WEIGHT_INIT'] == 1:
                    init_weight = torch.ones_like(output)
                else:
                    init_weight = torch.zeros_like(output)
                WEIGHT_OPS = {
                    0: P_OHEM(init_weight),
                    # 'adaptive': Loss_Adaptive(init_weight),
                    1: Max(init_weight),
                    2: One(init_weight),
                }
            output_tmp = output.cpu().detach().numpy()
            input = input * args.input_std + args.input_mean
            # with torch.no_grad():
            #     continuity, loss_b = filter(input, output)

            if params['kernel'] == 2 or params['kernel'] == 4:
                with torch.no_grad():
                    continuity, loss_b = filter(input, output)
                difference = output - continuity
            else:
                continuity, loss_b = filter(input, output)
                difference = continuity

            if params['constraint'] == 2:
                loss_b = 0

            # difference = continuity - torch.zeros_like(continuity)
            post_difference = UNARY_OPS[params['UNARY_OPS']](difference)
            weight_search = WEIGHT_OPS[params['WEIGHT_OPS']](post_difference, epoch)
            loss_search = torch.mean(torch.abs(post_difference * weight_search))
            # if params['gradient']==1:
            #     if epoch>=1:
            #         drdx, drdy= get_gradient(continuity, device,nx=args.nx, length=args.length)
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
            # loss = loss_search + loss_b + 1e-6 * (gradient_loss_search1 + gradient_loss_search2)
            loss = loss_search + loss_b
            loss.backward()
            optimizer.step()
            Res = Res + loss.item()
        scheduler.step()
        model.eval()
        for iteration, batch in enumerate(valid_data_loader):
            (input, truth) = batch
            input, truth = input.to(device), truth.to(device)
            output = model(input)
            output_k = output + 298
            input = input * args.input_std + args.input_mean
            laplace_frac, loss_b = filter(input, output)
            frac, _ = filter(input, output)
            loss_jacobi = F.l1_loss(
                output, frac
            )
            val_mae = F.l1_loss(output_k, truth).detach().cpu().numpy()
            mae_valid.append(val_mae)
        res = Res / len(training_data_loader)
        mae = float(np.mean(mae_valid))
        Res_list.append(float(res))
        Error.append(float(mae))
        print('Epoch is ', epoch)
        print("Res Loss is", (Res / len(training_data_loader)))
        print(" mean absolute error is", (mae))
        if epoch > 0:
            if mae <= value:
                value = mae
                numepoch = epoch
                print('min epoch:', numepoch)
                torch.save(model.state_dict(), 'retrain_layout.pth')
    min_o_error = Error[numepoch]
    print('valid error:', min_o_error)
    mae_test=[]
    model.load_state_dict(torch.load('retrain_layout.pth', map_location=torch.device('cpu')))
    for iteration, batch in enumerate(test_data_loader):
        (input, truth) = batch
        input, truth = input.to(device), truth.to(device)
        output = model(input)
        output_k = output + 298
        test_mae = F.l1_loss(output_k, truth).cpu().detach().numpy()
        mae_test.append(test_mae)

    final_mae = float(np.mean(mae_test))
    print("test mean absolute error is", final_mae)

    import matplotlib.pyplot as plt

    fig1 = plt.figure(1)
    plt.plot(Res_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    # plt.yscale('log')
    plt.tight_layout()
    # 保存训练损失曲线图
    fig1.savefig('training_loss_curve.tif', dpi=550)
    fig2 = plt.figure(2)
    # 绘制预测误差曲线
    plt.plot(Error, label='Prediction Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Prediction Error of Validation set')
    plt.legend()
    plt.yscale('log')
    # 保存预测误差曲线图
    fig2.savefig('prediction_error_curve.tif', dpi=550)
    print('drawing done')
