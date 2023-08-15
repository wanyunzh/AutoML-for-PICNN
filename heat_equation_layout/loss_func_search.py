import argparse
from utils.get_dataset import get_dataset
from nni.retiarii import fixed_arch
from search_structure import UNet
import random
from torch.optim.lr_scheduler import ExponentialLR
from utils.hpo_utils import *

parser = argparse.ArgumentParser(description='This is hyperparameter for this PDE dataset')
parser.add_argument("--input_mean", default=0, type=float)
parser.add_argument("--input_std", default=10000, type=float)
parser.add_argument("--seed", type=int, default=34, help="seed")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--input_dim", default=200, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--nx", default=200,type=int)
parser.add_argument("--cuda", default=4,type=int)
parser.add_argument("--length", default=0.1,type=float)
parser.add_argument("--bc", default=[[0.0450, 0.0], [0.0550, 0.0]],type=list,help="Dirichlet boundaries", )
args, unknown_args = parser.parse_known_args()


if __name__ == '__main__':

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    params = {
        'constraint': 2,
        'loss function': 2,
        'kernel': 2,
    }
    import nni
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)

    training_data_loader, valid_data_loader, test_data_loader = get_dataset(args=args)
    with fixed_arch('layout_ori.json'):
        model=UNet(num_classes=1, in_channels=1)
    step = 0
    id_list=[]
    epochs = 30
    Res_list = []
    Error = []
    value = 100000
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.85)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    for epoch in range(0,epochs):
        model.train()
        Res = 0
        mae_valid=[]
        for iteration, batch in enumerate(training_data_loader):
            input, truth = batch
            input, truth = input.to(device), truth.to(device)
            optimizer.zero_grad()
            output = model(input)
            input= input * args.input_std + args.input_mean
            filter = Get_loss(params=params, device=device,nx=args.nx, length=args.length,
                                       bcs=args.bc)

            if params['kernel'] == 2 or params['kernel'] == 4:
                with torch.no_grad():
                    # 算出来的是T(xi, yj )撇
                    laplace_frac, loss_b = filter(input, output)

                if params['constraint'] == 2:
                    loss_b = 0
                else:
                    pass
                if params['loss function'] == 0:
                    loss_laplace = torch.mean(
                        torch.abs((output - laplace_frac) - torch.zeros_like(output - laplace_frac)))
                    print('loss residue:', loss_laplace)
                    loss = loss_laplace + loss_b
                if params['loss function'] == 1:
                    continuity = torch.abs(output - laplace_frac)
                    loss_ori = torch.sum(torch.abs(continuity))
                    if epoch != 0 and epoch % 3 == 0:
                        step += 1
                        for j in range(args.batch_size):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(out)
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % 8000
                                if remain == (args.batch_size * iteration + j) % 8000:
                                    add_res = out.view(-1)[id]
                                    loss_ori = loss_ori + torch.abs(add_res)
                    # Code For RAR+residue
                    loss = loss_ori / (
                            args.batch_size * step + args.batch_size * continuity.shape[
                        2] * continuity.shape[3]) + loss_b

                if params['loss function'] == 2:
                    loss_fun = P_OHEM(loss_fun=F.l1_loss)
                    loss_laplace = loss_fun(output - laplace_frac, torch.zeros_like(output - laplace_frac))
                    loss = loss_laplace + loss_b

            else:
                # 算出来的是T(xi, yj )撇
                continuity, loss_b = filter(input, output)
                if params['constraint'] == 2:
                    loss_b = 0
                else:
                    pass
                if params['loss function'] == 0:
                    loss_laplace = torch.mean(torch.abs((continuity) - torch.zeros_like(continuity)))
                    print('loss residue', loss_laplace)
                    loss = loss_laplace + loss_b
                if params['loss function'] == 1:
                    continuity = torch.abs(continuity)
                    loss_ori = torch.sum(torch.abs(continuity))
                    if epoch != 0 and epoch % 3 == 0:
                        step += 1
                        for j in range(args.batch_size):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(out)
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % 8000
                                if remain == (args.batch_size * iteration + j) % 8000:
                                    add_res = out.view(-1)[id]
                                    loss_ori = loss_ori + torch.abs(add_res)
                    # Code For RAR+residue
                    loss = loss_ori / (
                            args.batch_size * step + args.batch_size * continuity.shape[
                        2] * continuity.shape[3]) + loss_b

                if params['loss function'] == 2:
                    loss_fun = P_OHEM(loss_fun=F.l1_loss)
                    loss_laplace = loss_fun(continuity, torch.zeros_like(continuity))
                    loss = loss_laplace + loss_b
            loss.backward()
            optimizer.step()
            scheduler.step()
            Res = Res + loss.item()
        model.eval()
        for iteration, batch in enumerate(valid_data_loader):
            (input,truth) = batch
            input, truth = input.to(device), truth.to(device)
            output = model(input)
            output_k=output+298
            val_mae = F.l1_loss(output_k, truth).detach().numpy()
            mae_valid.append(val_mae)
        res = Res / len(training_data_loader)
        mae = float(np.mean(mae_valid))
        Res_list.append(res)
        Error.append(mae)
        print('Epoch is ', epoch)
        print("Res Loss is", (Res / len(training_data_loader)))
        print(" mean absolute error is", (mae))
        if epoch>0:
            if mae<=value:
                value=mae
                numepoch=epoch
                print('min epoch:',numepoch)
    min_o_error = Error[numepoch]
    print('valid error:',min_o_error)
    nni.report_final_result(min_o_error)



