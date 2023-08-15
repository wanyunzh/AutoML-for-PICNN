from nni.retiarii import fixed_arch
import argparse
from utils.get_dataset import get_dataset
import nni
import random
from torch.optim.lr_scheduler import ExponentialLR
from utils.hpo_utils import *
from search_structure import UNet

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
    params = {
        'constraint': 2,
        'loss function': 2,
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
    for epoch in range(0, epochs):
        model.train()
        Res = 0
        mae_valid = []
        for iteration, batch in enumerate(training_data_loader):
            input, truth = batch
            input, truth = input.to(device), truth.to(device)
            optimizer.zero_grad()
            output = model(input)
            input = input * args.input_std + args.input_mean
            filter = Get_loss(params=params, device=device, nx=args.nx, length=args.length,
                              bcs=args.bc)
            laplace_frac, loss_b = filter(input, output)
            loss_fun = P_OHEM(loss_fun=F.l1_loss)
            loss_laplace = loss_fun(output - laplace_frac, torch.zeros_like(output - laplace_frac))
            loss = loss_laplace + loss_b
            loss.backward()
            optimizer.step()
            scheduler.step()
            Res = Res + loss.item()
        model.eval()
        for iteration, batch in enumerate(valid_data_loader):
            (input, truth) = batch
            input, truth = input.to(device), truth.to(device)
            output = model(input)
            output_k = output + 298
            val_mae = F.l1_loss(output_k, truth).detach().numpy()
            mae_valid.append(val_mae)
        res = Res / len(training_data_loader)
        mae = float(np.mean(mae_valid))
        Res_list.append(res)
        Error.append(mae)
        print('Epoch is ', epoch)
        print("Res Loss is", (Res / len(training_data_loader)))
        print(" mean absolute error is", (mae))
        if epoch > 0:
            if mae <= value:
                value = mae
                numepoch = epoch
                print('min epoch:', numepoch)
    min_o_error = Error[numepoch]
    print('valid error:', min_o_error)
    mae_test=[]
    for iteration, batch in enumerate(test_data_loader):
        (input, truth) = batch
        input, truth = input.to(device), truth.to(device)
        output = model(input)
        output_k = output + 298
        test_mae = F.l1_loss(output_k, truth).detach().numpy()
        mae_test.append(test_mae)

    final_mae = float(np.mean(mae_test))
    print("test mean absolute error is", final_mae)
