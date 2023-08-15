import torch
import torch.optim as optim
import random
from utils.get_dataset import get_dataset
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from search_structure import UNet
from utils.hpo_utils import *
from darcy_oneshot import search

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
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    model=search()
    model = model.to(device)
    batch_size = 32
    lr = 0.001
    training_data_loader, valid_data_loader, test_data_loader = get_dataset(batch_size=batch_size)
    epochs = 300
    Res_list = []
    Error = []
    value = 10
    relative_l2_test = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(training_data_loader),
                                                    epochs=epochs, div_factor=2)

    for epoch in range(0, epochs):
        model.train()
        Res = 0
        relative_l2_valid = []
        for iteration, batch in enumerate(training_data_loader):
            input = batch
            input = input.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss_boundary = boundary_condition(output)
            sobel_filter = Filter(output.shape[2], correct=True, filter_size=3, device=device)
            loss_pde = loss_origin(input, output, sobel_filter)
            loss = loss_pde + loss_boundary * 10
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_lr()
            if iteration==1:
                print("lr:", current_lr)
            Res = Res + loss.item()

        model.eval()
        for iteration, batch in enumerate(valid_data_loader):
            (input, truth) = batch
            input, truth = input.to(device), truth.to(device)
            output = model(input)
            output1 = output[:, 0:1, :, :]
            truth1 = truth[:, 0:1, :, :]

            for i in range(batch_size):
                output1_each = output1[i, 0:1, :, :]
                truth1_each = truth1[i, 0:1, :, :]
                error_each = torch.sqrt(
                    criterion(truth1_each, output1_each) / criterion(truth1_each, truth1_each * 0)).item()
                relative_l2_valid.append(error_each)

        res = Res / len(training_data_loader)
        relative_l2 = float(np.mean(relative_l2_valid))
        Res_list.append(res)
        Error.append(relative_l2)
        print('Epoch is ', epoch)
        print("Res Loss is", (Res / len(training_data_loader)))
        print(" relative_l2 error is", (relative_l2))
        if epoch > 100:
            if relative_l2 <= value:
                value = relative_l2
                numepoch = epoch
                print('min epoch:', numepoch)
                torch.save(model.state_dict(), 'retrain_struct_oneshot.pth')
    min_o_error = Error[numepoch]
    print('valid error:', min_o_error)
    model.load_state_dict(torch.load('retrain_struct_oneshot.pth'))
    model.eval()
    for iteration, batch in enumerate(test_data_loader):
        (input, truth) = batch
        input, truth = input.to(device), truth.to(device)
        output = model(input)
        output1 = output[:, 0:1, :, :]
        truth1 = truth[:, 0:1, :, :]
        for i in range(batch_size):
            output1_each = output1[i, 0:1, :, :]
            truth1_each = truth1[i, 0:1, :, :]
            error_each = torch.sqrt(
                criterion(truth1_each, output1_each) / criterion(truth1_each, truth1_each * 0)).item()
            relative_l2_test.append(error_each)
    test_error = float(np.mean(relative_l2_test))
    print('test error:',test_error)