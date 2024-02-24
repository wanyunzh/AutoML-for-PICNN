import scipy.io as sio
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
dir='./data/complex_component/FDM/train/'
train_file='./data/train.txt'
valid_file='./data/val.txt'
test_file='./data/test.txt'

def get_sample_name(file):
    sample_name=[]
    with open(file,'r') as f:
        f=f.readlines()
        for line in f:
            dir_file=dir+line.strip()
            sample_name.append(dir_file)
    return sample_name

def get_data(file_type,args,batch_size):
    path_list=get_sample_name(file_type)
    input=[]
    truth=[]
    transform_input = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            torch.tensor([args.input_mean]),
            torch.tensor([args.input_std]),
        ),
    ])
    transform_truth = transforms.Compose([
        transforms.ToTensor(),
    ])

    for i in range(len(path_list)):
    # for i in range(30):
        path=path_list[i]
        mats = sio.loadmat(path)
        x_data = mats.get('F').astype(np.float32)
        y_data = mats.get('u')
        x = transform_input(x_data)
        y = transform_truth(y_data)
        input.append(x)
        truth.append(y)

    # input=torch.Tensor(input).unsqueeze(dim=1)
    # truth=torch.Tensor(truth).unsqueeze(dim=1)
    data_loader = DataLoader(list(zip(input, truth)), batch_size=batch_size, shuffle=False)

    return data_loader

def get_dataset(args):
    training_data_loader=get_data(train_file,args,batch_size=args.batch_size)
    valid_data_loader=get_data(valid_file,args,batch_size=16)
    test_data_loader=get_data(test_file,args,batch_size=1)
    return training_data_loader,valid_data_loader,test_data_loader

# training_data_loader,valid_data_loader,test_data_loader=get_dataset(args=)
