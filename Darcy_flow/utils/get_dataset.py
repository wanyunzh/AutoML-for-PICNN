"""
Load args and model from a directory
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from argparse import Namespace
import h5py
import json


def load_data(dir, num, batch_size, target=False):
    with h5py.File(dir, 'r') as f:
        input_data = f['input'][:num]
        if target:
            truth_data = f['output'][:num]

    # data_tuple = (torch.FloatTensor(input_data), ) if only_input else (
    #         torch.FloatTensor(input_data), torch.FloatTensor(truth_data))
    # tmp=TensorDataset(*data_tuple)
    # data_loader = DataLoader(tmp,
    #     batch_size=batch_size, shuffle=False, drop_last=True)
    x = torch.tensor(input_data, dtype=torch.float32)
    if target is False:
        data_loader = DataLoader(x, batch_size=batch_size, shuffle=False,drop_last=True)
    else:
        y = torch.tensor(truth_data, dtype=torch.float32)
        data_loader = DataLoader(list(zip(x, y)), batch_size=batch_size, shuffle=False,drop_last=True)
    return data_loader


def get_dataset(batch_size=32):

    dir_train_val = './datasets/64x64/kle512_lhs10000_train.hdf5'
    test_dir = './datasets/64x64/kle512_lhs1000_val.hdf5'
    train_num, val_num,test_num = 4096, 128,512

    training_data_loader = load_data(dir_train_val, train_num, batch_size,
                                target=False)
    valid_data_loader = load_data(dir_train_val, val_num, batch_size,
                                target=True)
    test_data_loader = load_data(test_dir, test_num,
                                        64, target=True)
    return training_data_loader,valid_data_loader,test_data_loader
