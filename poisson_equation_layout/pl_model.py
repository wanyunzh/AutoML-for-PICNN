import pytorch_lightning as pl
import torch
from nni.retiarii import fixed_arch
from torch.utils.data import DataLoader
from search_structure import UNet
from utils.get_dataset import get_data



class LitUNet(pl.LightningModule):
    def __init__(self, args, params):
        dir = './data/complex_component/FDM/train/'
        train_file = './data/train.txt'
        valid_file = './data/val.txt'
        test_file = './data/test.txt'
        self.train_dataset = get_data(train_file, self.args, batch_size=self.args.batch_size)
        self.val_dataset = get_data(valid_file, self.args, batch_size=16)
        self.test_dataset = get_data(test_file, self.args, batch_size=1)
        super().__init__()
        self.save_hyperparameters(args)
        with fixed_arch('layout_struct.json'):
            self.model = UNet(in_channels=1, num_classes=1)
        self.params = params
        self.filter = Get_loss(params=params, device=self.device, nx=self.args.nx, length=self.args.length, bcs=self.args.bc)

        # Dataset related


    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)

    def training_step(self, batch, batch_idx):
        input, truth = batch
        # Your training logic here
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input, truth = batch
        # Your validation logic here
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        input, truth = batch
        # Your test logic here
        return {'test_loss': test_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [scheduler]
