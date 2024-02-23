import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
import argparse
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='This is hyperparameter for this PDE dataset')
parser.add_argument("--input_mean", default=0, type=float)
parser.add_argument("--input_std", default=10000, type=float)
parser.add_argument("--seed", type=int, default=34, help="seed")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--input_dim", default=200, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--nx", default=200,type=int)
parser.add_argument("--cuda", default=7,type=int)
parser.add_argument("--length", default=0.1,type=float)
parser.add_argument("--bc", default=[[0.0450, 0.0], [0.0550, 0.0]],type=list,help="Dirichlet boundaries", )
args, unknown_args = parser.parse_known_args()
val_mae_mean_list = []
training_loss_list=[]
class CustomCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # 获取当前 epoch 的训练损失
        current_training_loss = trainer.callback_metrics['loss']

        # 将训练损失的值添加到训练损失列表中
        training_loss_list.append(float(current_training_loss))

def main(args):
    params = {
        'constraint': 2,
        'loss function': 2,
        'kernel': 2,
    }
    device = torch.device(f"cuda:{2}" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = LitUNet(args, params)
    checkpoint_callback = ModelCheckpoint(dirpath='.', filename='model_{epoch}', save_top_k=1, monitor='val_mae_mean',
                                         mode='min')
    custom_callback = CustomCallback()

    # Configure the trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],  # Add your custom callback if you have one
        devices=[2],  # Specify the GPU devices here
        accelerator="gpu",  # Indicate that you are using GPUs
        precision=16 if args.use_16bit else 32,
        val_check_interval=args.val_check_interval,
        # resume_from_checkpoint=args.resume_from_checkpoint if hasattr(args, 'resume_from_checkpoint') else None,
        profiler=args.profiler if hasattr(args, 'profiler') else None,
        benchmark=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, custom_callback],
        devices=[2],  # Specify the GPU devices here
        accelerator="gpu",  # Indicate that you are using GPUs
        precision=32,
        val_check_interval=1,
        profiler="store_true",
        benchmark=True,
    )

    # Fit the model
    trainer.fit(model)
    best_model_path = checkpoint_callback.best_model_path
    print('best model path: ', best_model_path)
    best_val_mae = checkpoint_callback.best_model_score
    best_val_mae = float(best_val_mae)
    print('valid error:', best_val_mae)
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    model.to(device)

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
    main(args)
