import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class TrackStateMLP(pl.LightningModule):

    def __init__(
        self,
        input_dim=10,
        state_dim=3,
        hidden_dim=64,
        num_layers=3,
        lr=1e-3,
        dropout=0.1
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.state_dim = state_dim

        # backbone
        layers = []

        in_dim = input_dim

        for _ in range(num_layers):

            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # state prediction
        self.state_head = nn.Linear(hidden_dim, state_dim)

        # loss
        self.criterion = nn.MSELoss()

    def forward(self, x_cont):
        h = self.backbone(x_cont)
        state = self.state_head(h)
        return state

    def _compute_loss(self, pred_state, target_state):
        return self.criterion(pred_state, target_state)

        return loss.mean()

    def training_step(self, batch, batch_idx):
        x_cont, target_state = batch
        pred_state = self(x_cont)
        loss = self._compute_loss(pred_state, target_state)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_cont, target_state = batch
        pred_state = self(x_cont)
        loss = self._compute_loss(pred_state, target_state)
        self.log("val_loss",loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class LossTracker(Callback):
    """
    PyTorch Lightning callback to track training and validation losses.
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Record training loss at the end of each epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer instance.
        pl_module : pl.LightningModule
            The model being trained.
        """
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Record validation loss at the end of each epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer instance.
        pl_module : pl.LightningModule
            The model being validated.
        """
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())