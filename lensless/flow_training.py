
from __future__ import annotations

import piq
from torch.nn.functional import mse_loss
from torch.optim import Adam
from pytorch_lightning import LightningModule

vgg_loss = piq.LPIPS()


class FlowTrainingSystem(LightningModule):
    """
    Lightning wrapper for the physics-guided flow model.
    """

    def __init__(
        self,
        model,
        region_of_interest,
        lr: float = 5e-4,
        sample_steps: int = 20,
    ):
        super().__init__()
        self.model = model
        self.region_of_interest = region_of_interest
        self.lr = lr
        self.sample_steps = sample_steps

    def configure_optimizers(self):
        return {"optimizer": Adam(self.parameters(), lr=self.lr)}

    def forward(self, inputs, **kwargs):
        return self.model.reconstruct(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "Train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "Val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "Test")

    def step(self, batch, batch_idx, label):
        y, x_gt = batch
        stats = self.model.cfm_loss(y, x_gt)
        loss = stats["loss"]

        self.log(f"Loss/{label}", loss)
        self.log(f"VelocityMSE/{label}", stats["v_mse"])
        self.log(f"FreeVelocityMSE/{label}", stats["u_mse"])
        self.log(f"PhysicsGradNorm/{label}", stats["physics_norm"])
        self.log(f"InitMSE/{label}", stats["init_mse"])

        if batch_idx % 50 == 0:
            x_init = self.model.backbone_reconstruction(y)
            x_hat = self.model.sample(y, steps=self.sample_steps, method="heun")

            roi_mse = mse_loss(
                self.region_of_interest(x_hat),
                self.region_of_interest(x_gt),
            )
            self.log(f"ROI-MSE/{label}", roi_mse)

            lpips = vgg_loss(x_hat, x_gt)
            self.log(f"LPIPS/{label}", lpips)

            self.visualize_predictions(label, x_init, x_hat, x_gt)

        return loss

    def visualize_predictions(self, label, x_init, x_hat, x_gt):
        image = __import__("torch").cat(
            [
                self.region_of_interest(x_gt),
                self.region_of_interest(x_init),
                self.region_of_interest(x_hat),
            ],
            dim=-2,
        )
        self.logger.experiment.add_images(f"Preds/{label}", image, self.global_step)
