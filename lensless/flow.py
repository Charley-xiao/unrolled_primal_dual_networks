
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .model import ImageOptimizer
from .operators import LenslessCamera
from .unet import UNet


class MeasurementConsistentOperator(nn.Module):
    """
    Sensor-domain data-consistency operator built on top of the repo's LenslessCamera.

    The original camera expects a padded image, then applies convolution and a crop.
    Here we define a sensor-domain forward model

        H_s(x) = crop( H( pad(x) ) )

    so that x can stay in the same image space as the ground-truth target.
    """

    def __init__(self, psfs: torch.Tensor):
        super().__init__()
        self.psfs = nn.Parameter(psfs, requires_grad=False)

    def _camera(self):
        return LenslessCamera(self.psfs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        camera = self._camera()
        return camera.forward(camera.pad(x))

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        camera = self._camera()
        return camera.crop(camera.adjoint(residual))

    def grad_data_fidelity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        residual = self.forward(x) - y
        return self.adjoint(residual)


class LambdaSchedule(nn.Module):
    """
    Time-dependent guidance strength λ(t).

    By default the physics term is weak early and strong late.
    """

    def __init__(
        self,
        max_value: float = 1.0,
        min_value: float = 0.0,
        power: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_value = float(max_value)
        self.min_value = float(min_value)
        self.power = float(power)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        s = t.clamp(0.0, 1.0).pow(self.power)
        return self.min_value + (self.max_value - self.min_value) * s


class PhysicsGuidedVelocityField(nn.Module):
    """
    Learn the free drift u_theta(x, t, y). The final physics-aware velocity is

        v(x, t, y) = u_theta(x, t, y) - λ(t) H*(H x - y).

    We also condition on x_init = backbone(y), which is a deterministic function of y,
    so the notation u_theta(x, t, y) remains consistent.
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_width: int = 32,
    ) -> None:
        super().__init__()
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=3,
            init_features=hidden_width,
            output_padding=[(1, 0), (1, 0), (1, 0), (0, 0)],
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x_init: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = x_t.shape
        t_map = t.view(b, 1, 1, 1).expand(b, 1, h, w)
        features = torch.cat([x_t, y, x_init, t_map], dim=1)
        return self.unet(features)


class PhysicsGuidedFlowModel(nn.Module):
    """
    End-to-end physics-guided conditional flow matching model.

    The deterministic backbone is reused from the unrolled primal-dual repo.
    The flow stage learns a free velocity u_theta and the physical term is added analytically.
    """

    def __init__(
        self,
        psf: torch.Tensor,
        backbone: nn.Module,
        lambda_max: float = 0.5,
        lambda_min: float = 0.0,
        lambda_power: float = 2.0,
        hidden_width: int = 32,
        detach_backbone_condition: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.operator = MeasurementConsistentOperator(psf)
        self.lambda_schedule = LambdaSchedule(
            max_value=lambda_max,
            min_value=lambda_min,
            power=lambda_power,
        )
        self.velocity_field = PhysicsGuidedVelocityField(
            in_channels=10,
            hidden_width=hidden_width,
        )
        self.detach_backbone_condition = detach_backbone_condition

    def backbone_reconstruction(self, y: torch.Tensor) -> torch.Tensor:
        if isinstance(self.backbone, ImageOptimizer):
            x_init = self.backbone(y, denoise=False)
        else:
            x_init = self.backbone(y, denoise=False)
        if self.detach_backbone_condition:
            x_init = x_init.detach()
        return x_init

    def free_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_init is None:
            x_init = self.backbone_reconstruction(y)
        return self.velocity_field(x_t, t, y, x_init)

    def physics_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        x_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if u is None:
            u = self.free_velocity(x_t, t, y, x_init=x_init)
        physics_grad = self.operator.grad_data_fidelity(x_t, y)
        lam = self.lambda_schedule(t).view(-1, 1, 1, 1)
        return u - lam * physics_grad

    def cfm_loss(
        self,
        y: torch.Tensor,
        x1: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Train u_theta so that the final velocity
            v = u_theta - λ(t) H*(H x_t - y)
        matches the standard straight-path CFM target x1 - x0.
        Therefore the target for u_theta is
            (x1 - x0) + λ(t) H*(H x_t - y).
        """
        x0 = noise_scale * torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        t_view = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * x1

        x_init = self.backbone_reconstruction(y)
        physics_grad = self.operator.grad_data_fidelity(x_t, y)
        lam = self.lambda_schedule(t).view(-1, 1, 1, 1)

        u_pred = self.free_velocity(x_t, t, y, x_init=x_init)
        target_u = (x1 - x0) + lam * physics_grad

        loss = F.mse_loss(u_pred, target_u)
        with torch.no_grad():
            v_pred = u_pred - lam * physics_grad
            v_target = (x1 - x0)
            velocity_mse = F.mse_loss(v_pred, v_target)
            physics_norm = physics_grad.square().mean()
            init_mse = F.mse_loss(x_init, x1)

        return {
            "loss": loss,
            "u_mse": loss.detach(),
            "v_mse": velocity_mse,
            "physics_norm": physics_norm,
            "init_mse": init_mse,
        }

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        steps: int = 40,
        method: str = "heun",
        noise_scale: float = 1.0,
        clamp: bool = True,
    ) -> torch.Tensor:
        x = noise_scale * torch.randn_like(y)
        x_init = self.backbone_reconstruction(y)

        ts = torch.linspace(0.0, 1.0, steps + 1, device=y.device)
        for i in range(steps):
            t0 = ts[i].expand(y.shape[0])
            t1 = ts[i + 1].expand(y.shape[0])
            dt = (t1 - t0).view(-1, 1, 1, 1)

            u0 = self.free_velocity(x, t0, y, x_init=x_init)
            v0 = self.physics_velocity(x, t0, y, u=u0, x_init=x_init)

            if method.lower() == "euler":
                x = x + dt * v0
            elif method.lower() == "heun":
                x_euler = x + dt * v0
                u1 = self.free_velocity(x_euler, t1, y, x_init=x_init)
                v1 = self.physics_velocity(x_euler, t1, y, u=u1, x_init=x_init)
                x = x + 0.5 * dt * (v0 + v1)
            else:
                raise ValueError(f"Unknown method: {method}")

        if clamp:
            x = x.clamp(0.0, 1.0)
        return x

    @torch.no_grad()
    def reconstruct(
        self,
        y: torch.Tensor,
        steps: int = 40,
        method: str = "heun",
    ) -> torch.Tensor:
        return self.sample(y, steps=steps, method=method)


def strip_lightning_prefix(state_dict: dict[str, torch.Tensor], prefix: str = "model.") -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = value
        else:
            out[key] = value
    return out
