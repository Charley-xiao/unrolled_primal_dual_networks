
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import tqdm

from lensless.diffusercam import LenslessLearningCollection, region_of_interest
from lensless.evaluate import EvaluationSystem
from lensless.flow import PhysicsGuidedFlowModel, strip_lightning_prefix
from lensless.flow_training import FlowTrainingSystem
from lensless.model import ImageOptimizer
from lensless.model_colors import ImageOptimizerMixColors

DEFAULT_LOGS_DIR = "logs/"
DEFAULT_RESULTS_DIR = "results/"

BACKBONE_CLASSES = {
    "learned-primal": ImageOptimizer,
    "learned-primal-and-model": ImageOptimizer,
    "learned-primal-dual": ImageOptimizer,
    "learned-primal-dual-and-model": ImageOptimizer,
    "learned-primal-dual-and-five-models": ImageOptimizer,
    "learned-primal-dual-and-color-mixing": ImageOptimizerMixColors,
}

BACKBONE_CONFIGS = {
    "learned-primal": {
        "width": 5,
        "depth": 10,
        "learned_models": 0,
        "primal_only": True,
    },
    "learned-primal-and-model": {
        "width": 5,
        "depth": 10,
        "learned_models": 1,
        "primal_only": True,
    },
    "learned-primal-dual": {
        "width": 5,
        "depth": 10,
        "learned_models": 0,
    },
    "learned-primal-dual-and-model": {
        "width": 5,
        "depth": 10,
        "learned_models": 1,
    },
    "learned-primal-dual-and-five-models": {
        "width": 5,
        "depth": 10,
        "learned_models": 5,
    },
    "learned-primal-dual-and-color-mixing": {
        "depth": 10,
    },
}


def build_backbone(psf: torch.Tensor, name: str):
    cls = BACKBONE_CLASSES[name]
    cfg = BACKBONE_CONFIGS[name]
    return cls(psf, **cfg)


def maybe_load_backbone_checkpoint(backbone, checkpoint_path: str | None):
    if checkpoint_path is None:
        return backbone
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_lightning_prefix(checkpoint["state_dict"], prefix="model.")
    backbone.load_state_dict(state_dict, strict=False)
    return backbone


@torch.no_grad()
def evaluate_and_save(model, dataloader, results_dir: Path, steps: int, method: str):
    results_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for i, (y, x_gt) in enumerate(tqdm.tqdm(dataloader, desc="Evaluating")):
        y = y.cuda()
        x_gt = x_gt.cuda()

        x_init = model.backbone_reconstruction(y)
        x_hat = model.sample(y, steps=steps, method=method)

        for j in range(x_hat.shape[0]):
            idx = i * dataloader.batch_size + j
            torchvision.utils.save_image(x_hat[j], results_dir / f"{idx:05d}_sample.png")
            torchvision.utils.save_image(x_init[j], results_dir / f"{idx:05d}_init.png")
            torchvision.utils.save_image(x_gt[j], results_dir / f"{idx:05d}_gt.png")
            torch.save(
                {
                    "measurement": y[j].cpu(),
                    "x_init": x_init[j].cpu(),
                    "sample": x_hat[j].cpu(),
                    "ground_truth": x_gt[j].cpu(),
                },
                results_dir / f"{idx:05d}.pt",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to the DiffuserCam dataset root.")
    parser.add_argument("--logs-dir", type=str, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--backbone-model",
        type=str,
        default="learned-primal-dual-and-five-models",
        choices=list(BACKBONE_CLASSES),
    )
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-max", type=float, default=0.5)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-power", type=float, default=2.0)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--sample-method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--detach-backbone-condition", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="Flow checkpoint for evaluation.")

    args = parser.parse_args()

    datasets = LenslessLearningCollection(args.dataset)
    train_loader = DataLoader(
        datasets.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    psf = datasets.psf.unsqueeze(0)
    backbone = build_backbone(psf, args.backbone_model)
    backbone = maybe_load_backbone_checkpoint(backbone, args.backbone_checkpoint)

    model = PhysicsGuidedFlowModel(
        psf=psf,
        backbone=backbone,
        lambda_max=args.lambda_max,
        lambda_min=args.lambda_min,
        lambda_power=args.lambda_power,
        hidden_width=args.hidden_width,
        detach_backbone_condition=args.detach_backbone_condition,
    )

    if args.eval:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in --eval mode.")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        wrapper = EvaluationSystem(model, checkpoint)
        wrapper.cuda()
        evaluate_and_save(
            wrapper.model,
            val_loader,
            Path(args.results_dir),
            steps=args.sample_steps,
            method=args.sample_method,
        )
        return

    logger = TensorBoardLogger(save_dir=args.logs_dir, name="physics_guided_flow")
    checkpoint_cb = ModelCheckpoint(
        monitor="Loss/Val",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="flow-{epoch:02d}-{Loss/Val:.4f}",
    )
    system = FlowTrainingSystem(
        model=model,
        region_of_interest=region_of_interest,
        lr=args.lr,
        sample_steps=min(args.sample_steps, 20),
    )

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_cb],
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
    )
    trainer.fit(system, train_loader, val_loader)


if __name__ == "__main__":
    main()
